# src/goes_natural_science_kg/eval/harness.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

from google import genai

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.eval.cases import ROLES, case_payload, load_cases
from goes_natural_science_kg.eval.experiment import evaluate_request
from goes_natural_science_kg.eval.metrics import score_output
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.base import Contract, canonical_json, content_hash
from goes_natural_science_kg.schemas.orchestration import CurriculumProposal, Decomposition, Verdict
from goes_natural_science_kg.schemas.prompt_evaluation import (
    AnnotatedCase,
    BatchOutput,
    EvaluationCell,
    EvaluationMetric,
    ExperimentSettings,
    PromptRegistry,
    ReferenceReview,
)

OUTPUTS: dict[str, type[Contract]] = {
    "decomposition": Decomposition,
    "curricularization": CurriculumProposal,
}


async def run_cell(  # noqa: C901 - bounded batch revision loop keeps per-case failures together
    client: genai.Client | None,
    limiter: asyncio.Semaphore,
    settings: ExperimentSettings,
    registry: PromptRegistry,
    cases: tuple[AnnotatedCase, ...],
    role: str,
    technique: str,
    replicate: int,
    cache: Path,
    output_dir: Path,
    negative_ids: set[str],
    request_locks: dict[str, asyncio.Lock] | None = None,
) -> None:
    if request_locks is None:
        request_locks = {}
    prompt_id = role.replace("_", "-") + "-" + technique.replace("_", "-")
    cell_id = cell_identity(cases, registry, prompt_id, replicate, settings)
    result_path = output_dir / (cell_id + ".json")
    if result_path.exists():
        stored = EvaluationCell.model_validate_json(result_path.read_bytes())
        if stored.id != cell_id:
            raise ValueError("cell identity mismatch")
        validate_cell_prompts(stored, registry, cache)
        return
    contract = OUTPUTS.get(role, Verdict)
    batch_contract: Any = BatchOutput[contract]  # type: ignore[valid-type]
    model = settings.generator_model if role in OUTPUTS else settings.judge_model
    pending = {c.id: c for c in cases}
    history = []
    metrics = []
    feedback: dict[str, Any] = {}
    for revision in range(settings.max_revisions + 1):
        payload = {
            "cases": [case_payload(c, role, c.id in negative_ids) for c in pending.values()],
            "feedback": feedback,
        }
        observation = await evaluate_request(
            client,
            limiter,
            cache,
            settings,
            registry,
            prompt_id,
            payload,
            model,
            batch_contract,
            replicate,
            revision,
            request_locks,
        )
        history.append(observation.request_sha256)
        try:
            batch = batch_contract.model_validate_json(observation.response or "")
            outputs = {r.case_id: r.output for r in batch.results}
            if len(outputs) != len(batch.results) or set(outputs) != set(pending):
                raise ValueError("batch result IDs differ from input IDs")
        except ValueError:
            outputs = {}
        reviews = {}
        if role in OUTPUTS and outputs:
            review_payload = {
                "cases": [
                    {
                        "case_id": key,
                        "expected": pending[key].expected.model_dump(mode="json"),
                        "candidate": value.model_dump(mode="json"),
                        "evidence": [p.model_dump(mode="json") for p in pending[key].evidence],
                    }
                    for key, value in outputs.items()
                ]
            }
            reviewed = await evaluate_request(
                client,
                limiter,
                cache,
                settings,
                registry,
                "reference-review" if role == "decomposition" else "curriculum-reference-review",
                review_payload,
                settings.judge_model,
                BatchOutput[ReferenceReview],
                replicate,
                revision,
                request_locks,
            )
            history.append(reviewed.request_sha256)
            try:
                reviews = {
                    r.case_id: r.output
                    for r in BatchOutput[ReferenceReview]
                    .model_validate_json(reviewed.response or "")
                    .results
                }
            except ValueError:
                reviews = {}
        remaining = {}
        feedback = {}
        for key, case in pending.items():
            values: dict[str, Any] = {
                "coverage": None,
                "prerequisite_precision": None,
                "prerequisite_recall": None,
                "invalid_reference_rate": None,
                "unsupported_claim_rate": None,
                "judge_correct": None,
                "passed": False,
                "notes": (),
            }
            valid = key in outputs
            try:
                if valid:
                    values.update(
                        score_output(
                            case, role, outputs[key], reviews.get(key), key in negative_ids
                        )
                    )
            except ValueError as error:
                values["notes"] = (str(error),)
            metric = EvaluationMetric(
                case_id=key,
                role=role,
                technique=technique,
                replicate=replicate,
                schema_valid=valid,
                revisions=revision,
                censored=revision == settings.max_revisions and not values["passed"],
                **values,
            )
            metrics.append(metric.model_dump(mode="json"))
            if not metric.passed and revision < settings.max_revisions:
                remaining[key] = case
                feedback[key] = {
                    "previous": outputs[key].model_dump(mode="json") if key in outputs else None,
                    "errors": metric.notes,
                    "schema_valid": valid,
                    "coverage_below_threshold": metric.coverage is not None
                    and metric.coverage < settings.coverage_threshold,
                    "prerequisites_correct": metric.prerequisite_recall == 1.0
                    and metric.prerequisite_precision == 1.0,
                    "unsupported_claims": metric.unsupported_claim_rate,
                    "judge_classification_correct": metric.judge_correct,
                }
        pending = remaining
        if not pending:
            break
    result = {
        "schema_version": "evaluation-cell/1.0",
        "id": cell_id,
        "case_ids": [c.id for c in cases],
        "role": role,
        "technique": technique,
        "replicate": replicate,
        "observations": history,
        "metrics": metrics,
    }
    validated = EvaluationCell.model_validate(result)
    atomic_bytes(result_path, (canonical_json(validated) + "\n").encode())
    print(role, technique, replicate, "finished", len(cases), "cases", flush=True)


async def run_experiment(
    cases_path: Path,
    prompts: Path,
    settings: ExperimentSettings,
    cache: Path,
    output_dir: Path,
    *,
    online: bool = False,
) -> None:
    cases = tuple(c for c in load_cases(cases_path) if c.split == "evaluation")
    if len(cases) != 40:
        raise ValueError("the registered comparison requires exactly 40 held-out cases")
    registry = load_registry(prompts)
    negative_ids = {c.id for i, c in enumerate(cases) if i % 2}
    client = (
        genai.Client(
            vertexai=True,
            project=settings.project,
            location=settings.location,
            http_options={"timeout": 180000, "retry_options": {"attempts": 3}},
        )
        if online
        else None
    )
    limiter = asyncio.Semaphore(settings.concurrency)
    jobs = []
    request_locks: dict[str, asyncio.Lock] = {}
    for role in ROLES:
        for technique in ("cot", "few_shot", "cot_few_shot"):
            for replicate in range(settings.replicates):
                ordered = list(cases)
                # Paired experimental assignment requires a reproducible, non-security seed.
                random.Random(settings.seed + replicate).shuffle(ordered)  # nosec B311
                for first in range(0, len(ordered), settings.batch_size):
                    jobs.append(
                        run_cell(
                            client,
                            limiter,
                            settings,
                            registry,
                            tuple(ordered[first : first + settings.batch_size]),
                            role,
                            technique,
                            replicate,
                            cache,
                            output_dir,
                            negative_ids,
                            request_locks,
                        )
                    )
    try:
        await asyncio.gather(*jobs)
    finally:
        if client:
            await client.aio.aclose()


def cell_identity(
    cases: tuple[AnnotatedCase, ...],
    registry: PromptRegistry,
    prompt_id: str,
    replicate: int,
    settings: ExperimentSettings,
) -> str:
    return content_hash(
        {
            "cases": [c.model_dump(mode="json") for c in cases],
            "scorer_version": "1.1.0",
            "prompt": registry.get(prompt_id, "1.0.0").sha256,
            "replicate": replicate,
            "settings": settings.model_dump(mode="json"),
        }
    )


def validate_cell_prompts(cell: EvaluationCell, registry: PromptRegistry, cache: Path) -> None:
    from goes_natural_science_kg.eval.experiment import read_observation

    for key in cell.observations:
        record = read_observation(cache, key)
        artifact = registry.get(record.request["prompt_id"], record.request["prompt_version"])
        if artifact.sha256 != record.request["prompt_sha256"]:
            raise ValueError(
                "a measured generator or evaluator prompt has changed; rerun the experiment"
            )
