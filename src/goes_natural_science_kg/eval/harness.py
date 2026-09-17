# src/goes_natural_science_kg/eval/harness.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, cast

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
    CellSpec,
    EvaluationCell,
    EvaluationMetric,
    ExperimentArm,
    ExperimentSettings,
    FollowUpCell,
    FollowUpPlan,
    PromptRegistry,
    PromptVariantRef,
    ReferenceReview,
    Technique,
)

OUTPUTS: dict[str, type[Contract]] = {
    "decomposition": Decomposition,
    "curricularization": CurriculumProposal,
}


async def run_cell(
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
    """Historical three-technique cell; identity and request bytes are unchanged."""
    prompt_id = role.replace("_", "-") + "-" + technique.replace("_", "-")
    spec = CellSpec(
        role=role,
        technique=cast(Technique, technique),
        prompt=PromptVariantRef(prompt_id=prompt_id, version="1.0.0"),
        generator_model=settings.generator_model,
        generator_location=settings.location,
        judge_model=settings.judge_model,
        judge_location=settings.location,
    )
    await run_spec_cell(
        {settings.location: client} if client is not None else None,
        limiter,
        settings,
        registry,
        cases,
        spec,
        replicate,
        cache,
        output_dir,
        negative_ids,
        request_locks if request_locks is not None else {},
        cell_id=cell_identity(cases, registry, prompt_id, replicate, settings),
        cell_model=EvaluationCell,
        cell_extra={},
    )


async def run_spec_cell(  # noqa: C901 - bounded batch revision loop keeps per-case failures together
    clients: dict[str, genai.Client] | None,
    limiter: asyncio.Semaphore,
    settings: ExperimentSettings | FollowUpPlan,
    registry: PromptRegistry,
    cases: tuple[AnnotatedCase, ...],
    spec: CellSpec,
    replicate: int,
    cache: Path,
    output_dir: Path,
    negative_ids: set[str],
    request_locks: dict[str, asyncio.Lock],
    *,
    cell_id: str,
    cell_model: type[EvaluationCell],
    cell_extra: dict[str, Any],
) -> None:
    result_path = output_dir / (cell_id + ".json")
    if result_path.exists():
        stored = cell_model.model_validate_json(result_path.read_bytes())
        if stored.id != cell_id:
            raise ValueError("cell identity mismatch")
        validate_cell_prompts(stored, registry, cache)
        return
    role, technique = spec.role, spec.technique
    contract = OUTPUTS.get(role, Verdict)
    batch_contract: Any = BatchOutput[contract]  # type: ignore[valid-type]
    model = spec.generator_model if role in OUTPUTS else spec.judge_model
    model_location = spec.generator_location if role in OUTPUTS else spec.judge_location
    generator_client = clients.get(model_location) if clients else None
    judge_client = clients.get(spec.judge_location) if clients else None
    if clients is not None and (generator_client is None or judge_client is None):
        raise ValueError("no provider client for the requested Vertex location")
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
            generator_client,
            limiter,
            cache,
            settings,
            registry,
            spec.prompt.prompt_id,
            payload,
            model,
            batch_contract,
            replicate,
            revision,
            request_locks,
            prompt_version=spec.prompt.version,
            location=model_location,
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
                judge_client,
                limiter,
                cache,
                settings,
                registry,
                "reference-review" if role == "decomposition" else "curriculum-reference-review",
                review_payload,
                spec.judge_model,
                BatchOutput[ReferenceReview],
                replicate,
                revision,
                request_locks,
                location=spec.judge_location,
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
        "schema_version": cell_model.model_fields["schema_version"].default,
        "id": cell_id,
        "case_ids": [c.id for c in cases],
        "role": role,
        "technique": technique,
        "replicate": replicate,
        "observations": history,
        "metrics": metrics,
        **cell_extra,
    }
    validated = cell_model.model_validate(result)
    atomic_bytes(result_path, (canonical_json(validated) + "\n").encode())
    print(
        role, spec.prompt.prompt_id, model, replicate, "finished", len(cases), "cases", flush=True
    )


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


def followup_cell_identity(
    cases: tuple[AnnotatedCase, ...],
    registry: PromptRegistry,
    plan: FollowUpPlan,
    arm: ExperimentArm,
    replicate: int,
) -> str:
    return content_hash(
        {
            "cases": [c.model_dump(mode="json") for c in cases],
            "scorer_version": "1.1.0",
            "prompt": registry.get(arm.prompt.prompt_id, arm.prompt.version).sha256,
            "replicate": replicate,
            "plan": plan.model_dump(mode="json"),
            "arm": arm.key,
        }
    )


def arm_spec(plan: FollowUpPlan, arm: ExperimentArm, registry: PromptRegistry) -> CellSpec:
    artifact = registry.get(arm.prompt.prompt_id, arm.prompt.version)
    if artifact.metadata.role != plan.role:
        raise ValueError("arm prompt role differs from the plan role")
    return CellSpec(
        role=plan.role,
        technique=artifact.metadata.technique,
        prompt=arm.prompt,
        generator_model=arm.generator_model,
        generator_location=arm.location,
        judge_model=plan.judge_model,
        judge_location=plan.judge_location,
    )


def evaluation_cases(cases_path: Path) -> tuple[AnnotatedCase, ...]:
    cases = tuple(c for c in load_cases(cases_path) if c.split == "evaluation")
    if len(cases) != 40:
        raise ValueError("the registered comparison requires exactly 40 held-out cases")
    return cases


async def run_followup(
    plan: FollowUpPlan,
    cases_path: Path,
    prompts: Path,
    cache: Path,
    output_dir: Path,
    *,
    online: bool = False,
    credentials: Any = None,
) -> None:
    """Run every arm on the same paired batches; identical requests reuse recorded observations.

    ``credentials`` optionally injects a google-auth credential object (a transport resource,
    never persisted); by default Application Default Credentials are used.
    """
    cases = evaluation_cases(cases_path)
    unknown = set(plan.tuning_case_ids) - {c.id for c in cases}
    if unknown:
        raise ValueError(
            "tuning case ids outside the evaluation split: " + ", ".join(sorted(unknown))
        )
    registry = load_registry(prompts)
    negative_ids = {c.id for i, c in enumerate(cases) if i % 2}
    locations: set[str] = {plan.judge_location, *(a.location for a in plan.arms)}
    clients: dict[str, genai.Client] | None = (
        {
            location: genai.Client(
                vertexai=True,
                project=plan.project,
                location=location,
                credentials=credentials,
                http_options={"timeout": 180000, "retry_options": {"attempts": 3}},
            )
            for location in sorted(locations)
        }
        if online
        else None
    )
    limiter = asyncio.Semaphore(plan.concurrency)
    request_locks: dict[str, asyncio.Lock] = {}
    plan_sha256 = content_hash(plan)
    jobs = []
    for arm in plan.arms:
        spec = arm_spec(plan, arm, registry)
        for replicate in range(plan.replicates):
            ordered = list(cases)
            # Paired assignment shares the historical shuffle so cached requests are reused.
            random.Random(plan.seed + replicate).shuffle(ordered)  # nosec B311
            for first in range(0, len(ordered), plan.batch_size):
                batch = tuple(ordered[first : first + plan.batch_size])
                jobs.append(
                    run_spec_cell(
                        clients,
                        limiter,
                        plan,
                        registry,
                        batch,
                        spec,
                        replicate,
                        cache,
                        output_dir,
                        negative_ids,
                        request_locks,
                        cell_id=followup_cell_identity(batch, registry, plan, arm, replicate),
                        cell_model=FollowUpCell,
                        cell_extra={
                            "arm": arm.key,
                            "spec": spec.model_dump(mode="json"),
                            "plan_sha256": plan_sha256,
                        },
                    )
                )
    try:
        await asyncio.gather(*jobs)
    finally:
        for client in (clients or {}).values():
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
