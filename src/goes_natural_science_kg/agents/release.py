# src/goes_natural_science_kg/agents/release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Produce resumable proposal candidates with evidence-bound specialist review.
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import ValidationError
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.eval.experiment import evaluate_request
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.base import Contract, canonical_json, content_hash
from goes_natural_science_kg.schemas.release import (
    ReleaseActivityBank,
    ReleasePanel,
    ReleaseRuntime,
    ReleaseSettings,
    ReleaseState,
    ReleaseUnitResult,
    ReleaseVerdict,
    UnitAssignment,
    UnitDesign,
    UnitPatch,
    UnitSkillDesign,
)


def design_errors(design: UnitDesign, assignment: UnitAssignment) -> tuple[str, ...]:  # noqa: C901
    errors = []
    if design.unit_key != assignment.key:
        errors.append("unit identity changed")
    micros = [m for skill in design.skills for m in skill.micros]
    keys = {m.key for m in micros}
    anchors = {a.key: a for a in assignment.anchors}
    if len(keys) != len(micros) or len({s.key for s in design.skills}) != len(design.skills):
        errors.append("duplicate skill or micro-skill keys")
    if len({a.key for a in design.activities}) != len(design.activities):
        errors.append("duplicate activity keys")
    for m in micros:
        if (
            m.verb.lower() in {"knows", "understands", "learns", "recognizes"}
            or len(m.verb.split()) != 1
        ):
            errors.append("non-observable or compound verb: " + m.key)
        if not set(m.prerequisites) <= keys or m.key in m.prerequisites:
            errors.append("invalid prerequisite: " + m.key)
        for binding in m.support:
            if (
                binding.anchor_key not in anchors
                or binding.quote not in anchors[binding.anchor_key].text
            ):
                errors.append("unverified exact quotation: " + m.key + ":" + binding.anchor_key)
        initial = [a for a in design.activities if a.kind == "teach" and m.key in a.micro_keys]
        if (
            len(initial) != 1
            or initial[0].micro_keys != (m.key,)
            or initial[0].minutes < m.estimated_minutes
        ):
            errors.append(
                "initial teaching must be unique, atomic and sufficiently timed: " + m.key
            )
    for a in design.activities:
        if not set(a.micro_keys) <= keys:
            errors.append("unknown activity micro-skill: " + a.key)
        if a.minutes % 5:
            errors.append("activity minutes must use five-minute planning increments: " + a.key)
    positions = {key: i for i, key in enumerate(sorted(keys))}
    pairs = [(positions[p], positions[m.key]) for m in micros for p in m.prerequisites if p in keys]
    if pairs:
        rows, columns = np.asarray(pairs).T
        matrix = csr_matrix((np.ones(len(pairs)), (rows, columns)), shape=(len(keys), len(keys)))
        _, labels = connected_components(matrix, directed=True, connection="strong")
        if np.any(np.bincount(labels) > 1) or np.any(matrix.diagonal()):
            errors.append("prerequisite cycle")
    if sum(a.minutes for a in design.activities) < assignment.weeks * 300:
        errors.append("insufficient authored bank for the declared unit allocation")
    return tuple(errors)


def panel_accepts(
    verdicts: tuple[ReleaseVerdict, ...], errors: tuple[str, ...], settings: ReleaseSettings
) -> bool:
    if (
        errors
        or {v.criterion for v in verdicts} != set(settings.judge_models)
        or len(verdicts) != 5
    ):
        return False
    votes = {
        v.criterion: v.passed
        and v.score >= settings.score_threshold
        and not any(i.severity in {"critical", "major"} for i in v.issues)
        for v in verdicts
    }
    return (
        votes["evidence"]
        and votes["age"]
        and sum(votes.values()) >= settings.minimum_votes
        and not any(i.severity == "critical" for v in verdicts for i in v.issues)
    )


def transport_schema(value: Any) -> Any:
    """Reduce provider grammar complexity; Pydantic remains the acceptance contract."""
    omitted = {
        "title",
        "default",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "propertyNames",
    }
    if isinstance(value, dict):
        return {k: transport_schema(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, list):
        return [transport_schema(v) for v in value]
    return value


async def request(
    runtime: ReleaseRuntime,
    prompt: str,
    payload: dict[str, Any],
    model: str,
    contract: type[Contract],
    revision: int,
) -> tuple[Contract, str]:
    for attempt in range(4):
        observation = await evaluate_request(
            runtime.clients.get("global") if runtime.clients else None,
            runtime.limiter,
            runtime.cache,
            runtime.settings,
            runtime.registry,
            prompt,
            {**payload, "output_contract": contract.model_json_schema()},
            model,
            contract,
            0,
            revision,
            runtime.request_locks,
            location="global",
            schema_override=transport_schema(contract.model_json_schema()),
            retry_attempt=attempt,
        )
        transient = any(
            code in (observation.error or "")
            for code in ("429", "503", "504", "ReadTimeout", "TimeoutError")
        )
        if observation.status == "ok" and observation.response is not None:
            return contract.model_validate_json(observation.response), observation.request_sha256
        if not transient or attempt == 3:
            raise RuntimeError("provider request failed: " + observation.request_sha256)
        await asyncio.sleep(2 ** (attempt + 2))
    raise RuntimeError("unreachable retry state")


def apply_patch(design: UnitDesign, patch: UnitPatch) -> UnitDesign:
    """Apply explicit editorial replacements without erasing unaffected activities."""
    if patch.unit_key != design.unit_key:
        raise ValueError("patch changes unit identity")
    skills = {s.key: s for s in design.skills}
    if not {s.key for s in patch.skills} <= set(skills):
        raise ValueError("patch introduces an unknown parent skill")
    skills.update({s.key: s for s in patch.skills})
    activities = {a.key: a for a in design.activities}
    if not set(patch.remove_activity_keys) <= set(activities):
        raise ValueError("patch removes an unknown activity")
    for key in patch.remove_activity_keys:
        del activities[key]
    activities.update({a.key: a for a in patch.activities})
    return UnitDesign(
        unit_key=design.unit_key,
        skills=tuple(skills.values()),
        activities=tuple(activities.values()),
        progression=patch.progression or design.progression,
        limitations=patch.limitations if patch.limitations is not None else design.limitations,
    )


async def run_unit(state: ReleaseState) -> dict[str, Any]:  # noqa: C901 - bounded generate/check/review/revise state machine
    runtime, assignment = state["runtime"], state["assignment"]
    identity = content_hash(
        {
            "implementation": content_hash(Path(__file__).read_text()),
            "assignment": assignment.model_dump(mode="json"),
            "settings": runtime.settings.model_dump(mode="json"),
            "prompts": [
                a.sha256 for a in runtime.registry.artifacts if a.metadata.id.startswith("release-")
            ],
        }
    )
    path = Path(runtime.output) / "units" / (identity + ".json")
    if path.exists():
        cached = ReleaseUnitResult.model_validate_json(path.read_bytes())
        if cached.assignment != assignment:
            raise ValueError("unit checkpoint mismatch")
        return {"results": [cached]}
    initial = final = None
    panels: list[ReleasePanel] = []
    ids: list[str] = []
    failures: list[str] = []
    feedback: dict[str, Any] = {}
    revision = 0
    for revision in range(runtime.settings.max_revisions + 1):
        payload = {
            "assignment": assignment.model_dump(mode="json"),
            "contexts": runtime.contexts,
            "previous": final.model_dump(mode="json") if final else None,
            "feedback": feedback,
        }
        try:
            if revision == 0:
                draft, key = await request(
                    runtime,
                    "release-skills",
                    payload,
                    runtime.settings.generator_model,
                    UnitSkillDesign,
                    revision,
                )
                ids.append(key)
                skill_design = UnitSkillDesign.model_validate(draft)
                bank, key = await request(
                    runtime,
                    "release-activities",
                    {
                        "proposal": skill_design.model_dump(mode="json"),
                        "grade": assignment.grade,
                        "weeks": assignment.weeks,
                        "minimum_bank_minutes": assignment.weeks * 345,
                        "maximum_bank_minutes": assignment.weeks * 420,
                        "contexts": runtime.contexts,
                    },
                    runtime.settings.generator_model,
                    ReleaseActivityBank,
                    revision,
                )
                ids.append(key)
                activity_bank = ReleaseActivityBank.model_validate(bank)
                if activity_bank.unit_key != assignment.key:
                    raise ValueError("activity bank unit changed")
                final = UnitDesign(**skill_design.model_dump(), activities=activity_bank.activities)
            else:
                response, key = await request(
                    runtime,
                    "release-patch" if final else "release-optimize",
                    payload,
                    runtime.settings.optimizer_model,
                    UnitPatch if final else UnitDesign,
                    revision,
                )
                ids.append(key)
                final = (
                    apply_patch(final, UnitPatch.model_validate(response))
                    if final
                    else UnitDesign.model_validate(response)
                )
            if initial is None:
                initial = final
            errors = design_errors(final, assignment)
            if errors:
                panel = ReleasePanel(
                    revision=revision,
                    model_by_criterion={},
                    verdicts=(),
                    deterministic_errors=errors,
                    accepted=False,
                    request_ids=(),
                )
                panels.append(panel)
                feedback = panel.model_dump(mode="json")
                continue
            judge_payload = {
                "assignment": assignment.model_dump(mode="json"),
                "proposal": final.model_dump(mode="json"),
                "deterministic_errors": errors,
            }
            judged = await asyncio.gather(
                *(
                    request(
                        runtime,
                        "release-judge-" + criterion,
                        judge_payload,
                        model,
                        ReleaseVerdict,
                        revision,
                    )
                    for criterion, model in runtime.settings.judge_models.items()
                ),
                return_exceptions=True,
            )
            verdicts = []
            round_ids = []
            for item in judged:
                if isinstance(item, BaseException):
                    failures.append(str(item))
                else:
                    verdicts.append(ReleaseVerdict.model_validate(item[0]))
                    round_ids.append(item[1])
            ids.extend(round_ids)
            accepted = panel_accepts(tuple(verdicts), errors, runtime.settings)
            panel = ReleasePanel(
                revision=revision,
                model_by_criterion=runtime.settings.judge_models,
                verdicts=tuple(verdicts),
                deterministic_errors=errors,
                accepted=accepted,
                request_ids=tuple(round_ids),
            )
            panels.append(panel)
            feedback = panel.model_dump(mode="json")
            if accepted:
                break
        except Exception as error:
            detail = (
                canonical_json(error.errors(include_url=False, include_input=False))
                if isinstance(error, ValidationError)
                else str(error)[:1000]
            )
            failures.append(type(error).__name__ + ": " + detail)
            feedback = {"failure": failures[-1]}
    result = ReleaseUnitResult(
        assignment=assignment,
        initial=initial,
        final=final,
        panels=tuple(panels),
        revisions=revision,
        status="panel_passed"
        if panels and panels[-1].accepted
        else "needs_human_review"
        if final
        else "failed",
        request_ids=tuple(ids),
        errors=tuple(failures),
    )
    atomic_bytes(path, (canonical_json(result) + "\n").encode())
    print(assignment.key + ": " + result.status + " revisions=" + str(revision), flush=True)
    return {"results": [result]}


def dispatch(state: ReleaseState) -> list[Send]:
    return [
        Send("unit", {"runtime": state["runtime"], "assignment": a})
        for a in state["runtime"].assignments
    ]


async def run_release(
    assignments: tuple[UnitAssignment, ...],
    settings: ReleaseSettings,
    contexts: tuple[dict[str, Any], ...],
    output: Path,
    cache: Path,
    online: bool,
) -> tuple[ReleaseUnitResult, ...]:
    clients = (
        {
            "global": genai.Client(
                vertexai=True,
                project=settings.project,
                location="global",
                http_options=types.HttpOptions(timeout=240000),
            )
        }
        if online
        else {}
    )
    runtime = ReleaseRuntime(
        assignments=assignments,
        settings=settings,
        contexts=contexts,
        cache=cache,
        output=output,
        clients=clients,
        registry=load_registry(Path("prompts")),
        limiter=asyncio.Semaphore(settings.concurrency),
        request_locks={},
    )
    graph = StateGraph(ReleaseState)
    graph.add_node("unit", run_unit)
    graph.add_conditional_edges(START, dispatch)
    graph.add_edge("unit", END)
    try:
        state = await graph.compile().ainvoke(
            {"runtime": runtime, "results": []}, config={"max_concurrency": settings.concurrency}
        )
    finally:
        for client in clients.values():
            await client.aio.aclose()
    return tuple(sorted(state["results"], key=lambda r: r.assignment.key))
