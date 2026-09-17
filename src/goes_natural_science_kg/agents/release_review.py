# src/goes_natural_science_kg/agents/release_review.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify editorial proposals once with scoped evidence and corrected specialist rubrics.
from __future__ import annotations

import asyncio
from pathlib import Path

from google import genai
from google.genai import types

from goes_natural_science_kg.agents.release import design_errors, panel_accepts, transport_schema
from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.eval.experiment import evaluate_request
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.release import (
    FinalProposalReview,
    ReleaseAssignmentSummary,
    ReleaseJudgePacket,
    ReleasePanel,
    ReleaseSettings,
    ReleaseUnitResult,
    ReleaseVerdict,
)


def judge_packet(result: ReleaseUnitResult, criterion: str) -> ReleaseJudgePacket:
    """Expose only evidence relevant to the criterion; include quoted-page context for entailment."""
    if result.final is None:
        raise ValueError("no proposal to review")
    assignment, design = result.assignment, result.final
    cited = {
        s.anchor_key for skill in design.skills for micro in skill.micros for s in micro.support
    }
    pages = {(a.source_id, a.page) for a in assignment.anchors if a.key in cited}
    evidence = (
        assignment.anchors
        if criterion == "curricular"
        else tuple(a for a in assignment.anchors if (a.source_id, a.page) in pages)
        if criterion == "evidence"
        else ()
    )
    return ReleaseJudgePacket(
        assignment=ReleaseAssignmentSummary(**assignment.model_dump(exclude={"anchors"})),
        proposal=design,
        evidence=evidence,
        deterministic_errors=design_errors(design, assignment),
    )


async def review_proposals(
    results: tuple[ReleaseUnitResult, ...],
    settings: ReleaseSettings,
    output: Path,
    *,
    online: bool = False,
) -> tuple[FinalProposalReview, ...]:
    """A single final verification stage; it never resets or extends optimizer revision limits."""
    client = (
        genai.Client(
            vertexai=True,
            project=settings.project,
            location="global",
            http_options=types.HttpOptions(timeout=240000),
        )
        if online
        else None
    )
    registry = load_registry(Path("prompts"))
    limiter = asyncio.Semaphore(settings.concurrency)
    locks: dict[str, asyncio.Lock] = {}

    async def judge(result: ReleaseUnitResult, criterion: str) -> tuple[ReleaseVerdict, str]:
        packet = judge_packet(result, criterion)
        for attempt in range(4):
            record = await evaluate_request(
                client,
                limiter,
                output / "responses",
                settings,
                registry,
                "release-judge-" + criterion,
                packet.model_dump(mode="json"),
                settings.judge_models[criterion],
                ReleaseVerdict,
                0,
                0,
                locks,
                prompt_version="1.1.0",
                location="global",
                schema_override=transport_schema(ReleaseVerdict.model_json_schema()),
                retry_attempt=attempt,
            )
            if record.status == "ok" and record.response is not None:
                verdict = ReleaseVerdict.model_validate_json(record.response)
                if verdict.criterion != criterion:
                    raise ValueError("judge returned a different criterion")
                return verdict, record.request_sha256
            if attempt == 3 or not any(
                code in (record.error or "") for code in ("429", "503", "504", "Timeout")
            ):
                raise RuntimeError("provider review failed: " + record.request_sha256)
            await asyncio.sleep(2 ** (attempt + 2))
        raise RuntimeError("unreachable retry state")

    async def unit(result: ReleaseUnitResult) -> FinalProposalReview:
        if result.final is None:
            raise ValueError("missing final unit")
        errors = design_errors(result.final, result.assignment)
        observed = (
            await asyncio.gather(
                *(judge(result, criterion) for criterion in settings.judge_models),
                return_exceptions=True,
            )
            if not errors
            else []
        )
        failures = tuple(str(item) for item in observed if isinstance(item, BaseException))
        valid = [item for item in observed if not isinstance(item, BaseException)]
        votes = tuple(v for v, _ in valid)
        review = FinalProposalReview(
            unit_key=result.assignment.key,
            candidate_sha256=content_hash(result.final),
            panel=ReleasePanel(
                revision=0,
                model_by_criterion=settings.judge_models,
                verdicts=votes,
                deterministic_errors=errors,
                accepted=panel_accepts(votes, errors, settings),
                request_ids=tuple(key for _, key in valid),
            ),
            errors=failures,
        )
        atomic_bytes(
            output / "units" / (result.assignment.key + ".json"),
            (canonical_json(review) + "\n").encode(),
        )
        print(
            result.assignment.key + ": final verification " + str(review.panel.accepted), flush=True
        )
        return review

    try:
        return tuple(await asyncio.gather(*(unit(r) for r in results)))
    finally:
        if client:
            await client.aio.aclose()
