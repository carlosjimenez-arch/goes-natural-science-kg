# src/goes_natural_science_kg/eval/release_audit.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Publish measured provider and review outcomes without exposing source-bearing prompts.
from pathlib import Path

from goes_natural_science_kg.eval.experiment import read_observation
from goes_natural_science_kg.schemas.release import (
    ProviderCallSummary,
    ReleaseRunAudit,
    ReleaseUnitResult,
)


def release_audit(
    results: tuple[ReleaseUnitResult, ...], caches: tuple[Path, ...]
) -> ReleaseRunAudit:
    calls = {}
    for cache in caches:
        for path in sorted(cache.glob("*.json")):
            record = read_observation(cache, path.stem)
            calls[record.request_sha256] = ProviderCallSummary(
                request_sha256=record.request_sha256,
                response_sha256=record.response_sha256,
                prompt_id=record.request["prompt_id"],
                prompt_version=record.request["prompt_version"],
                model=record.request["model"],
                model_version=record.model_version,
                status=record.status,
                error=record.error,
                latency_seconds=record.latency_seconds,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                estimated_usd=record.estimated_usd,
                transport_retry=record.request.get("retry_attempt", 0),
            )
    verified = 0
    for result in results:
        anchors = {a.key: a for a in result.assignment.anchors}
        if result.final:
            verified += sum(
                s.anchor_key in anchors and s.quote in anchors[s.anchor_key].text
                for skill in result.final.skills
                for micro in skill.micros
                for s in micro.support
            )
    return ReleaseRunAudit(
        calls=tuple(calls[k] for k in sorted(calls)),
        unit_statuses={r.assignment.key: r.status for r in results},
        revisions={r.assignment.key: r.revisions for r in results},
        scientific_location_checks=verified,
        passing_model_panels=sum(r.status == "panel_passed" for r in results),
        estimated_usd_known=sum(
            c.estimated_usd for c in calls.values() if c.estimated_usd is not None
        ),
        calls_with_unknown_cost=sum(c.estimated_usd is None for c in calls.values()),
        limitations=(
            "Exact quotation location does not establish scientific entailment.",
            "Specialist panel votes are correlated and uncalibrated; no teacher validation exists.",
            "Costs are recorded list-price estimates, not invoices; unknown values are not zero.",
            "Latency measures individual service calls, not total parallel wall-clock runtime.",
            "This audit includes failed initial transport experiments and interrupted development runs.",
        ),
    )
