# src/goes_natural_science_kg/eval/calibration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Derive micro-skill confidence from panel evidence and calibrate it against human labels.
"""Self-reported confidence is retained for comparison but never published as the estimate."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.calibration import (
    MAX_REVISIONS,
    MINIMUM_CALIBRATION_SAMPLES,
    CalibrationBlock,
    CalibrationRecord,
    CalibrationSample,
    ConfidenceComponents,
    ConfidenceEstimate,
    ConfidenceReport,
    ConfidenceSummary,
)
from goes_natural_science_kg.schemas.orchestration import (
    EvaluableMicro,
    ItemOutcome,
    OrchestrationReport,
)

Floats = NDArray[np.float64]
JUDGING_PREFERENCE = {"L2": 0, "L3": 1, "L1": 2, "L0": 3}


def panel_components(outcome: ItemOutcome, micro: EvaluableMicro) -> ConfidenceComponents:
    """Read the final ballot; micro-skills inherit the item-level panel signals."""
    reported = micro.skill.confidence
    if not outcome.ballots or not outcome.ballots[-1].verdicts:
        return ConfidenceComponents(
            self_reported=reported,
            vote_fraction=None,
            mean_judge_score=None,
            supported_claim_fraction=None,
            revision_fraction=None,
            hard_error_free=None,
        )
    ballot = outcome.ballots[-1]
    claims = [c for v in ballot.verdicts for c in v.claims]
    return ConfidenceComponents(
        self_reported=reported,
        vote_fraction=ballot.votes / len(ballot.verdicts),
        mean_judge_score=float(np.mean([v.score for v in ballot.verdicts])),
        supported_claim_fraction=(
            sum(c.supported for c in claims) / len(claims) if claims else None
        ),
        revision_fraction=outcome.revisions / MAX_REVISIONS,
        hard_error_free=not ballot.hard_errors,
    )


def raw_score(components: ConfidenceComponents) -> float | None:
    """Declared policy: equal-weight mean of available panel signals; hard errors dominate."""
    if components.hard_error_free is None:
        return None
    if not components.hard_error_free:
        return 0.0
    signals = [
        components.vote_fraction,
        components.mean_judge_score,
        components.supported_claim_fraction,
        None if components.revision_fraction is None else 1.0 - components.revision_fraction,
    ]
    present = [s for s in signals if s is not None]
    return float(np.clip(np.mean(present), 0.0, 1.0)) if present else None


def fit_isotonic(samples: Iterable[CalibrationSample], fitted_at: datetime) -> CalibrationRecord:
    """Pool-adjacent-violators isotonic regression of human acceptance on the raw score."""
    ordered = sorted(samples, key=lambda s: (s.raw_score, s.entity_id))
    if len(ordered) < MINIMUM_CALIBRATION_SAMPLES:
        raise ValueError("insufficient calibration sample")
    labels = np.array([float(s.human_accepted) for s in ordered], dtype=np.float64)
    if labels.min() == labels.max():
        raise ValueError("calibration requires both accepted and rejected samples")
    scores = np.array([s.raw_score for s in ordered], dtype=np.float64)
    blocks = _pool_adjacent_violators(scores, labels)
    calibrated = apply_blocks(scores, blocks)
    dataset = content_hash([s.model_dump(mode="json") for s in ordered])
    return CalibrationRecord(
        id=content_hash({"method": "isotonic_pav", "dataset_sha256": dataset}),
        method="isotonic_pav",
        dataset_sha256=dataset,
        sample_size=len(ordered),
        positives=int(labels.sum()),
        blocks=blocks,
        brier_score=float(np.mean((calibrated - labels) ** 2)),
        expected_calibration_error=expected_calibration_error(calibrated, labels),
        fitted_at=fitted_at,
    )


def _pool_adjacent_violators(scores: Floats, labels: Floats) -> tuple[CalibrationBlock, ...]:
    """Merge equal scores first, then pool neighbouring blocks while rates decrease."""
    uniques, inverse = np.unique(scores, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    positives = np.bincount(inverse, weights=labels).astype(np.int64)
    lower = list(uniques)
    upper = list(uniques)
    count = list(counts)
    pos = list(positives)
    index = 0
    while index < len(count) - 1:
        if pos[index] * count[index + 1] > pos[index + 1] * count[index]:
            upper[index] = upper[index + 1]
            count[index] += count[index + 1]
            pos[index] += pos[index + 1]
            del lower[index + 1], upper[index + 1], count[index + 1], pos[index + 1]
            index = max(index - 1, 0)
        else:
            index += 1
    return tuple(
        CalibrationBlock(
            lower=float(lo), upper=float(hi), count=int(n), positives=int(p), rate=float(p / n)
        )
        for lo, hi, n, p in zip(lower, upper, count, pos, strict=True)
    )


def apply_blocks(scores: Floats, blocks: tuple[CalibrationBlock, ...]) -> Floats:
    """Step function: each score takes the rate of the last block starting at or below it."""
    lowers = np.array([b.lower for b in blocks], dtype=np.float64)
    rates = np.array([b.rate for b in blocks], dtype=np.float64)
    positions = np.clip(np.searchsorted(lowers, scores, side="right") - 1, 0, len(blocks) - 1)
    return rates[positions]


def apply_calibration(score: float, record: CalibrationRecord) -> float:
    return float(apply_blocks(np.array([score], dtype=np.float64), record.blocks)[0])


def expected_calibration_error(probabilities: Floats, labels: Floats, bins: int = 10) -> float:
    """Equal-width binning of predicted probabilities; weighted absolute gap to empirical rate."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += (
                mask.sum() / len(labels) * abs(probabilities[mask].mean() - labels[mask].mean())
            )
    return float(np.clip(total, 0.0, 1.0))


def estimate_confidence(
    outcome: ItemOutcome, micro: EvaluableMicro, calibration: CalibrationRecord | None
) -> ConfidenceEstimate:
    components = panel_components(outcome, micro)
    raw = raw_score(components)
    if raw is None:
        return ConfidenceEstimate(
            entity_id=micro.skill.id,
            item_id=outcome.item_id,
            source="self_reported",
            raw_score=components.self_reported,
            value=components.self_reported,
            components=components,
        )
    if calibration is None:
        return ConfidenceEstimate(
            entity_id=micro.skill.id,
            item_id=outcome.item_id,
            source="panel_derived",
            raw_score=raw,
            value=raw,
            components=components,
        )
    return ConfidenceEstimate(
        entity_id=micro.skill.id,
        item_id=outcome.item_id,
        source="calibrated",
        raw_score=raw,
        value=apply_calibration(raw, calibration),
        components=components,
        calibration_id=calibration.id,
        calibrated=True,
    )


def confidence_report(
    report: OrchestrationReport, calibration: CalibrationRecord | None
) -> ConfidenceReport:
    """Recompute every micro-skill estimate from the recorded ballots; nothing is cached.

    A micro-skill appears in the L2 item that judged it and again in the parent (L1) and
    curriculum (L3) items that carry it forward. The estimate comes from the judging level.
    """
    chosen: dict[str, tuple[ItemOutcome, EvaluableMicro]] = {}
    for outcome in sorted(report.items, key=lambda o: (JUDGING_PREFERENCE[o.level], o.item_id)):
        for micro in outcome.micros:
            chosen.setdefault(micro.skill.id, (outcome, micro))
    estimates = [
        estimate_confidence(outcome, micro, calibration)
        for _, (outcome, micro) in sorted(chosen.items())
    ]
    reported = np.array([e.components.self_reported for e in estimates], dtype=np.float64)
    values = np.array([e.value for e in estimates], dtype=np.float64)
    summary = ConfidenceSummary(
        estimates=len(estimates),
        calibrated=sum(e.calibrated for e in estimates),
        self_reported_mean=float(reported.mean()) if estimates else None,
        value_mean=float(values.mean()) if estimates else None,
        mean_absolute_shift=float(np.abs(values - reported).mean()) if estimates else None,
    )
    return ConfidenceReport(
        report_sha256=content_hash(report),
        calibration_id=None if calibration is None else calibration.id,
        estimates=tuple(estimates),
        summary=summary,
    )
