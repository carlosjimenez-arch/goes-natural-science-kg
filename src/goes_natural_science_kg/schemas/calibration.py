# src/goes_natural_science_kg/schemas/calibration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Replace self-reported micro-skill confidence with panel-derived, calibratable estimates.
"""A confidence value carries its source; only human-labelled calibration makes it a probability."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Digest, Probability, StableId, Text

ConfidenceSource = Literal["self_reported", "panel_derived", "calibrated"]
CalibrationMethod = Literal["isotonic_pav"]
MAX_REVISIONS = 3
MINIMUM_CALIBRATION_SAMPLES = 30


class ConfidenceComponents(Contract):
    """Observable panel signals for one evaluated item; None means the signal was absent."""

    self_reported: Probability
    vote_fraction: Probability | None
    mean_judge_score: Probability | None
    supported_claim_fraction: Probability | None
    revision_fraction: Probability | None
    hard_error_free: bool | None


class ConfidenceEstimate(Contract):
    schema_version: Literal["confidence-estimate/1.0"] = "confidence-estimate/1.0"
    entity_id: StableId
    item_id: Text
    source: ConfidenceSource
    raw_score: Probability
    value: Probability
    components: ConfidenceComponents
    calibration_id: Digest | None = None
    calibrated: bool = False

    @model_validator(mode="after")
    def source_consistency(self) -> Self:
        if self.calibrated != (self.source == "calibrated"):
            raise ValueError("calibrated flag must match the calibrated source")
        if self.calibrated != (self.calibration_id is not None):
            raise ValueError("calibrated estimates require a calibration id and vice versa")
        if self.source == "self_reported" and self.value != self.components.self_reported:
            raise ValueError("self-reported estimate must equal the reported value")
        return self


class CalibrationSample(Contract):
    """One human-reviewed item: the panel raw score and whether the reviewer accepted it."""

    schema_version: Literal["calibration-sample/1.0"] = "calibration-sample/1.0"
    entity_id: StableId
    raw_score: Probability
    human_accepted: bool
    reviewer: Text
    reviewed_at: AwareDatetime


class CalibrationBlock(Contract):
    lower: Probability
    upper: Probability
    count: Annotated[int, Field(strict=True, ge=1)]
    positives: Annotated[int, Field(strict=True, ge=0)]
    rate: Probability

    @model_validator(mode="after")
    def block_bounds(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("block lower bound exceeds upper bound")
        if self.positives > self.count:
            raise ValueError("positives exceed block count")
        return self


class CalibrationRecord(Contract):
    schema_version: Literal["confidence-calibration/1.0"] = "confidence-calibration/1.0"
    id: Digest
    method: CalibrationMethod
    dataset_sha256: Digest
    sample_size: Annotated[int, Field(strict=True, ge=MINIMUM_CALIBRATION_SAMPLES)]
    positives: Annotated[int, Field(strict=True, ge=1)]
    blocks: Annotated[tuple[CalibrationBlock, ...], Field(min_length=1)]
    brier_score: Probability
    expected_calibration_error: Probability
    fitted_at: AwareDatetime

    @model_validator(mode="after")
    def monotone_partition(self) -> Self:
        if sum(b.count for b in self.blocks) != self.sample_size:
            raise ValueError("block counts do not partition the sample")
        if sum(b.positives for b in self.blocks) != self.positives:
            raise ValueError("block positives do not sum to the recorded positives")
        if self.positives >= self.sample_size:
            raise ValueError("calibration requires both accepted and rejected samples")
        for previous, current in zip(self.blocks, self.blocks[1:], strict=False):
            if current.lower <= previous.upper:
                raise ValueError("calibration blocks must be disjoint and increasing")
            if current.rate < previous.rate:
                raise ValueError("calibration rates must be non-decreasing")
        return self


class ConfidenceSummary(Contract):
    estimates: Annotated[int, Field(strict=True, ge=0)]
    calibrated: Annotated[int, Field(strict=True, ge=0)]
    self_reported_mean: Probability | None
    value_mean: Probability | None
    mean_absolute_shift: Probability | None


class ConfidenceReport(Contract):
    """Pure function of an orchestration report and an optional calibration record."""

    schema_version: Literal["confidence-report/1.0"] = "confidence-report/1.0"
    report_sha256: Digest
    calibration_id: Digest | None
    estimates: tuple[ConfidenceEstimate, ...]
    summary: ConfidenceSummary

    @model_validator(mode="after")
    def ordered_and_consistent(self) -> Self:
        ids = [e.entity_id for e in self.estimates]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("estimates must be unique and sorted by entity id")
        if any(e.calibration_id not in (None, self.calibration_id) for e in self.estimates):
            raise ValueError("estimate calibration differs from report calibration")
        if self.summary.estimates != len(self.estimates):
            raise ValueError("summary count differs from estimates")
        return self
