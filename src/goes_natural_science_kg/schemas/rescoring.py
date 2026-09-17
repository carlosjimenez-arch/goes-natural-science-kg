# src/goes_natural_science_kg/schemas/rescoring.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Contract a revised decomposition score that separates reference agreement from design quality.
"""Scorer 1.1.0 stays frozen; these contracts describe an offline re-reading of the same responses."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Digest, Probability, Slug, Text

RESCORER_VERSION = "2.0.0"


class ReferenceAgreement(Contract):
    """How far the candidate reproduces the draft reference decomposition."""

    node_match_rate: Probability
    matched_nodes: Annotated[int, Field(strict=True, ge=0)]
    expected_nodes: Annotated[int, Field(strict=True, ge=1)]
    candidate_nodes: Annotated[int, Field(strict=True, ge=0)]
    edge_precision_matched: Probability | None
    edge_recall_matched: Probability | None
    comparable_expected_edges: Annotated[int, Field(strict=True, ge=0)]
    comparable_candidate_edges: Annotated[int, Field(strict=True, ge=0)]


class DesignQuality(Contract):
    """Reference-free checks: they hold even if the draft reference is wrong."""

    single_observable_verb: Probability
    distinct_cognitive_domains: Annotated[int, Field(strict=True, ge=0)]
    has_reasoning_step: bool
    monotone_cognitive_chain: bool
    grade_band_contains_grade: bool
    minutes_within_assignment: bool
    citations_resolve: bool
    structural_errors: tuple[Text, ...] = ()


class RevisedMetric(Contract):
    schema_version: Literal["revised-decomposition-metric/1.0"] = "revised-decomposition-metric/1.0"
    arm: Slug
    case_id: Text
    grade: Annotated[int, Field(strict=True, ge=2, le=6)]
    domain: Text
    revision: Annotated[int, Field(strict=True, ge=0, le=3)]
    schema_valid: bool
    agreement: ReferenceAgreement | None
    design: DesignQuality | None
    frozen_passed: bool
    revised_passed: bool

    @model_validator(mode="after")
    def invalid_outputs_carry_no_scores(self) -> Self:
        if not self.schema_valid and (self.agreement or self.design):
            raise ValueError("an invalid output cannot carry scores")
        if self.revised_passed and not self.schema_valid:
            raise ValueError("an invalid output cannot pass")
        return self


class RescoreSliceRow(Contract):
    dimension: Literal["grade", "domain"]
    key: Text
    cases: Annotated[int, Field(strict=True, ge=1)]
    frozen_pass_rate: Probability
    revised_pass_rate: Probability
    node_match_rate: Probability | None
    edge_recall_matched: Probability | None
    insufficient: bool


class RescoreArmSummary(Contract):
    arm: Slug
    generator_model: Text
    prompt_id: Slug
    case_replicates: Annotated[int, Field(strict=True, ge=1)]
    frozen_pass_rate: Probability
    revised_pass_rate: Probability
    node_match_rate: Probability | None
    edge_precision_matched: Probability | None
    edge_recall_matched: Probability | None
    single_observable_verb: Probability | None
    monotone_cognitive_chain_rate: Probability
    reasoning_step_rate: Probability
    citations_resolve_rate: Probability
    structural_error_rate: Probability
    slices: tuple[RescoreSliceRow, ...]
    worst_grade_slice: RescoreSliceRow | None


class RescoreReport(Contract):
    """A re-reading of recorded responses under a revised rule; no provider call is made."""

    schema_version: Literal["decomposition-rescore-report/1.0"] = "decomposition-rescore-report/1.0"
    rescorer_version: Literal["2.0.0"] = "2.0.0"
    source_report_sha256: Digest
    dataset_sha256: Digest
    rescored_case_replicates: Annotated[int, Field(strict=True, ge=1)]
    arms: tuple[RescoreArmSummary, ...]
    metrics: tuple[RevisedMetric, ...]
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def metrics_are_ordered(self) -> Self:
        if self.rescored_case_replicates != len(self.metrics):
            raise ValueError("summary count differs from metrics")
        return self
