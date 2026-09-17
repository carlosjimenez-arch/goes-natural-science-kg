# src/goes_natural_science_kg/schemas/reviewer_agreement.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Contract inter-reviewer agreement on the equivalence judgments that gate decomposition scores.
"""Model raters are not human ground truth; agreement bounds trust, it does not establish it."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Digest, Probability, Slug, Text
from goes_natural_science_kg.schemas.prompt_evaluation import GeneratorModel, VertexLocation


class ReviewerArm(Contract):
    key: Slug
    model: GeneratorModel
    location: VertexLocation = "us-central1"
    recorded: bool = False


class ReviewerProbePlan(Contract):
    """Re-judge recorded revision-0 candidates with independent reviewer models."""

    schema_version: Literal["reviewer-probe/1.0"] = "reviewer-probe/1.0"
    name: Slug
    project: Text
    prompt_id: Slug = "reference-review"
    prompt_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = "1.0.0"
    generator_arms: Annotated[tuple[Slug, ...], Field(min_length=1)]
    reviewers: Annotated[tuple[ReviewerArm, ...], Field(min_length=2)]
    reference_reviewer: Slug
    replicates: Annotated[int, Field(ge=1, le=3)] = 3
    temperature: float = 0.2
    thinking_budget: int = 128
    seed: int = 0
    location: VertexLocation = "us-central1"

    @model_validator(mode="after")
    def reviewers_are_unique(self) -> Self:
        keys = [r.key for r in self.reviewers]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate reviewer key")
        if self.reference_reviewer not in keys:
            raise ValueError("reference reviewer must be one of the reviewers")
        if sum(r.recorded for r in self.reviewers) != 1:
            raise ValueError("exactly one reviewer replays recorded observations")
        return self


class PairLabel(Contract):
    """One cell of the candidate-by-expected grid, as labelled by one reviewer."""

    generator_arm: Slug
    replicate: Annotated[int, Field(strict=True, ge=0, le=2)]
    case_id: Text
    candidate_key: Text
    expected_key: Text
    equivalent: bool


class ReviewerNodeMatch(Contract):
    """One reviewer's aggregate effect on the score, over the units it actually labelled."""

    reviewer: Slug
    model: Text
    node_match_rate: Probability
    implied_revised_pass_rate: Probability
    cases: Annotated[int, Field(strict=True, ge=1)]
    equivalent_pairs: Annotated[int, Field(strict=True, ge=0)]
    judged_pairs: Annotated[int, Field(strict=True, ge=1)]
    failed_requests: Annotated[int, Field(strict=True, ge=0)] = 0


class PairwiseAgreement(Contract):
    reviewer_a: Slug
    reviewer_b: Slug
    pairs: Annotated[int, Field(strict=True, ge=1)]
    observed_agreement: Probability
    cohen_kappa: float
    both_equivalent: Annotated[int, Field(strict=True, ge=0)]
    only_a_equivalent: Annotated[int, Field(strict=True, ge=0)]
    only_b_equivalent: Annotated[int, Field(strict=True, ge=0)]
    neither_equivalent: Annotated[int, Field(strict=True, ge=0)]


class ReviewerAgreementReport(Contract):
    schema_version: Literal["reviewer-agreement-report/1.0"] = "reviewer-agreement-report/1.0"
    plan_sha256: Digest
    dataset_sha256: Digest
    complete: bool
    human_reviewed: Literal[False] = False
    estimated_total_usd: float
    provider_requests: int
    missing_usage_requests: int
    reviewers: tuple[ReviewerNodeMatch, ...]
    agreements: tuple[PairwiseAgreement, ...]
    krippendorff_alpha: float | None
    node_match_rate_spread: Probability
    implied_pass_rate_spread: Probability
    limitations: tuple[Text, ...]
