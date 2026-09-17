# src/goes_natural_science_kg/schemas/prompt_evaluation.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import (
    Contract,
    Digest,
    Slug,
    Text,
    content_hash,
)
from goes_natural_science_kg.schemas.orchestration import (
    Decomposition,
    EvidencePacket,
    SkillInput,
)

Technique = Literal["cot", "few_shot", "cot_few_shot", "structured"]


class ChangeLogEntry(Contract):
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    change: Text


class VersionedPrompt(Contract):
    schema_version: Literal["versioned-prompt/1.0"] = "versioned-prompt/1.0"
    id: Slug
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    role: Text
    technique: Technique
    variables: tuple[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], ...]
    changelog: Annotated[tuple[ChangeLogEntry, ...], Field(min_length=1)]
    language: Literal["es-SV"] = "es-SV"
    output_contract: Text
    example_ids: tuple[Slug, ...] = ()

    @model_validator(mode="after")
    def recorded_version(self) -> Self:
        if self.changelog[-1].version != self.version:
            raise ValueError("current version missing from changelog")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("duplicate prompt variable")
        if self.technique in {"few_shot", "cot_few_shot"} and not 4 <= len(self.example_ids) <= 6:
            raise ValueError("few-shot prompt requires four to six source-backed examples")
        if self.technique == "cot" and self.example_ids:
            raise ValueError("pure CoT cannot contain examples")
        return self


class PromptArtifact(Contract):
    metadata: VersionedPrompt
    body: Text
    sha256: Digest
    legacy: bool = False
    path: Text

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.sha256 != content_hash(
            {"metadata": self.metadata.model_dump(mode="json"), "body": self.body}
        ):
            raise ValueError("prompt digest mismatch")
        return self


class PromptRegistry(Contract):
    schema_version: Literal["prompt-registry/1.0"] = "prompt-registry/1.0"
    artifacts: tuple[PromptArtifact, ...]

    @model_validator(mode="after")
    def identities(self) -> Self:
        keys = [a.metadata.id + "@" + a.metadata.version for a in self.artifacts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate prompt id/version")
        return self

    def get(self, prompt_id: str, version: str) -> PromptArtifact:
        artifact = next(
            (
                a
                for a in self.artifacts
                if a.metadata.id == prompt_id and a.metadata.version == version
            ),
            None,
        )
        if artifact is None:
            raise KeyError(f"unknown prompt version: {prompt_id}@{version}")
        return artifact

    def render(self, prompt_id: str, version: str, variables: dict[str, str]) -> str:
        import re

        artifact = self.get(prompt_id, version)
        if set(variables) != set(artifact.metadata.variables):
            raise ValueError("prompt variables mismatch")
        if artifact.legacy:
            return artifact.body.format(**variables)
        return re.sub(r"\{\{([a-z][a-z0-9_]*)\}\}", lambda m: variables[m.group(1)], artifact.body)


class AnnotatedCase(Contract):
    schema_version: Literal["annotated-science-case/1.0"] = "annotated-science-case/1.0"
    id: Slug
    split: Literal["evaluation", "demonstration"]
    annotation_status: Literal["agent_authored_pending_human_review", "human_reviewed"]
    annotator: Text
    reviewer: Text | None = None
    skill: SkillInput
    evidence: tuple[EvidencePacket, ...]
    expected: Decomposition
    rationale: Text

    @model_validator(mode="after")
    def human_provenance(self) -> Self:
        if self.annotation_status == "human_reviewed" and not self.reviewer:
            raise ValueError("human reviewer required")
        return self


class CaseOutput[T](Contract):
    case_id: Text
    output: T


class BatchOutput[T](Contract):
    results: tuple[CaseOutput[T], ...]


class EvaluationObservation(Contract):
    schema_version: Literal["evaluation-observation/1.0"] = "evaluation-observation/1.0"
    request_sha256: Digest
    request: dict[str, Any]
    response: str | None
    response_sha256: Digest | None
    status: Literal["ok", "failed"]
    error: Text | None = None
    model_version: Text | None = None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    estimated_usd: float | None

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if content_hash(self.request) != self.request_sha256:
            raise ValueError("request digest mismatch")
        if self.response is not None and content_hash(self.response) != self.response_sha256:
            raise ValueError("response digest mismatch")
        return self


class EvaluationMetric(Contract):
    case_id: Text
    role: Text
    technique: Technique
    replicate: int
    schema_valid: bool
    coverage: float | None
    prerequisite_precision: float | None
    prerequisite_recall: float | None
    invalid_reference_rate: float | None
    unsupported_claim_rate: float | None
    judge_correct: bool | None
    passed: bool
    revisions: int
    censored: bool
    notes: tuple[Text, ...] = ()


class ExperimentSettings(Contract):
    schema_version: Literal["prompt-experiment/1.0"] = "prompt-experiment/1.0"
    project: Text
    location: Text = "us-central1"
    generator_model: Literal["gemini-2.5-flash"] = "gemini-2.5-flash"
    judge_model: Literal["gemini-2.5-pro"] = "gemini-2.5-pro"
    replicates: Literal[3] = 3
    batch_size: Annotated[int, Field(ge=1, le=8)] = 4
    concurrency: Annotated[int, Field(ge=1, le=16)] = 8
    temperature: float = 0.2
    thinking_budget: int = 128
    max_revisions: Literal[3] = 3
    coverage_threshold: Annotated[float, Field(ge=0.8, le=0.8)] = 0.8
    variance_limit: Annotated[float, Field(ge=0.1, le=0.1)] = 0.1
    seed: int = 0


class SemanticMatch(Contract):
    candidate_key: Text
    expected_key: Text
    equivalent: bool
    rationale: Text


class SourceSupport(Contract):
    candidate_key: Text
    source_ref: Text
    paragraph_id: Text | None
    quote: str
    supported: bool
    rationale: Text


class ReferenceReview(Contract):
    matches: tuple[SemanticMatch, ...]
    support: tuple[SourceSupport, ...]


class EvaluationCell(Contract):
    schema_version: Literal["evaluation-cell/1.0"] = "evaluation-cell/1.0"
    id: Digest
    case_ids: tuple[Text, ...]
    role: Text
    technique: Technique
    replicate: int
    observations: tuple[Digest, ...]
    metrics: tuple[EvaluationMetric, ...]

    @model_validator(mode="after")
    def complete_cases(self) -> Self:
        if set(self.case_ids) != {m.case_id for m in self.metrics}:
            raise ValueError("missing case metrics")
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("duplicate cell case")
        if any(
            (m.role, m.technique, m.replicate) != (self.role, self.technique, self.replicate)
            for m in self.metrics
        ):
            raise ValueError("metric identity differs from cell")
        for key in self.case_ids:
            items = [m for m in self.metrics if m.case_id == key]
            if [m.revisions for m in items] != list(range(len(items))) or len(items) > 4:
                raise ValueError("non-contiguous or unbounded revision history")
            if not items[-1].passed and not items[-1].censored:
                raise ValueError("failed case must terminate as censored")
        return self


class ReplicateSummary(Contract):
    replicate: int
    cases: int
    metrics: dict[str, float | None]
    metric_sample_counts: dict[str, int]
    estimated_usd: float
    missing_usage_requests: int
    mean_request_seconds: float
    provider_requests: int


class VariantSummary(Contract):
    role: Text
    technique: Technique
    cases: int
    replicates: int
    initial_pass_rate: float
    final_pass_rate: float
    final_pass_sd: float
    schema_rate: float
    coverage: float | None
    prerequisite_precision: float | None
    prerequisite_recall: float | None
    invalid_reference_rate: float | None
    unsupported_claim_rate: float | None
    judge_accuracy: float | None
    mean_revisions: float
    censored_rate: float
    estimated_usd: float
    missing_usage_requests: int
    mean_request_seconds: float
    provider_requests: int
    eligible: bool
    rejection_reasons: tuple[Text, ...]
    replicate_metrics: tuple[ReplicateSummary, ...]
    metric_sd: dict[str, float | None]


class EvaluationReport(Contract):
    schema_version: Literal["prompt-evaluation-report/1.0"] = "prompt-evaluation-report/1.0"
    dataset_sha256: Digest
    human_reviewed: bool
    complete: bool
    estimated_total_usd: float
    missing_usage_requests: int
    provider_requests: int
    expected_cells: int
    completed_cells: int
    variants: tuple[VariantSummary, ...]
    provisional_choices: dict[str, str | None]
    production_choices: dict[str, str | None]
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def promotion_requires_review(self) -> Self:
        if any(self.production_choices.values()) and (not self.complete or not self.human_reviewed):
            raise ValueError("production promotion requires complete measurements and human review")
        return self


# --- Follow-up experiments: parametrized arms over prompt versions and generator models ---

GeneratorModel = Literal[
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
]
VertexLocation = Literal["us-central1", "global"]
SliceDimension = Literal["grade", "domain", "split"]


class PromptVariantRef(Contract):
    prompt_id: Slug
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")] = "1.0.0"


class ExperimentArm(Contract):
    key: Slug
    prompt: PromptVariantRef
    generator_model: GeneratorModel = "gemini-2.5-flash"
    location: VertexLocation = "us-central1"


class FollowUpPlan(Contract):
    """A registered comparison of prompt/model arms on the same forty held-out cases."""

    schema_version: Literal["prompt-followup/1.0"] = "prompt-followup/1.0"
    name: Slug
    role: Literal["decomposition", "curricularization"]
    project: Text
    judge_model: Literal["gemini-2.5-pro"] = "gemini-2.5-pro"
    judge_location: VertexLocation = "us-central1"
    replicates: Literal[3] = 3
    batch_size: Annotated[int, Field(ge=1, le=8)] = 4
    concurrency: Annotated[int, Field(ge=1, le=16)] = 8
    temperature: float = 0.2
    thinking_budget: int = 128
    max_revisions: Literal[3] = 3
    coverage_threshold: Annotated[float, Field(ge=0.8, le=0.8)] = 0.8
    seed: int = 0
    arms: Annotated[tuple[ExperimentArm, ...], Field(min_length=2)]
    baseline_arm: Slug
    tuning_case_ids: tuple[Text, ...] = ()
    bootstrap_resamples: Annotated[int, Field(ge=200, le=20000)] = 2000

    @model_validator(mode="after")
    def arms_are_unique(self) -> Self:
        keys = [a.key for a in self.arms]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate arm key")
        if self.baseline_arm not in keys:
            raise ValueError("baseline arm must be one of the arms")
        if len(set(self.tuning_case_ids)) != len(self.tuning_case_ids):
            raise ValueError("duplicate tuning case id")
        return self


class CellSpec(Contract):
    role: Text
    technique: Technique
    prompt: PromptVariantRef
    generator_model: Text
    generator_location: Text
    judge_model: Text
    judge_location: Text


class FollowUpCell(EvaluationCell):
    schema_version: Literal["evaluation-cell/2.0"] = "evaluation-cell/2.0"  # type: ignore[assignment]
    arm: Slug
    spec: CellSpec
    plan_sha256: Digest


class SliceRow(Contract):
    dimension: SliceDimension
    key: Text
    cases: Annotated[int, Field(strict=True, ge=1)]
    case_replicates: Annotated[int, Field(strict=True, ge=1)]
    final_pass_rate: float
    coverage: float | None
    prerequisite_recall: float | None
    unsupported_claim_rate: float | None
    insufficient: bool


class ArmComparison(Contract):
    arm: Slug
    baseline: Slug
    final_pass_delta: float
    delta_ci_low: float
    delta_ci_high: float
    bootstrap_resamples: int
    improves: bool


class ArmSummary(Contract):
    arm: ExperimentArm
    summary: VariantSummary
    slices: tuple[SliceRow, ...]
    worst_slice: SliceRow | None
    comparison: ArmComparison | None
    usd_per_passed_case: float | None


class FollowUpReport(Contract):
    schema_version: Literal["prompt-followup-report/1.0"] = "prompt-followup-report/1.0"
    plan_sha256: Digest
    dataset_sha256: Digest
    complete: bool
    human_reviewed: bool
    estimated_total_usd: float
    missing_usage_requests: int
    provider_requests: int
    arms: tuple[ArmSummary, ...]
    ranking: tuple[Slug, ...]
    provisional_choice: Slug | None
    production_choice: Slug | None
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def promotion_requires_review(self) -> Self:
        if self.production_choice is not None and (not self.complete or not self.human_reviewed):
            raise ValueError("production promotion requires complete measurements and human review")
        return self
