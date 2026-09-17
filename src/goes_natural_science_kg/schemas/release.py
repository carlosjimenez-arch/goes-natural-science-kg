# src/goes_natural_science_kg/schemas/release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Contract evidence-grounded proposal snapshots and explicit review limitations.
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Self, TypedDict

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Grade, Slug, Text
from goes_natural_science_kg.schemas.skills import CognitiveDomain


class ReleaseAnchor(Contract):
    key: Text
    chunk_id: Text
    source_id: Text
    source_url: Text
    page: int
    paragraph_id: Text
    start: int
    end: int
    text: Text


class UnitAssignment(Contract):
    key: Slug
    grade: Grade
    title: Text
    weeks: int
    anchors: tuple[ReleaseAnchor, ...]
    continuity: Text


class SupportedClaim(Contract):
    anchor_key: Text
    quote: Annotated[str, Field(min_length=8, max_length=240)]
    claim: Text


class ReleaseMicro(Contract):
    key: Slug
    verb: Text
    knowledge_object: Text
    cognitive_domain: CognitiveDomain
    estimated_minutes: Annotated[int, Field(ge=10, le=90)]
    prerequisites: tuple[Slug, ...] = ()
    misconceptions: tuple[Text, ...]
    mastery: Text
    assessment_task: Text
    expected_response: Text
    scoring_rule: Text
    support: Annotated[tuple[SupportedClaim, ...], Field(min_length=1)]
    key_for_retrieval: bool = False


class ReleaseSkill(Contract):
    key: Slug
    label: Text
    domain: Literal["physical-science", "life-science", "earth-space-science"]
    pedagogical_rationale: Text
    micros: Annotated[tuple[ReleaseMicro, ...], Field(min_length=4, max_length=14)]


class ReleaseActivity(Contract):
    key: Slug
    micro_keys: Annotated[tuple[Slug, ...], Field(min_length=1)]
    kind: Literal["teach", "practice", "retrieval", "bridge", "assessment"]
    minutes: Annotated[int, Field(ge=10, le=120)]
    cognitive_domain: CognitiveDomain
    inquiry: bool
    theme: Slug
    task: Text
    mastery_criterion: Text
    materials: tuple[Text, ...]
    safety: Text
    differentiation: Text
    rationale_for_minutes: Text
    context_tasks: dict[Slug, Text] = Field(default_factory=dict)


class UnitDesign(Contract):
    schema_version: Literal["release-unit/1.0"] = "release-unit/1.0"
    unit_key: Slug
    skills: Annotated[tuple[ReleaseSkill, ...], Field(min_length=2, max_length=3)]
    activities: tuple[ReleaseActivity, ...]
    progression: Text
    limitations: tuple[Text, ...]


class UnitSkillDesign(Contract):
    unit_key: Slug
    skills: Annotated[tuple[ReleaseSkill, ...], Field(min_length=2, max_length=3)]
    progression: Text
    limitations: tuple[Text, ...]


class ReleaseActivityBank(Contract):
    unit_key: Slug
    activities: tuple[ReleaseActivity, ...]


class UnitPatch(Contract):
    unit_key: Slug
    skills: tuple[ReleaseSkill, ...] = ()
    activities: tuple[ReleaseActivity, ...] = ()
    remove_activity_keys: tuple[Slug, ...] = ()
    progression: Text | None = None
    limitations: tuple[Text, ...] | None = None


class ReleaseIssue(Contract):
    entity_key: Text
    severity: Literal["critical", "major", "minor"]
    finding: Text
    required_change: Text


class ReleaseVerdict(Contract):
    criterion: Literal["curricular", "cognitive", "graph", "evidence", "age"]
    passed: bool
    score: Annotated[float, Field(ge=0, le=1)]
    issues: tuple[ReleaseIssue, ...]
    rationale: Text


class ReleasePanel(Contract):
    revision: int
    model_by_criterion: dict[str, str]
    verdicts: tuple[ReleaseVerdict, ...]
    deterministic_errors: tuple[str, ...]
    accepted: bool
    request_ids: tuple[str, ...]


class ReleaseUnitResult(Contract):
    assignment: UnitAssignment
    initial: UnitDesign | None
    final: UnitDesign | None
    panels: tuple[ReleasePanel, ...]
    revisions: int
    status: Literal["panel_passed", "needs_human_review", "failed"]
    request_ids: tuple[str, ...]
    errors: tuple[str, ...] = ()


class ReleaseSettings(Contract):
    schema_version: Literal["release-settings/1.0"] = "release-settings/1.0"
    project: str
    seed: int = 42
    temperature: float = 0.2
    thinking_budget: int = 1024
    max_revisions: Literal[3] = 3
    concurrency: Annotated[int, Field(ge=1, le=16)] = 5
    generator_model: str = "gemini-3-flash-preview"
    optimizer_model: str = "gemini-3.1-pro-preview"
    judge_models: dict[str, str] = Field(
        default_factory=lambda: {
            "curricular": "gemini-2.5-pro",
            "cognitive": "gemini-3.1-pro-preview",
            "graph": "gemini-3-flash-preview",
            "evidence": "gemini-2.5-pro",
            "age": "gemini-3.1-pro-preview",
        }
    )
    score_threshold: Annotated[float, Field(ge=0, le=1)] = 0.8
    minimum_votes: Annotated[int, Field(ge=4, le=5)] = 4

    @model_validator(mode="after")
    def criteria(self) -> Self:
        if set(self.judge_models) != {"curricular", "cognitive", "graph", "evidence", "age"}:
            raise ValueError("all five specialist criteria are required")
        return self


class ReleaseRuntime(Contract):
    assignments: tuple[UnitAssignment, ...]
    settings: ReleaseSettings
    contexts: tuple[dict[str, Any], ...]
    cache: Any = Field(exclude=True)
    output: Any = Field(exclude=True)
    clients: Any = Field(exclude=True)
    registry: Any = Field(exclude=True)
    limiter: Any = Field(exclude=True)
    request_locks: Any = Field(exclude=True)


class ReleaseState(TypedDict, total=False):
    runtime: ReleaseRuntime
    assignment: UnitAssignment
    results: Annotated[list[ReleaseUnitResult], operator.add]


class PublishedBinding(Contract):
    entity_id: Text
    source_id: Text
    source_url: Text
    chunk_id: Text
    page: int
    paragraph_id: Text
    quote_start: int
    quote_end: int
    quote_sha256: Text
    claim: Text
    location_verified: bool


class PedagogyRecord(Contract):
    entity_id: Text
    unit_key: Text
    grade: Grade
    assessment_task: Text
    expected_response: Text
    scoring_rule: Text
    confidence_basis: Literal["uncalibrated_unit_panel_vote_fraction"]
    human_validated: Literal[False] = False
    limitations: tuple[Text, ...]


class ProposalChange(Contract):
    entity_id: Text
    operation: Literal["added", "removed", "modified"]
    changed_fields: tuple[Text, ...]


class ProposalDiff(Contract):
    schema_version: Literal["proposal-diff/1.0"] = "proposal-diff/1.0"
    before_sha256: Text
    after_sha256: Text
    changes: tuple[ProposalChange, ...]


class ContinuityProposal(Contract):
    id: Slug
    source_micro_id: Text
    target_micro_id: Text
    target_grade: Grade
    task: Text
    expected_response: Text
    scoring_rule: Text
    minutes: Annotated[int, Field(ge=10, le=90)]
    rationale: Text


class PublishedUnitReview(Contract):
    unit_key: Slug
    status: Literal["panel_passed", "needs_human_review", "failed"]
    revisions: int
    panels: tuple[ReleasePanel, ...]
    request_ids: tuple[str, ...]
    assignment_sha256: Text
    errors: tuple[str, ...]


class PublishedReviews(Contract):
    schema_version: Literal["proposal-reviews/1.0"] = "proposal-reviews/1.0"
    units: tuple[PublishedUnitReview, ...]


class ProviderCallSummary(Contract):
    prompt_version: Text
    request_sha256: Text
    response_sha256: Text | None
    prompt_id: Text
    model: Text
    model_version: Text | None
    status: Text
    error: Text | None
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_usd: float | None
    transport_retry: int


class ReleaseRunAudit(Contract):
    schema_version: Literal["release-run-audit/1.0"] = "release-run-audit/1.0"
    calls: tuple[ProviderCallSummary, ...]
    unit_statuses: dict[str, str]
    revisions: dict[str, int]
    scientific_location_checks: int
    passing_model_panels: int
    human_validated: Literal[False] = False
    estimated_usd_known: float
    calls_with_unknown_cost: int
    limitations: tuple[Text, ...]


class EditorialCorrection(Contract):
    unit_key: Slug
    entity_key: Text
    category: Literal[
        "transport_normalization",
        "scientific_alignment",
        "cognitive_alignment",
        "safety",
        "continuity",
        "assessment",
        "coverage",
    ]
    reason: Text
    before_sha256: Text
    after_sha256: Text
    human_validated: Literal[False] = False


class ReleaseAssignmentSummary(Contract):
    key: Slug
    grade: Grade
    title: Text
    weeks: int
    continuity: Text


class ReleaseJudgePacket(Contract):
    assignment: ReleaseAssignmentSummary
    proposal: UnitDesign
    evidence: tuple[ReleaseAnchor, ...]
    deterministic_errors: tuple[str, ...]


class FinalProposalReview(Contract):
    schema_version: Literal["final-proposal-review/1.0"] = "final-proposal-review/1.0"
    unit_key: Slug
    candidate_sha256: Text
    stage: Literal["single-pass-editorial-verification"] = "single-pass-editorial-verification"
    prompt_version: Literal["1.1.0"] = "1.1.0"
    panel: ReleasePanel
    errors: tuple[str, ...]
    human_validated: Literal[False] = False


class DraftAttempt(Contract):
    schema_version: Literal["proposal-draft-attempt/1.0"] = "proposal-draft-attempt/1.0"
    status: Literal["rejected_before_scheduling"] = "rejected_before_scheduling"
    graph_sha256: Text
    structural_errors: dict[str, tuple[Text, ...]]
    limitation: Text


class BindingAudit(Contract):
    schema_version: Literal["proposal-binding-audit/1.0"] = "proposal-binding-audit/1.0"
    checked: int
    source_sha256: dict[Text, Text]
    errors: tuple[Text, ...]
    semantic_entailment_verified: Literal[False] = False


class ProposalCritiqueItem(Contract):
    category: Text
    observation: Text
    action: Text
    remaining_limit: Text
    evidence_paths: tuple[Text, ...]
    source_urls: tuple[Text, ...] = ()


class ProposalCritique(Contract):
    schema_version: Literal["proposal-critique/1.0"] = "proposal-critique/1.0"
    retrieved_at: Text
    human_validated: Literal[False] = False
    items: tuple[ProposalCritiqueItem, ...]
    release_gate: Literal["proposal_only_requires_teacher_review"] = (
        "proposal_only_requires_teacher_review"
    )


class PublishedActivityBank(Contract):
    schema_version: Literal["published-activity-bank/1.0"] = "published-activity-bank/1.0"
    unit_key: Slug
    activities: tuple[ReleaseActivity, ...]


class ScheduleHint(Contract):
    activity_id: Slug
    grade: Grade
    start: int
    end: int


class SolverMeasurement(Contract):
    elapsed_seconds: float
    status: Text


class ProposalMeasurement(Contract):
    schema_version: Literal["proposal-measurement/1.0"] = "proposal-measurement/1.0"
    first_run: SolverMeasurement
    repeat_seconds: float
    byte_identical_report: bool
    unhinted_attempt: Text
    seed: int
