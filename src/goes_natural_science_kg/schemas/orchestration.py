# src/goes_natural_science_kg/schemas/orchestration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import (
    Contract,
    Grade,
    PositiveMinutes,
    Probability,
    Slug,
    StableId,
    Text,
    content_hash,
)
from goes_natural_science_kg.schemas.curriculum import ContextTag, GradeCurriculum
from goes_natural_science_kg.schemas.ingestion import ChunkId, EvidenceReadyMicroSkill
from goes_natural_science_kg.schemas.skills import (
    CognitiveDomain,
    EdgeType,
    GradeBand,
    Skill,
)


class Domain(StrEnum):
    PHYSICAL = "physical-science"
    LIFE = "life-science"
    EARTH = "earth-space-science"


class JudgeRole(StrEnum):
    CURRICULAR = "curricular_judge"
    COGNITIVE = "cognitive_judge"
    GRAPH = "graph_judge"
    EVIDENCE = "evidence_judge"
    AGE = "age_appropriateness_judge"


class LoopPolicy(Contract):
    max_revisions: Literal[3] = 3
    score_threshold: Annotated[Probability, Field(ge=0.8, le=0.8)] = 0.8
    minimum_votes: Literal[4] = 4


class OrchestrationSettings(Contract):
    schema_version: Literal["orchestration-settings/1.0"] = "orchestration-settings/1.0"
    project: Text
    location: Text = "us-central1"
    optimizer_model: Text = "gemini-2.5-flash"
    judge_model: Text = "gemini-2.5-pro"
    seed: int
    policy: LoopPolicy = Field(default_factory=LoopPolicy)
    max_concurrency: Annotated[int, Field(ge=1, le=16)] = 4
    cache_version: Text = "orchestration/1.0"

    @model_validator(mode="after")
    def distinct_models(self) -> Self:
        if self.optimizer_model == self.judge_model:
            raise ValueError("optimizer and judge models must differ")
        return self


class SkillInput(Contract):
    skill: Skill
    domain: Domain
    suggested_grade: Grade
    source_refs: Annotated[tuple[ChunkId, ...], Field(min_length=1)]


class OrchestrationInput(Contract):
    schema_version: Literal["orchestration-input/1.0"] = "orchestration-input/1.0"
    skills: Annotated[tuple[SkillInput, ...], Field(min_length=1)]
    budget_minutes: dict[Grade, Annotated[int, Field(strict=True, gt=0, le=9600)]]

    @model_validator(mode="after")
    def unique_skills(self) -> Self:
        ids = [s.skill.id for s in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate input skills")
        if any(s.suggested_grade not in self.budget_minutes for s in self.skills):
            raise ValueError("missing grade budget")
        return self


class Assignment(Contract):
    skill_id: StableId
    grade: Grade
    domain: Domain
    minutes: PositiveMinutes
    instructions: Text


class RootPlan(Contract):
    schema_version: Literal["root-plan/1.0"] = "root-plan/1.0"
    assignments: tuple[Assignment, ...]
    rationale: Text
    source_refs: tuple[ChunkId, ...]


class WorkerDirective(Contract):
    skill_id: StableId
    instructions: Text


class DomainPlan(Contract):
    schema_version: Literal["domain-plan/1.0"] = "domain-plan/1.0"
    workers: tuple[WorkerDirective, ...]
    rationale: Text
    source_refs: tuple[ChunkId, ...]


class ProposalPrerequisite(Contract):
    id: StableId | Slug
    type: Literal[EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE]


class MicroProposal(Contract):
    key: Slug
    observable_verb: Text
    knowledge_object: Text
    cognitive_domain: CognitiveDomain
    grade_band: GradeBand
    estimated_minutes: PositiveMinutes
    prerequisites: tuple[ProposalPrerequisite, ...] = ()
    misconceptions: tuple[Text, ...] = ()
    evidence_of_mastery: Text
    source_refs: Annotated[tuple[ChunkId, ...], Field(min_length=1)]
    confidence: Probability
    task: Text
    atomicity_reason: Text


class Decomposition(Contract):
    schema_version: Literal["decomposition/1.0"] = "decomposition/1.0"
    micros: Annotated[tuple[MicroProposal, ...], Field(min_length=1)]
    granularity_rationale: Text


class EvaluableMicro(Contract):
    skill: EvidenceReadyMicroSkill
    task: Text
    atomicity_reason: Text


class UnitProposal(Contract):
    key: Slug
    label: Text
    micro_skill_ids: tuple[StableId, ...]
    contexts: Annotated[tuple[ContextTag, ...], Field(min_length=1)]
    contextualized_task: Text
    instruction_minutes: PositiveMinutes
    assessment_minutes: PositiveMinutes
    review_minutes: Annotated[int, Field(strict=True, ge=0)]
    setup_minutes: Annotated[int, Field(strict=True, ge=0)]


class CurriculumProposal(Contract):
    schema_version: Literal["curriculum-proposal/1.0"] = "curriculum-proposal/1.0"
    units: tuple[UnitProposal, ...]
    rationale: Text
    source_refs: tuple[ChunkId, ...]


Candidate = RootPlan | DomainPlan | Decomposition | CurriculumProposal


class EvidenceAnchor(Contract):
    page: int
    paragraph_id: Text
    text: str


class EvidencePacket(Contract):
    chunk_id: ChunkId
    source_url: Text
    source_type: Text
    anchors: tuple[EvidenceAnchor, ...]


class Claim(Contract):
    id: Text
    text: Text
    source_refs: tuple[ChunkId, ...]
    kind: Literal["alignment", "scientific", "design"]


class ClaimVerdict(Contract):
    claim_id: Text
    supported: bool
    source_ref: ChunkId | None
    paragraph_id: Text | None
    quote: str
    rationale: Text


class Verdict(Contract):
    schema_version: Literal["judge-verdict/1.0"] = "judge-verdict/1.0"
    role: JudgeRole
    score: Probability
    passed: bool
    rationale: Text
    claims: tuple[ClaimVerdict, ...] = ()


class Ballot(Contract):
    revision: Annotated[int, Field(ge=0, le=3)]
    verdicts: tuple[Verdict, ...]
    hard_errors: tuple[Text, ...]
    accepted: bool
    votes: int


class WorkItem(Contract):
    id: Text
    level: Literal["L0", "L1", "L2", "L3"]
    assignments: tuple[Assignment, ...] = ()
    instructions: Text
    micros: tuple[EvaluableMicro, ...] = ()
    grade: Grade | None = None
    budget_minutes: Annotated[int, Field(ge=0, le=9600)] = 9600


class ItemOutcome(Contract):
    schema_version: Literal["item-outcome/1.0"] = "item-outcome/1.0"
    item_id: Text
    logical_key: Text
    level: Literal["L0", "L1", "L2", "L3"]
    status: Literal["approved", "needs_human_review"]
    revisions: Annotated[int, Field(ge=0, le=3)]
    ballots: tuple[Ballot, ...]
    candidate: Candidate | None
    micros: tuple[EvaluableMicro, ...] = ()
    curriculum: GradeCurriculum | None = None
    expansion_ratio: float | None = None
    children: tuple[Text, ...] = ()


class OrchestrationReport(Contract):
    schema_version: Literal["orchestration-report/1.0"] = "orchestration-report/1.0"
    input_sha256: Text
    execution_sha256: Text
    prompt_versions: dict[str, Text]
    settings_sha256: Text
    items: tuple[ItemOutcome, ...]
    approved_curricula: tuple[GradeCurriculum, ...]
    complete: bool


class CachedGeneration(Contract):
    schema_version: Literal["cached-generation/1.0"] = "cached-generation/1.0"
    request: dict[str, Any]
    request_sha256: Text
    response: str
    response_sha256: Text
    provider_model_version: Text

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if (
            content_hash(self.request) != self.request_sha256
            or content_hash(self.response) != self.response_sha256
        ):
            raise ValueError("generation cache digest mismatch")
        return self


class AgentRuntime(Contract):
    settings: OrchestrationSettings
    request: OrchestrationInput
    evidence: tuple[EvidencePacket, ...]
    cache: Path
    prompts: Path
    client: Any = Field(default=None, exclude=True)
    limiter: Any = Field(default=None, exclude=True)
    request_locks: Any = Field(default_factory=dict, exclude=True)
