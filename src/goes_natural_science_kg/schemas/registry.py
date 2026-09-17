# src/goes_natural_science_kg/schemas/registry.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Export JSON Schema directly from the frozen Pydantic contracts.
"""Committed exports are snapshots, never independent hand-written schemas."""

from datetime import datetime
from pathlib import Path

from goes_natural_science_kg.schemas.artifacts import ArtifactDigest, ArtifactManifest
from goes_natural_science_kg.schemas.base import Contract, canonical_json, content_hash
from goes_natural_science_kg.schemas.calibration import (
    CalibrationRecord,
    CalibrationSample,
    ConfidenceEstimate,
    ConfidenceReport,
)
from goes_natural_science_kg.schemas.configuration import GCPSettings
from goes_natural_science_kg.schemas.corpus import SourceAllowlist, SourceDocument
from goes_natural_science_kg.schemas.curriculum import ContextTag, CurriculumUnit, GradeCurriculum
from goes_natural_science_kg.schemas.decisions import ArchitectureDecision
from goes_natural_science_kg.schemas.graph import EvidenceGraphSnapshot, GraphDiff, GraphSnapshot
from goes_natural_science_kg.schemas.identity import Identity
from goes_natural_science_kg.schemas.ingestion import (
    CorpusCoverage,
    DiscoveryRecord,
    DocumentOutcome,
    EmbeddingReceipt,
    EmbeddingRecord,
    EvidenceBinding,
    EvidenceChunk,
    EvidenceReadyMicroSkill,
    FetchResult,
    IngestionSettings,
    IngestionTimings,
    NormalizedText,
    ParsedDocument,
    PdfExcerpt,
    SectionPlan,
    TableRecord,
)
from goes_natural_science_kg.schemas.orchestration import (
    Ballot,
    CachedGeneration,
    CurriculumProposal,
    Decomposition,
    DomainPlan,
    EvaluableMicro,
    EvidencePacket,
    ItemOutcome,
    OrchestrationInput,
    OrchestrationReport,
    OrchestrationSettings,
    RootPlan,
    Verdict,
    WorkItem,
)
from goes_natural_science_kg.schemas.prompt_evaluation import (
    AnnotatedCase,
    ArmSummary,
    EvaluationCell,
    EvaluationMetric,
    EvaluationObservation,
    EvaluationReport,
    ExperimentSettings,
    FollowUpCell,
    FollowUpPlan,
    FollowUpReport,
    PromptRegistry,
    ReferenceReview,
    ReplicateSummary,
    VariantSummary,
    VersionedPrompt,
)
from goes_natural_science_kg.schemas.prompts import (
    AgentPromptMetadata,
    EvidenceReview,
    PromptMetadata,
)
from goes_natural_science_kg.schemas.release import (
    BindingAudit,
    ContinuityProposal,
    DraftAttempt,
    EditorialCorrection,
    FinalProposalReview,
    PedagogyRecord,
    ProposalChange,
    ProposalCritique,
    ProposalCritiqueItem,
    ProposalDiff,
    ProposalMeasurement,
    ProviderCallSummary,
    PublishedActivityBank,
    PublishedBinding,
    PublishedReviews,
    PublishedUnitReview,
    ReleaseActivity,
    ReleaseActivityBank,
    ReleaseAnchor,
    ReleaseAssignmentSummary,
    ReleaseIssue,
    ReleaseJudgePacket,
    ReleaseMicro,
    ReleasePanel,
    ReleaseRunAudit,
    ReleaseSettings,
    ReleaseSkill,
    ReleaseUnitResult,
    ReleaseVerdict,
    ScheduleHint,
    SolverMeasurement,
    SupportedClaim,
    UnitAssignment,
    UnitDesign,
    UnitPatch,
    UnitSkillDesign,
)
from goes_natural_science_kg.schemas.rescoring import RescoreReport, RevisedMetric
from goes_natural_science_kg.schemas.reviewer_agreement import (
    ReviewerAgreementReport,
    ReviewerProbePlan,
)
from goes_natural_science_kg.schemas.sequencing import (
    ContinuityThread,
    LearningActivity,
    LinkedGraph,
    LocalContext,
    ScheduledUnit,
    SequencingInput,
    SequencingSettings,
    SolvedGradeCurriculum,
    SolverReport,
)
from goes_natural_science_kg.schemas.skills import Edge, MicroSkill, Skill
from goes_natural_science_kg.schemas.workflow import LicenseGateState

SCHEMA_MODELS: tuple[type[Contract], ...] = (
    ReleaseAssignmentSummary,
    ReleaseJudgePacket,
    FinalProposalReview,
    EditorialCorrection,
    ProviderCallSummary,
    ReleaseRunAudit,
    PublishedReviews,
    PublishedUnitReview,
    UnitPatch,
    BindingAudit,
    DraftAttempt,
    ProposalCritique,
    ProposalCritiqueItem,
    ContinuityProposal,
    UnitSkillDesign,
    ReleaseActivityBank,
    ReleaseAnchor,
    UnitAssignment,
    ProposalMeasurement,
    SolverMeasurement,
    ScheduleHint,
    SupportedClaim,
    ReleaseMicro,
    ReleaseSkill,
    ReleaseActivity,
    UnitDesign,
    ReleaseIssue,
    ReleaseVerdict,
    ReleasePanel,
    ReleaseUnitResult,
    ReleaseSettings,
    PublishedActivityBank,
    PublishedBinding,
    PedagogyRecord,
    ProposalChange,
    ProposalDiff,
    EvidenceGraphSnapshot,
    FollowUpPlan,
    FollowUpCell,
    FollowUpReport,
    ArmSummary,
    RevisedMetric,
    RescoreReport,
    ReviewerProbePlan,
    ReviewerAgreementReport,
    CalibrationRecord,
    CalibrationSample,
    ConfidenceEstimate,
    ConfidenceReport,
    SequencingInput,
    SequencingSettings,
    LearningActivity,
    LocalContext,
    ContinuityThread,
    SolvedGradeCurriculum,
    ScheduledUnit,
    SolverReport,
    LinkedGraph,
    VersionedPrompt,
    PromptRegistry,
    AnnotatedCase,
    EvaluationObservation,
    EvaluationCell,
    EvaluationMetric,
    ExperimentSettings,
    ReferenceReview,
    ReplicateSummary,
    VariantSummary,
    EvaluationReport,
    OrchestrationSettings,
    OrchestrationInput,
    OrchestrationReport,
    RootPlan,
    DomainPlan,
    Decomposition,
    CurriculumProposal,
    EvaluableMicro,
    EvidencePacket,
    Verdict,
    Ballot,
    WorkItem,
    ItemOutcome,
    CachedGeneration,
    AgentPromptMetadata,
    CorpusCoverage,
    DiscoveryRecord,
    DocumentOutcome,
    EmbeddingRecord,
    EmbeddingReceipt,
    EvidenceBinding,
    EvidenceChunk,
    EvidenceReadyMicroSkill,
    FetchResult,
    IngestionSettings,
    IngestionTimings,
    NormalizedText,
    ParsedDocument,
    PdfExcerpt,
    SectionPlan,
    TableRecord,
    ArchitectureDecision,
    ArtifactManifest,
    ContextTag,
    CurriculumUnit,
    Edge,
    EvidenceReview,
    GCPSettings,
    GradeCurriculum,
    GraphDiff,
    GraphSnapshot,
    Identity,
    LicenseGateState,
    MicroSkill,
    PromptMetadata,
    Skill,
    SourceAllowlist,
    SourceDocument,
)


def schema_exports() -> dict[str, str]:
    return {
        model.__name__ + ".json": canonical_json(model.model_json_schema()) + "\n"
        for model in sorted(SCHEMA_MODELS, key=lambda m: m.__name__)
    }


def export_schemas(output: Path, generated_at: datetime, seed: int) -> None:
    """Write deterministic exports and their manifest using only supplied run metadata."""
    import hashlib

    exports = schema_exports()
    artifacts = {}
    for name, payload in exports.items():
        data = payload.encode("utf-8")
        artifacts[name] = ArtifactDigest(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    manifest = ArtifactManifest(
        generated_at=generated_at,
        git_sha=None,
        git_dirty=None,
        seed=seed,
        settings_hash=content_hash({"schema_export": "1.0"}),
        inputs={
            f"schemas/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        artifacts=artifacts,
        notes="Schema-only export; no stochastic algorithm. Git state not supplied by this command.",
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in exports.items():
        (output / name).write_text(payload, encoding="utf-8")
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
