# src/goes_natural_science_kg/schemas/ingestion.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Freeze discovery, lossless layout, evidence and ingestion result contracts.
"""Offsets address the immutable extracted Unicode text, never PDF binary bytes."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from goes_natural_science_kg.schemas.base import (
    Contract,
    Digest,
    Grade,
    StableId,
    Text,
    content_hash,
)
from goes_natural_science_kg.schemas.corpus import SourceDocument, SourceUrl
from goes_natural_science_kg.schemas.skills import MicroSkill

Natural = Annotated[int, Field(strict=True, ge=0)]
PageNumber = Annotated[int, Field(strict=True, ge=1)]
ChunkId = Annotated[str, Field(pattern=r"^chunk-[a-z0-9-]+-[0-9a-f]{16}$")]
BBox = tuple[float, float, float, float]


class SectionPlan(Contract):
    """One complete curricular section; PDF pages are physical, one-based inclusive."""

    key: Text
    title: Text
    first_page: PageNumber
    last_page: PageNumber
    grades: tuple[Grade, ...] = ()
    start_at: str | None = None
    end_before: str | None = None

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.first_page > self.last_page or len(set(self.grades)) != len(self.grades):
            raise ValueError("invalid section range or duplicate grades")
        return self


class PdfExcerpt(Contract):
    """Audited derived fixture; local page n maps to physical source_pages[n-1]."""

    sha256: Digest
    source_pages: tuple[PageNumber, ...]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.source_pages or tuple(sorted(set(self.source_pages))) != self.source_pages:
            raise ValueError("excerpt pages must be unique and ordered")
        return self


class DiscoveryRecord(Contract):
    schema_version: Literal["discovery-record/1.0"] = "discovery-record/1.0"
    document: SourceDocument
    discovered_at: AwareDatetime
    discovery_url: SourceUrl
    role: Literal["national_curriculum", "international_comparator", "official_teaching_resource"]
    framework: Text | None = None
    applicability_evidence: Text
    sections: tuple[SectionPlan, ...] = ()
    segmentation_review: Text | None = None
    reviewed_sha256: Digest | None = None
    boundary_pattern: str | None = None
    repair_pdf_structure: bool = False
    excerpt: PdfExcerpt | None = None

    @model_validator(mode="after")
    def coherent_review(self) -> Self:
        pages: set[int] = set()
        for section in self.sections:
            current = set(range(section.first_page, section.last_page + 1))
            if current & pages:
                raise ValueError("overlapping selected sections")
            pages.update(current)
        if self.segmentation_review and (not self.sections or not self.reviewed_sha256):
            raise ValueError("review requires sections and an immutable document hash")
        if self.role == "international_comparator" and not self.framework:
            raise ValueError("comparators require a named framework")
        return self


def parse_plan_hash(record: DiscoveryRecord) -> str:
    """Operational timestamps, URL aliases and review prose never change evidence identity."""
    return content_hash(
        {
            "parser": "pdfplumber-0.11.10-layout-v1",
            "source_id": record.document.id,
            "document_sha256": record.document.sha256,
            "sections": [s.model_dump(mode="json") for s in record.sections],
            "boundary_pattern": record.boundary_pattern,
            "repair_pdf_structure": record.repair_pdf_structure,
            "excerpt": record.excerpt.model_dump(mode="json") if record.excerpt else None,
        }
    )


class Paragraph(Contract):
    id: Text
    page: PageNumber
    bbox: BBox
    start: Natural
    end: Natural
    text: str

    @model_validator(mode="after")
    def offsets(self) -> Self:
        if self.end - self.start != len(self.text) or not self.text:
            raise ValueError("paragraph offsets do not match text")
        return self


class TableRecord(Contract):
    id: Text
    page: PageNumber
    bbox: BBox
    cells: tuple[tuple[str | None, ...], ...]
    cell_bboxes: tuple[BBox | None, ...]
    detection_method: Literal["ruled", "text_alignment_candidate"]


class ParsedDocument(Contract):
    schema_version: Literal["parsed-document/1.0"] = "parsed-document/1.0"
    source_id: StableId
    document_sha256: Digest
    parser_version: Literal["pdfplumber-0.11.10-layout-v1"] = "pdfplumber-0.11.10-layout-v1"
    plan_sha256: Digest
    text: str
    paragraphs: tuple[Paragraph, ...]
    tables: tuple[TableRecord, ...]
    sections: tuple[SectionPlan, ...]

    @model_validator(mode="after")
    def lossless(self) -> Self:
        cursor = 0
        for paragraph in self.paragraphs:
            if (
                paragraph.start != cursor
                or self.text[paragraph.start : paragraph.end] != paragraph.text
            ):
                raise ValueError("paragraphs must partition the extracted text")
            cursor = paragraph.end
        if cursor != len(self.text) or not self.text.strip():
            raise ValueError("empty or incomplete text layer; OCR/review required")
        if len({p.id for p in self.paragraphs}) != len(self.paragraphs):
            raise ValueError("duplicate paragraph IDs")
        return self


class NormalizedText(Contract):
    """Each normalized character maps to a half-open original span, including whitespace."""

    text: str
    original_spans: tuple[tuple[Natural, Natural], ...]
    original_length: Natural

    @model_validator(mode="after")
    def contiguous(self) -> Self:
        if len(self.text) != len(self.original_spans):
            raise ValueError("one span is required per normalized character")
        cursor = 0
        for start, end in self.original_spans:
            if start != cursor or end <= start:
                raise ValueError("normalization map must partition original text")
            cursor = end
        if cursor != self.original_length:
            raise ValueError("normalization lost source characters")
        return self


class EvidenceChunk(Contract):
    schema_version: Literal["evidence-chunk/1.0"] = "evidence-chunk/1.0"
    id: ChunkId
    source_id: StableId
    document_sha256: Digest
    parsed_sha256: Digest
    section_key: Text
    grades: tuple[Grade, ...]
    start: Natural
    end: Natural
    text: str
    paragraph_ids: tuple[Text, ...]
    normalized: NormalizedText

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.end - self.start != len(self.text) or not self.paragraph_ids:
            raise ValueError("chunk requires exact text and paragraph anchors")
        if self.normalized.original_length != len(self.text):
            raise ValueError("normalization belongs to another chunk")
        if self.id != chunk_id(self.source_id, self.parsed_sha256, self.start, self.end):
            raise ValueError("chunk ID does not match canonical identity")
        return self


def chunk_id(source_id: str, parsed_sha256: str, start: int, end: int) -> str:
    digest = content_hash({"version": 1, "parsed": parsed_sha256, "start": start, "end": end})
    return "chunk-" + source_id.removeprefix("source-").rsplit("-", 1)[0] + "-" + digest[:16]


class EvidenceBinding(Contract):
    """Required publication sidecar for historical MicroSkill/Edge document references."""

    schema_version: Literal["evidence-binding/1.0"] = "evidence-binding/1.0"
    entity_id: StableId
    source_refs: Annotated[tuple[ChunkId, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_refs(self) -> Self:
        if len(set(self.source_refs)) != len(self.source_refs):
            raise ValueError("duplicate evidence references")
        if not self.entity_id.startswith(("micro-", "edge-")):
            raise ValueError("evidence binding requires a micro-skill or edge")
        return self


class EvidenceReadyMicroSkill(MicroSkill):
    """Version 2 publication contract: source_refs address chunks, never only PDFs."""

    schema_version: Literal["micro-skill/2.0"] = "micro-skill/2.0"  # type: ignore[assignment]
    source_refs: Annotated[tuple[ChunkId, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def reference_shape(self) -> Self:
        if not self.parent_skill_id.startswith("skill-"):
            raise ValueError("parent_skill_id must reference a Skill")
        keys = [(p.type.value, p.id) for p in self.prerequisites]
        if len(keys) != len(set(keys)) or any(p.id == self.id for p in self.prerequisites):
            raise ValueError("duplicate or self prerequisite")
        if len(set(self.source_refs)) != len(self.source_refs):
            raise ValueError("duplicate evidence references")
        object.__setattr__(self, "source_refs", tuple(sorted(self.source_refs)))
        object.__setattr__(
            self,
            "prerequisites",
            tuple(sorted(self.prerequisites, key=lambda p: (p.type.value, p.id))),
        )
        return self


class DocumentOutcome(Contract):
    schema_version: Literal["document-outcome/1.0"] = "document-outcome/1.0"
    source_id: StableId
    status: Literal["indexed", "rejected", "failed", "needs_review"]
    error: Text | None = None
    parsed_sha256: Digest | None = None
    chunk_ids: tuple[ChunkId, ...] = ()
    pages: Natural = 0
    paragraphs: Natural = 0
    tables: Natural = 0


class GradeCoverage(Contract):
    grade: Grade
    source_ids: tuple[StableId, ...]
    distinct_document_hashes: tuple[Digest, ...]
    countries: tuple[Text, ...]
    meets_target: bool


class CorpusCoverage(Contract):
    schema_version: Literal["corpus-coverage/1.0"] = "corpus-coverage/1.0"
    grades: tuple[GradeCoverage, ...]
    comparators: dict[str, tuple[StableId, ...]]
    missing_comparators: tuple[Text, ...]
    complete: bool


class FetchResult(Contract):
    schema_version: Literal["fetch-result/1.0"] = "fetch-result/1.0"
    document: SourceDocument
    status: Literal["fetched", "cached", "rejected", "failed"]
    error: Text | None = None


class EmbeddingRecord(Contract):
    schema_version: Literal["embedding-record/1.0"] = "embedding-record/1.0"
    request_sha256: Digest
    model: Text
    task: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    text_sha256: Digest
    values: Annotated[tuple[Annotated[float, Field(allow_inf_nan=False)], ...], Field(min_length=1)]
    token_count: Natural
    truncated: Literal[False] = False
    vector_sha256: Digest

    @model_validator(mode="after")
    def vector_integrity(self) -> Self:
        if self.vector_sha256 != content_hash(list(self.values)):
            raise ValueError("embedding vector digest mismatch")
        return self


class IngestionSettings(Contract):
    concurrency: Annotated[int, Field(ge=1, le=8)] = 4
    host_interval_seconds: Annotated[float, Field(ge=0.1)] = 1.0
    attempts: Annotated[int, Field(ge=1, le=5)] = 3
    timeout_seconds: Annotated[float, Field(gt=0)] = 60.0
    max_download_bytes: Annotated[int, Field(gt=0)] = 100_000_000
    embedding_model: Literal["text-multilingual-embedding-002"] = "text-multilingual-embedding-002"
    embedding_dimensions: Literal[768] = 768
    project_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")]
    location: Literal["us-central1"] = "us-central1"


class OperationTiming(Contract):
    seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    parse_or_cache_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    pages: Natural | None = None
    chunks: Natural
    dimensions: Natural | None = None


class IngestionTimings(Contract):
    schema_version: Literal["ingestion-timings/1.0"] = "ingestion-timings/1.0"
    operations: dict[str, OperationTiming]
    environment: dict[str, str] = Field(default_factory=dict)


class EmbeddingReceipt(Contract):
    schema_version: Literal["embedding-receipt/1.0"] = "embedding-receipt/1.0"
    request_sha256: Digest
    response_sha256: Digest
    model: Text
    dimensions: Natural
