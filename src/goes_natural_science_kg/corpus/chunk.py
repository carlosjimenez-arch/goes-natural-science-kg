# src/goes_natural_science_kg/corpus/chunk.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Partition reviewed curricular sections without token windows or page splits.
"""A conservative section remains whole unless reviewed boundaries identify units."""

import re
from itertools import pairwise

from goes_natural_science_kg.corpus.normalize import normalize
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.ingestion import EvidenceChunk, ParsedDocument, chunk_id


def chunk_document(
    parsed: ParsedDocument, boundary_pattern: str | None = None
) -> tuple[EvidenceChunk, ...]:
    digest = content_hash(parsed)
    chunks: list[EvidenceChunk] = []
    for section in sorted(parsed.sections, key=lambda s: s.first_page):
        paragraphs = [
            p for p in parsed.paragraphs if section.first_page <= p.page <= section.last_page
        ]
        start, end = paragraphs[0].start, paragraphs[-1].end
        # Only whole paragraph boundaries may split a curricular section. A matched
        # code within a paragraph keeps that complete paragraph together.
        boundaries = [start]
        if boundary_pattern:
            pattern = re.compile(boundary_pattern)
            boundaries.extend(p.start for p in paragraphs[1:] if pattern.match(p.text))
        boundaries.append(end)
        for left, right in pairwise(boundaries):
            text = parsed.text[left:right]
            anchors = tuple(p.id for p in paragraphs if left <= p.start and p.end <= right)
            chunks.append(
                EvidenceChunk(
                    id=chunk_id(parsed.source_id, digest, left, right),
                    source_id=parsed.source_id,
                    document_sha256=parsed.document_sha256,
                    parsed_sha256=digest,
                    section_key=section.key,
                    grades=section.grades,
                    start=left,
                    end=right,
                    text=text,
                    paragraph_ids=anchors,
                    normalized=normalize(text),
                )
            )
    if "".join(c.text for c in chunks) != parsed.text:
        raise ValueError("semantic chunks failed to partition selected document text")
    return tuple(chunks)


def resolve_chunk(chunk: EvidenceChunk, parsed: ParsedDocument) -> tuple[dict[str, object], ...]:
    """Return exact source page/paragraph anchors, rejecting stale text or source hashes."""
    if (
        chunk.parsed_sha256 != content_hash(parsed)
        or chunk.document_sha256 != parsed.document_sha256
    ):
        raise ValueError("evidence references a different parsed document revision")
    if chunk.source_id != parsed.source_id or parsed.text[chunk.start : chunk.end] != chunk.text:
        raise ValueError("evidence text/source mismatch")
    anchors = tuple(p for p in parsed.paragraphs if chunk.start <= p.start and p.end <= chunk.end)
    if (
        tuple(p.id for p in anchors) != chunk.paragraph_ids
        or "".join(p.text for p in anchors) != chunk.text
    ):
        raise ValueError("evidence paragraph map mismatch")
    return tuple(
        {
            "source_id": parsed.source_id,
            "document_sha256": parsed.document_sha256,
            "page": p.page,
            "paragraph": p.id,
            "bbox": p.bbox,
            "start": p.start,
            "end": p.end,
            "text": p.text,
        }
        for p in anchors
    )
