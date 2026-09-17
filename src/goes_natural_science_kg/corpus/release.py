# src/goes_natural_science_kg/corpus/release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Reconstruct reviewed textbook units and traceable evidence from source manifests.
from __future__ import annotations

import re
from pathlib import Path

from goes_natural_science_kg.corpus.chunk import chunk_document
from goes_natural_science_kg.corpus.fetch import atomic_bytes, blob_path
from goes_natural_science_kg.corpus.parse import parse_pdf
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.ingestion import (
    DiscoveryRecord,
    Paragraph,
    ParsedDocument,
    parse_plan_hash,
)
from goes_natural_science_kg.schemas.release import ReleaseAnchor, UnitAssignment

# Unit durations transcribed from the reviewed official tables of contents.
WEEKS = {
    2: (6, 6, 4, 6, 5, 5),
    3: (5, 6, 5, 5, 6, 5),
    4: (5, 6, 6, 5, 5, 5),
    5: (5, 6, 5, 4, 5, 7),
    6: (4, 5, 4, 7, 7, 5),
}


def prepare_assignments(
    manifests: tuple[Path, ...], raw: Path, interim: Path
) -> tuple[UnitAssignment, ...]:
    """Reparse SHA-verified sources; preserve whole paragraphs and original offsets."""
    assignments = []
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            plan = DiscoveryRecord.model_validate_json(line)
            if not plan.document.sha256:
                raise ValueError("release source has not been fetched")
            folder = interim / plan.document.id
            parsed_path = folder / "parsed.json"
            parsed = (
                ParsedDocument.model_validate_json(parsed_path.read_bytes())
                if parsed_path.exists()
                else parse_pdf(blob_path(raw, plan.document.sha256), plan)
            )
            if (
                parsed.document_sha256 != plan.document.sha256
                or parsed.plan_sha256 != parse_plan_hash(plan)
            ):
                raise ValueError("cached extraction does not match source manifest")
            atomic_bytes(parsed_path, (canonical_json(parsed) + "\n").encode())
            chunks = chunk_document(parsed)
            atomic_bytes(
                folder / "chunks.jsonl", "".join(canonical_json(c) + "\n" for c in chunks).encode()
            )
            for chunk in chunks:
                grade = chunk.grades[0]
                unit = int(chunk.section_key.rsplit("-", 1)[1])
                paragraph_ids = set(chunk.paragraph_ids)
                by_page: dict[int, list[Paragraph]] = {}
                for paragraph in parsed.paragraphs:
                    if paragraph.id not in paragraph_ids:
                        continue
                    if 70 <= len(paragraph.text) <= 2500 and not re.search(
                        r"[\x00-\x08\x0b\x0c\x0e-\x1f]|[A-Z]{12,}", paragraph.text
                    ):
                        by_page.setdefault(paragraph.page, []).append(paragraph)
                chosen = [
                    p
                    for _, ps in sorted(by_page.items())
                    for p in sorted(ps, key=lambda p: -len(p.text))[:2]
                ]
                anchors = tuple(
                    ReleaseAnchor(
                        key=f"{chunk.section_key}-p{p.page}-{p.id}",
                        chunk_id=chunk.id,
                        source_id=chunk.source_id,
                        source_url=str(plan.document.url),
                        page=p.page,
                        paragraph_id=p.id,
                        start=p.start,
                        end=p.end,
                        text=p.text,
                    )
                    for p in chosen
                )
                assignments.append(
                    UnitAssignment(
                        key=chunk.section_key,
                        grade=grade,
                        title=next(s.title for s in parsed.sections if s.key == chunk.section_key),
                        weeks=WEEKS[grade][unit - 1],
                        anchors=anchors,
                        continuity="Explain entry knowledge and what the next grade should revisit. Do not infer cross-grade prerequisites solely from topic similarity.",
                    )
                )
    return tuple(sorted(assignments, key=lambda a: a.key))
