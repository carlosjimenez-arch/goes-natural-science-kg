# src/goes_natural_science_kg/corpus/parse.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Extract reviewable PDF text blocks and separate table geometry.
"""Line geometry defines paragraphs; page selection is explicit and hash-bound."""

import hashlib
import io
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader, PdfWriter

from goes_natural_science_kg.schemas.ingestion import (
    DiscoveryRecord,
    Paragraph,
    ParsedDocument,
    TableRecord,
    parse_plan_hash,
)


def paragraph_groups(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Preserve extractor order and group lines only across small vertical gaps."""
    groups: list[list[dict[str, Any]]] = []
    for line in lines:
        if (
            not groups
            or line["top"] - groups[-1][-1]["bottom"] > 5
            or line["top"] < groups[-1][-1]["top"] - 3
        ):
            groups.append([])
        groups[-1].append(line)
    return groups


def parse_pdf(path: Path, record: DiscoveryRecord) -> ParsedDocument:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = record.excerpt.sha256 if record.excerpt else record.document.sha256
    if expected and digest != expected:
        raise ValueError("PDF does not match source manifest")
    if record.excerpt:
        if record.document.sha256 is None:
            raise ValueError("excerpt requires original document hash")
        digest = record.document.sha256
    if record.reviewed_sha256 != digest or not record.segmentation_review:
        raise ValueError("reviewed semantic plan must match this PDF revision")
    paragraphs: list[Paragraph] = []
    tables: list[TableRecord] = []
    cursor = 0
    pdf_input = pdf_input_stream(path, payload, record.repair_pdf_structure)
    with pdfplumber.open(pdf_input) as pdf:
        physical_pages = (
            record.excerpt.source_pages if record.excerpt else tuple(range(1, len(pdf.pages) + 1))
        )
        if len(physical_pages) != len(pdf.pages):
            raise ValueError("excerpt page map has incorrect length")
        page_lookup = dict(zip(physical_pages, pdf.pages, strict=True))
        for section in sorted(record.sections, key=lambda s: s.first_page):
            if not all(n in page_lookup for n in range(section.first_page, section.last_page + 1)):
                raise ValueError("section extends beyond PDF")
            for number in range(section.first_page, section.last_page + 1):
                page = page_lookup[number].crop(page_lookup[number].bbox)
                page_paragraphs, page_tables, cursor = extract_page(page, number, cursor)
                paragraphs.extend(page_paragraphs)
                tables.extend(page_tables)
                page.close()
    return ParsedDocument(
        source_id=record.document.id,
        document_sha256=digest,
        plan_sha256=parse_plan_hash(record),
        text="".join(p.text for p in trim_sections(paragraphs, record)),
        paragraphs=trim_sections(paragraphs, record),
        tables=tuple(tables),
        sections=record.sections,
    )


def trim_sections(paragraphs: list[Paragraph], record: DiscoveryRecord) -> tuple[Paragraph, ...]:
    """Explicit heading bounds remove unrelated material sharing a physical page."""
    result = []
    cursor = 0
    for section in sorted(record.sections, key=lambda s: s.first_page):
        selected = [p for p in paragraphs if section.first_page <= p.page <= section.last_page]
        text = "".join(p.text for p in selected)
        start = unique_marker(text, section.start_at) if section.start_at is not None else 0
        end = (
            unique_marker(text, section.end_before) if section.end_before is not None else len(text)
        )
        if start >= end:
            raise ValueError("section markers are reversed or empty")
        offset = 0
        for paragraph in selected:
            left, right = max(0, start - offset), min(len(paragraph.text), end - offset)
            offset += len(paragraph.text)
            if left < right:
                part = paragraph.text[left:right]
                result.append(
                    Paragraph.model_validate(
                        paragraph.model_dump()
                        | {"text": part, "start": cursor, "end": cursor + len(part)}
                    )
                )
                cursor += len(part)
    return tuple(result)


def unique_marker(text: str, marker: str) -> int:
    if not marker or text.count(marker) != 1:
        raise ValueError("section marker must occur exactly once in the selected pages")
    return text.index(marker)


def pdf_input_stream(path: Path, payload: bytes, repair: bool) -> Path | io.BytesIO:
    pdf_input: Path | io.BytesIO = path
    if repair:
        # Explicit reviewed repair of cross-reference structure, never overwrite raw bytes.
        reader = PdfReader(io.BytesIO(payload), strict=False)
        writer = PdfWriter()
        for original_page in reader.pages:
            writer.add_page(original_page)
        repaired = io.BytesIO()
        writer.write(repaired)
        repaired.seek(0)
        pdf_input = repaired
    return pdf_input


def extract_page(
    page: pdfplumber.page.Page, number: int, cursor: int
) -> tuple[list[Paragraph], list[TableRecord], int]:
    paragraphs: list[Paragraph] = []
    tables: list[TableRecord] = []
    # Geometric text layer, not rendered pixels or PDF byte offsets.
    lines = page.extract_text_lines(
        layout=False, use_text_flow=True, strip=False, return_chars=False
    )
    if not lines:
        raise ValueError(f"page {number} has no text layer; OCR review required")
    for i, group in enumerate(paragraph_groups(lines)):
        text = "\n".join(line["text"] for line in group) + "\n"
        bbox = (
            min(g["x0"] for g in group),
            min(g["top"] for g in group),
            max(g["x1"] for g in group),
            max(g["bottom"] for g in group),
        )
        paragraphs.append(
            Paragraph(
                id=f"p{number}-paragraph-{i + 1}",
                page=number,
                bbox=bbox,
                start=cursor,
                end=cursor + len(text),
                text=text,
            )
        )
        cursor += len(text)
    detected = page.find_tables()
    method = "ruled"
    if not detected:
        detected = page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
        method = "text_alignment_candidate"
    for i, table in enumerate(detected):
        tables.append(
            TableRecord(
                id=f"p{number}-table-{i + 1}",
                page=number,
                bbox=table.bbox,
                cells=table.extract(),
                cell_bboxes=table.cells,
                detection_method=method,
            )
        )
    return paragraphs, tables, cursor
