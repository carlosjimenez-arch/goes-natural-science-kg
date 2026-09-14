# tests/property/test_chunk_offsets.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Prove lossless semantic chunking and exact normalization-to-source offsets.
from hypothesis import given
from hypothesis import strategies as st

from goes_natural_science_kg.corpus.chunk import chunk_document, resolve_chunk
from goes_natural_science_kg.corpus.normalize import normalize
from goes_natural_science_kg.schemas.ingestion import Paragraph, ParsedDocument, SectionPlan


@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=15))
def test_lossless_partition_and_offsets(parts):
    parts = ["Objective " + p + "\n" for p in parts]
    paragraphs = []
    cursor = 0
    for i, text in enumerate(parts):
        paragraphs.append(
            Paragraph(
                id=f"p-{i}",
                page=1 + i // 3,
                bbox=(0.0, 0.0, 100.0, 100.0),
                start=cursor,
                end=cursor + len(text),
                text=text,
            )
        )
        cursor += len(text)
    parsed = ParsedDocument(
        source_id="source-property-0123456789abcdef",
        document_sha256="a" * 64,
        plan_sha256="b" * 64,
        text="".join(parts),
        paragraphs=paragraphs,
        tables=(),
        sections=[
            SectionPlan(
                key="unit",
                title="Complete unit",
                first_page=1,
                last_page=paragraphs[-1].page,
                grades=[2],
            )
        ],
    )
    chunks = chunk_document(parsed, r"^Objective ")
    assert "".join(c.text for c in chunks) == parsed.text
    for chunk in chunks:
        assert parsed.text[chunk.start : chunk.end] == chunk.text
        assert "".join(a["text"] for a in resolve_chunk(chunk, parsed)) == chunk.text
        original = "".join(chunk.text[a:b] for a, b in chunk.normalized.original_spans)
        assert original == chunk.text
        for char, (a, b) in zip(
            chunk.normalized.text, chunk.normalized.original_spans, strict=True
        ):
            assert char == (" " if chunk.text[a:b].isspace() else chunk.text[a:b])


@given(st.text(max_size=500))
def test_normalization_preserves_every_character(text):
    result = normalize(text)
    assert "".join(text[a:b] for a, b in result.original_spans) == text
    assert normalize(result.text).text == result.text


def test_cross_page_objective_remains_one_chunk():
    paragraphs = (
        Paragraph(id="a", page=10, bbox=(0.0, 0.0, 10.0, 10.0), start=0, end=10, text="Objective "),
        Paragraph(id="b", page=11, bbox=(0.0, 0.0, 10.0, 10.0), start=10, end=19, text="continues"),
    )
    parsed = ParsedDocument(
        source_id="source-property-0123456789abcdef",
        document_sha256="a" * 64,
        plan_sha256="b" * 64,
        text="Objective continues",
        paragraphs=paragraphs,
        tables=(),
        sections=[
            SectionPlan(
                key="unit", title="Cross-page objective", first_page=10, last_page=11, grades=[2]
            )
        ],
    )
    assert len(chunk_document(parsed, r"^Objective")) == 1
