# tests/benchmarks/test_ingestion.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure real PDF parsing and vector indexing on reproducible offline evidence.
import asyncio
from pathlib import Path

import numpy as np
import pytest

from goes_natural_science_kg.corpus.chunk import chunk_document, resolve_chunk
from goes_natural_science_kg.corpus.discover import discover
from goes_natural_science_kg.corpus.embed import embed_texts, pooled_vector
from goes_natural_science_kg.corpus.index import build_index
from goes_natural_science_kg.corpus.normalize import normalize
from goes_natural_science_kg.corpus.parse import parse_pdf
from goes_natural_science_kg.schemas.ingestion import IngestionSettings, ParsedDocument

GOLDEN = Path(__file__).resolve().parents[1] / "golden/pdfs"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_parse_real_pdf(benchmark, position):
    record = discover(GOLDEN / "records.jsonl")[position]
    result = benchmark.pedantic(
        parse_pdf,
        args=(GOLDEN / (record.document.identity.key + ".pdf"), record),
        rounds=5,
        iterations=1,
    )
    assert len({p.page for p in result.paragraphs}) == 2


def test_index_real_evidence(benchmark, tmp_path):
    chunks = []
    vectors = []
    settings = IngestionSettings(project_id="recorded-golden-project")
    for record in discover(GOLDEN / "records.jsonl"):
        parsed = ParsedDocument.model_validate_json(
            (GOLDEN / (record.document.identity.key + ".pdf.expected.json")).read_bytes()
        )
        chunks.extend(chunk_document(parsed))
        embeddings = asyncio.run(
            embed_texts(tuple(p.text for p in parsed.paragraphs), settings, GOLDEN / "embeddings")
        )
        vectors.append(pooled_vector(embeddings))
    result = benchmark.pedantic(
        build_index, args=(tmp_path, tuple(chunks), np.asarray(vectors)), rounds=10, iterations=1
    )
    assert len(result) == 64


def test_semantic_chunk_and_trace(benchmark):
    record = discover(GOLDEN / "records.jsonl")[0]
    parsed = ParsedDocument.model_validate_json(
        (GOLDEN / (record.document.identity.key + ".pdf.expected.json")).read_bytes()
    )
    result = benchmark(chunk_document, parsed)
    assert resolve_chunk(result[0], parsed)


def test_normalize_real_text(benchmark):
    parsed = ParsedDocument.model_validate_json(
        (GOLDEN / "colombia-dba.pdf.expected.json").read_bytes()
    )
    result = benchmark(normalize, parsed.text)
    assert result.original_length == len(parsed.text)
