# tests/unit/test_evidence_integrity.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Reject stale evidence, corrupted vectors, missing caches and ambiguous section markers.
import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_natural_science_kg.corpus.chunk import chunk_document, resolve_chunk
from goes_natural_science_kg.corpus.embed import embed_texts, verify_receipt
from goes_natural_science_kg.corpus.parse import unique_marker
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.ingestion import (
    EmbeddingRecord,
    IngestionSettings,
    ParsedDocument,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden/pdfs"


def test_corrupted_embedding_and_changed_response_are_rejected():
    path = next((GOLDEN / "embeddings").glob("*.json"))
    if path.name == "manifest.json":
        path = next(p for p in (GOLDEN / "embeddings").glob("*.json") if p.name != "manifest.json")
    payload = json.loads(path.read_text())
    record = EmbeddingRecord.model_validate(payload)
    payload["values"][0] += 0.125
    with pytest.raises(ValidationError, match="digest mismatch"):
        EmbeddingRecord.model_validate(payload)
    verify_receipt(record, {record.request_sha256: content_hash(record)})
    with pytest.raises(ValueError, match="changed"):
        verify_receipt(record, {record.request_sha256: "0" * 64})


def test_offline_embedding_cache_miss_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline embedding cache miss"):
        asyncio.run(
            embed_texts(
                ("uncached scientific evidence",),
                IngestionSettings(project_id="offline-test-project"),
                tmp_path,
            )
        )


def test_stale_chunk_cannot_claim_a_changed_text_layer():
    parsed = ParsedDocument.model_validate_json(
        (GOLDEN / "colombia-dba.pdf.expected.json").read_bytes()
    )
    chunk = chunk_document(parsed)[0]
    revised = ParsedDocument.model_validate(parsed.model_dump() | {"document_sha256": "0" * 64})
    with pytest.raises(ValueError, match="different parsed document"):
        resolve_chunk(chunk, revised)


def test_ambiguous_heading_bounds_are_rejected():
    assert unique_marker("before\nScience\nafter", "\nScience\n") == 6
    for marker in ["", "absent", "repeat"]:
        with pytest.raises(ValueError, match="exactly once"):
            unique_marker("repeat repeat", marker)
