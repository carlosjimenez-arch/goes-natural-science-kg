# src/goes_natural_science_kg/corpus/trace.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Resolve graph evidence references to verified documents, pages and paragraphs.
"""Fail closed on unregistered chunks, stale manifests or changed extraction records."""

from pathlib import Path

from goes_natural_science_kg.corpus.chunk import resolve_chunk
from goes_natural_science_kg.corpus.index import load_index
from goes_natural_science_kg.corpus.manifest import read_manifest
from goes_natural_science_kg.schemas.corpus import GateStatus
from goes_natural_science_kg.schemas.ingestion import ParsedDocument


def trace_references(root: Path, references: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    if not references or len(set(references)) != len(references):
        raise ValueError("evidence requires nonempty unique chunk references")
    chunks = {c.id: c for c in load_index(root / "interim/index")[0]}
    documents = {d.id: d for d in read_manifest(root / "manifests/corpus.jsonl")}
    results: list[dict[str, object]] = []
    for reference in references:
        if reference not in chunks:
            raise ValueError("unregistered evidence chunk")
        chunk = chunks[reference]
        source = documents.get(chunk.source_id)
        if (
            source is None
            or source.license_gate_status != GateStatus.ACCEPTED
            or source.sha256 != chunk.document_sha256
        ):
            raise ValueError("evidence document is missing, rejected or stale")
        parsed = ParsedDocument.model_validate_json(
            (root / "interim/evidence" / (chunk.parsed_sha256 + ".json")).read_bytes()
        )
        results.append(
            {
                "chunk_id": chunk.id,
                "source_url": str(source.url),
                "source_title": source.title,
                "license": source.license.value,
                "anchors": resolve_chunk(chunk, parsed),
            }
        )
    return tuple(results)
