# src/goes_natural_science_kg/corpus/benchmark.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Benchmark real corpus parsing and indexing without model or network calls.
"""Bypass parsed caches; operating-system file caches remain uncontrolled and disclosed."""

import platform
import time
from datetime import datetime
from pathlib import Path

from goes_natural_science_kg.corpus.discover import discover
from goes_natural_science_kg.corpus.fetch import atomic_bytes, blob_path
from goes_natural_science_kg.corpus.index import build_index, load_index
from goes_natural_science_kg.corpus.manifest import read_manifest
from goes_natural_science_kg.corpus.parse import parse_pdf
from goes_natural_science_kg.corpus.pipeline import write_artifact_manifest
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.ingestion import (
    DiscoveryRecord,
    IngestionTimings,
    OperationTiming,
)


def benchmark_corpus(
    selection: Path, root: Path, output: Path, generated_at: datetime, seed: int, rounds: int = 3
) -> IngestionTimings:
    if not 1 <= rounds <= 10:
        raise ValueError("benchmark rounds must be between 1 and 10")
    documents = {d.id: d for d in read_manifest(root / "manifests/corpus.jsonl")}
    chunks, vectors = load_index(root / "interim/index")
    indexed = {c.source_id for c in chunks}
    records = [
        DiscoveryRecord.model_validate(r.model_dump() | {"document": documents[r.document.id]})
        for r in discover(selection)
        if r.document.id in indexed
    ]
    operations = {}
    for iteration in range(rounds):
        for record in records:
            if not record.document.sha256:
                raise ValueError("indexed document has no raw digest")
            started = time.perf_counter()
            parsed = parse_pdf(blob_path(root / "raw", record.document.sha256), record)
            operations[f"parse-{record.document.identity.key}-round-{iteration + 1}"] = (
                OperationTiming(
                    seconds=time.perf_counter() - started,
                    pages=len({p.page for p in parsed.paragraphs}),
                    chunks=sum(c.source_id == record.document.id for c in chunks),
                )
            )
        started = time.perf_counter()
        build_index(root / "interim/benchmark-index", chunks, vectors)
        operations[f"index-round-{iteration + 1}"] = OperationTiming(
            seconds=time.perf_counter() - started, chunks=len(chunks), dimensions=vectors.shape[1]
        )
    report = IngestionTimings(
        operations=operations,
        environment={
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cache_policy": "Parsed cache bypassed; OS file cache not flushed; embeddings already computed.",
        },
    )
    atomic_bytes(output / "corpus-benchmark.json", (canonical_json(report) + "\n").encode())
    write_artifact_manifest(
        output,
        generated_at,
        seed,
        content_hash({"rounds": rounds}),
        {
            "selection": content_hash([r.model_dump(mode="json") for r in records]),
            "index_chunks": content_hash([c.id for c in chunks]),
        },
    )
    return report
