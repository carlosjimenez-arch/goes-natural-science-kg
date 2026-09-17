# src/goes_natural_science_kg/corpus/index.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Build and query a deterministic local cosine index with complete evidence.
"""Vectors are dense NumPy arrays; stable IDs break exact-score ties."""

import io
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.ingestion import EvidenceChunk


def build_index(path: Path, chunks: tuple[EvidenceChunk, ...], vectors: NDArray[np.float64]) -> str:
    if vectors.ndim != 2 or len(chunks) != len(vectors) or not chunks:
        raise ValueError("index needs one vector per chunk")
    if len({c.id for c in chunks}) != len(chunks) or not np.isfinite(vectors).all():
        raise ValueError("duplicate IDs or invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero embedding vector")
    order = np.argsort([c.id for c in chunks], kind="stable")
    records = tuple(chunks[i] for i in order)
    matrix = (vectors / norms)[order]
    content = "".join(canonical_json(c) + "\n" for c in records).encode()
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    data = buffer.getvalue()
    import hashlib

    digest = content_hash(
        {
            "chunks": hashlib.sha256(content).hexdigest(),
            "vectors": hashlib.sha256(data).hexdigest(),
            "index_version": 1,
        }
    )
    # Immutable generation, published by an atomic pointer only after both writes.
    generation = path / digest
    for target, payload in (
        (generation / "chunks.jsonl", content),
        (generation / "vectors.npy", data),
        (path / "current.json", (canonical_json({"generation": digest}) + "\n").encode()),
    ):
        try:
            unchanged = target.read_bytes() == payload
        except FileNotFoundError:
            unchanged = False
        if not unchanged:
            atomic_bytes(target, payload)
    return digest


def load_index(path: Path) -> tuple[tuple[EvidenceChunk, ...], NDArray[np.float64]]:
    import hashlib

    digest = json.loads((path / "current.json").read_text())["generation"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("invalid index generation")
    generation = path / digest
    content, data = (
        (generation / "chunks.jsonl").read_bytes(),
        (generation / "vectors.npy").read_bytes(),
    )
    if digest != content_hash(
        {
            "chunks": hashlib.sha256(content).hexdigest(),
            "vectors": hashlib.sha256(data).hexdigest(),
            "index_version": 1,
        }
    ):
        raise ValueError("index digest mismatch")
    chunks = tuple(EvidenceChunk.model_validate_json(line) for line in content.splitlines())
    vectors = np.load(io.BytesIO(data), allow_pickle=False)
    return chunks, vectors


def search_index(
    path: Path, query: NDArray[np.float64], limit: int = 5
) -> tuple[tuple[EvidenceChunk, float], ...]:
    chunks, matrix = load_index(path)
    norm = np.linalg.norm(query)
    if query.shape != (matrix.shape[1],) or not np.isfinite(query).all() or norm == 0 or limit < 1:
        raise ValueError("invalid query vector or limit")
    scores = matrix @ (query / norm)
    order = np.argsort(-scores, kind="stable")[:limit]
    return tuple((chunks[i], float(scores[i])) for i in order)
