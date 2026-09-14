# src/goes_natural_science_kg/corpus/embed.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Cache real Vertex embedding responses without truncating evidence.
"""Internal paragraph views never replace the intact cited evidence chunk."""

import asyncio
import hashlib
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from numpy.typing import NDArray

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.ingestion import EmbeddingRecord, IngestionSettings


def embedding_request(text: str, settings: IngestionSettings, task: str) -> dict[str, object]:
    return {
        "cache_version": 1,
        "model": settings.embedding_model,
        "location": settings.location,
        "dimensions": settings.embedding_dimensions,
        "task": task,
        "text": text,
        "autoTruncate": False,
    }


async def embed_texts(
    texts: tuple[str, ...],
    settings: IngestionSettings,
    cache: Path,
    task: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "RETRIEVAL_DOCUMENT",
    *,
    client: httpx.AsyncClient | None = None,
    expected_hashes: dict[str, str] | None = None,
) -> tuple[EmbeddingRecord, ...]:
    """Cache-only unless an authenticated client is explicitly supplied by the runner."""
    semaphore = asyncio.Semaphore(settings.concurrency)
    unique = dict.fromkeys(texts)

    async def one(text: str) -> EmbeddingRecord:
        if not text.strip():
            raise ValueError("empty embedding input")
        key = content_hash(embedding_request(text, settings, task))
        path = cache / (key + ".json")
        text_digest = hashlib.sha256(text.encode()).hexdigest()
        if path.exists():
            record = EmbeddingRecord.model_validate_json(path.read_bytes())
            if (
                record.request_sha256 != key
                or record.text_sha256 != text_digest
                or record.model != settings.embedding_model
                or record.task != task
                or len(record.values) != settings.embedding_dimensions
            ):
                raise ValueError("embedding cache metadata mismatch")
            verify_receipt(record, expected_hashes)
            return record
        if client is None:
            raise FileNotFoundError(f"offline embedding cache miss: {key}")
        url = (
            f"https://{settings.location}-aiplatform.googleapis.com/v1/projects/{settings.project_id}"
            f"/locations/{settings.location}/publishers/google/models/{settings.embedding_model}:predict"
        )
        async with semaphore:
            response = await request_with_retries(client, url, text, task, settings)
        item = response.json()["predictions"][0]["embeddings"]
        record = EmbeddingRecord(
            request_sha256=key,
            model=settings.embedding_model,
            task=task,
            text_sha256=text_digest,
            values=item["values"],
            token_count=item["statistics"]["token_count"],
            truncated=item["statistics"]["truncated"],
            vector_sha256=content_hash(item["values"]),
        )
        if len(record.values) != settings.embedding_dimensions:
            raise ValueError("unexpected embedding dimensions")
        verify_receipt(record, expected_hashes)
        atomic_bytes(path, (canonical_json(record) + "\n").encode())
        return record

    records = await asyncio.gather(*(one(text) for text in unique))
    mapping = dict(zip(unique, records, strict=True))
    return tuple(mapping[text] for text in texts)


def verify_receipt(record: EmbeddingRecord, expected: dict[str, str] | None) -> None:
    if (
        expected
        and record.request_sha256 in expected
        and content_hash(record) != expected[record.request_sha256]
    ):
        raise ValueError(
            "embedding response changed from its versioned receipt; review a new embedding revision"
        )


async def request_with_retries(
    client: httpx.AsyncClient, url: str, text: str, task: str, settings: IngestionSettings
) -> httpx.Response:
    for attempt in range(settings.attempts):
        response = await client.post(
            url,
            json={
                "instances": [{"content": text, "task_type": task}],
                "parameters": {
                    "autoTruncate": False,
                    "outputDimensionality": settings.embedding_dimensions,
                },
            },
        )
        if (
            response.status_code not in {429, 500, 502, 503, 504}
            or attempt + 1 == settings.attempts
        ):
            response.raise_for_status()
            return response
        await asyncio.sleep(2.0**attempt)
    raise RuntimeError("unreachable retry state")


def pooled_vector(records: tuple[EmbeddingRecord, ...]) -> NDArray[np.float64]:
    """Token-weighted mean, L2-normalized; retrieval representation, not mastery evidence."""
    if not records or len({(r.model, r.task, len(r.values)) for r in records}) != 1:
        raise ValueError("pooling requires nonempty homogeneous embeddings")
    matrix = np.asarray([r.values for r in records], dtype=np.float64)
    weights = np.asarray([max(1, r.token_count) for r in records], dtype=np.float64)
    vector = np.average(matrix, axis=0, weights=weights)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid pooled embedding")
    return vector / norm
