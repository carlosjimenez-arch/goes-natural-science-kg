# src/goes_natural_science_kg/corpus/commands.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Provide explicit online/offline transport boundaries for corpus commands.
"""ADC credentials remain in memory; no tokens or account identifiers are serialized."""

from datetime import datetime
from pathlib import Path

import httpx

from goes_natural_science_kg.corpus.embed import embed_texts, pooled_vector
from goes_natural_science_kg.corpus.index import search_index
from goes_natural_science_kg.corpus.pipeline import ingest
from goes_natural_science_kg.corpus.trace import trace_references
from goes_natural_science_kg.schemas.ingestion import CorpusCoverage, IngestionSettings


def adc_headers() -> dict[str, str]:
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())  # type: ignore[no-untyped-call]
    if not isinstance(credentials.token, str):
        raise ValueError("ADC did not return an access token")
    return {"Authorization": "Bearer " + credentials.token}


def offline_transport(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("offline run attempted an uncached download", request=request)


async def run_ingestion(
    discovery: Path,
    root: Path,
    settings: IngestionSettings,
    generated_at: datetime,
    seed: int,
    online: bool,
) -> CorpusCoverage:
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds,
        headers={"User-Agent": "GOES-Science-Corpus/0.1 (educational research)"},
        transport=None if online else httpx.MockTransport(offline_transport),
    ) as downloads:
        if not online:
            return await ingest(
                discovery, root, settings, generated_at, seed, download_client=downloads
            )
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds, headers=adc_headers()
        ) as embeddings:
            return await ingest(
                discovery,
                root,
                settings,
                generated_at,
                seed,
                download_client=downloads,
                embedding_client=embeddings,
            )


async def run_search(
    query: str, root: Path, settings: IngestionSettings, online: bool, limit: int
) -> tuple[dict[str, object], ...]:
    if online:
        async with httpx.AsyncClient(
            timeout=settings.timeout_seconds, headers=adc_headers()
        ) as client:
            records = await embed_texts(
                (query,), settings, root / "interim/embeddings", "RETRIEVAL_QUERY", client=client
            )
    else:
        records = await embed_texts(
            (query,), settings, root / "interim/embeddings", "RETRIEVAL_QUERY"
        )
    hits = search_index(root / "interim/index", pooled_vector(records), limit)
    return tuple(
        {
            "chunk_id": chunk.id,
            "score": score,
            "grades": chunk.grades,
            "evidence": trace_references(root, (chunk.id,))[0],
        }
        for chunk, score in hits
    )
