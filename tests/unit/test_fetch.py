# tests/unit/test_fetch.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Check network-edge retries, gate rejection, cache integrity and redirects.
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from goes_natural_science_kg.corpus.discover import discover
from goes_natural_science_kg.corpus.fetch import blob_path, fetch_documents
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.ingestion import IngestionSettings

GOLDEN = Path(__file__).resolve().parents[1] / "golden/pdfs"
SETTINGS = IngestionSettings(project_id="offline-test-project", host_interval_seconds=0.1)
NOW = datetime(2026, 9, 14, tzinfo=UTC)


def source():
    record = discover(GOLDEN / "records.jsonl")[0]
    return SourceDocument.model_validate(
        record.document.model_dump() | {"sha256": None, "bytes": None, "downloaded_at": None}
    )


def test_cache_prevents_second_request(tmp_path):
    calls = []
    payload = (GOLDEN / "colombia-dba.pdf").read_bytes()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            first = await fetch_documents((source(),), tmp_path, SETTINGS, NOW, client)
            second = await fetch_documents((source(),), tmp_path, SETTINGS, NOW, client)
            return first, second

    first, second = asyncio.run(run())
    assert first[0].status == "fetched" and second[0].status == "cached"
    assert len(calls) == 1
    assert first[0].document.sha256 == second[0].document.sha256


def test_rejected_source_never_calls_network(tmp_path):
    rejected = SourceDocument.model_validate(
        source().model_dump()
        | {"license_gate_status": "rejected", "rejection_reason": "Unverified rights"}
    )

    def handler(request):
        raise AssertionError("rejected source reached HTTP")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_documents((rejected,), tmp_path, SETTINGS, NOW, client)

    assert asyncio.run(run())[0].status == "rejected"


def test_cross_host_redirect_is_not_followed(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://unreviewed.example/file.pdf"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_documents((source(),), tmp_path, SETTINGS, NOW, client)

    result = asyncio.run(run())[0]
    assert result.status == "failed" and "licence review" in result.error
    assert len(calls) == 1


def test_corrupt_cache_fails_without_redownload(tmp_path):
    import hashlib

    payload = (GOLDEN / "colombia-dba.pdf").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    record = SourceDocument.model_validate(
        source().model_dump() | {"sha256": digest, "bytes": len(payload), "downloaded_at": NOW}
    )
    path = blob_path(tmp_path, digest)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupt")

    def handler(request):
        raise AssertionError("corrupt cache silently redownloaded")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_documents((record,), tmp_path, SETTINGS, NOW, client)

    result = asyncio.run(run())[0]
    assert result.status == "failed" and "mismatch" in result.error


def test_retry_after_service_unavailable(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            503 if len(calls) == 1 else 200, content=(GOLDEN / "colombia-dba.pdf").read_bytes()
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_documents((source(),), tmp_path, SETTINGS, NOW, client)

    assert asyncio.run(run())[0].status == "fetched"
    assert len(calls) == 2
