# src/goes_natural_science_kg/corpus/fetch.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Fetch accepted PDFs politely into a verified content-addressed cache.
"""One writer per run; rejected sources never reach the HTTP client."""

import asyncio
import hashlib
import ipaddress
import os
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from goes_natural_science_kg.schemas.corpus import GateStatus, SourceDocument
from goes_natural_science_kg.schemas.ingestion import FetchResult, IngestionSettings


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        name = handle.name
        handle.write(payload)
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def blob_path(root: Path, digest: str) -> Path:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid content digest")
    return root / "sha256" / digest[:2] / (digest + ".pdf")


def cached_blob(root: Path, source: SourceDocument) -> Path | None:
    if source.sha256 is None:
        return None
    path = blob_path(root, source.sha256)
    if not path.exists():
        return None
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != source.sha256 or len(payload) != source.bytes:
        raise ValueError("cached PDF digest/size mismatch")
    return path


def public_https(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("only public HTTPS document URLs are supported")
    if parts.hostname in {"localhost", "metadata.google.internal"}:
        raise ValueError("private host is not a corpus source")
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("private IP is not a corpus source")


async def download_pdf(
    client: httpx.AsyncClient, source: SourceDocument, settings: IngestionSettings
) -> bytes:
    """Redirects stay on the reviewed host; new hosts require discovery and review."""
    url = str(source.url)
    for _ in range(6):
        public_https(url)
        async with client.stream("GET", url) as response:
            if response.is_redirect:
                target = str(response.url.join(response.headers["location"]))
                if urlsplit(target).hostname != urlsplit(url).hostname:
                    raise ValueError("cross-host redirect requires explicit licence review")
                url = target
                continue
            response.raise_for_status()
            payload = bytearray()
            async for piece in response.aiter_bytes():
                payload.extend(piece)
                if len(payload) > settings.max_download_bytes:
                    raise ValueError("PDF exceeds configured download limit")
            if not payload.startswith(b"%PDF-"):
                raise ValueError("response is not a PDF")
            return bytes(payload)
    raise ValueError("too many redirects")


async def fetch_documents(
    documents: tuple[SourceDocument, ...],
    root: Path,
    settings: IngestionSettings,
    downloaded_at: datetime,
    client: httpx.AsyncClient,
) -> tuple[FetchResult, ...]:
    semaphore = asyncio.Semaphore(settings.concurrency)
    locks = {str(d.url.host): asyncio.Lock() for d in documents}
    next_request: dict[str, float] = {}
    known = {str(d.url): d for d in documents if d.sha256}

    async def fetch_one(source: SourceDocument) -> FetchResult:
        if source.license_gate_status != GateStatus.ACCEPTED:
            return FetchResult(document=source, status="rejected")
        host = str(source.url.host)
        async with locks[host]:
            receipt = (
                root / "receipts" / (hashlib.sha256(str(source.url).encode()).hexdigest() + ".json")
            )
            previous = known.get(str(source.url), source)
            try:
                previous = restore_receipt(receipt, source, previous)
                if cached_blob(root, previous):
                    fields = previous.model_dump(include={"sha256", "bytes", "downloaded_at"})
                    restored = SourceDocument.model_validate(source.model_dump() | fields)
                    return FetchResult(document=restored, status="cached")
                payload = await with_retries(
                    source, host, client, settings, semaphore, next_request
                )
                digest = hashlib.sha256(payload).hexdigest()
                if source.sha256 and digest != source.sha256:
                    raise ValueError("remote PDF changed; review a new source revision")
                target = blob_path(root, digest)
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError("content-addressed cache collision or corruption")
                if not target.exists():
                    atomic_bytes(target, payload)
                result = SourceDocument.model_validate(
                    source.model_dump()
                    | {
                        "sha256": digest,
                        "bytes": len(payload),
                        "downloaded_at": downloaded_at,
                    }
                )
                known[str(source.url)] = result
                atomic_bytes(receipt, (result.model_dump_json() + "\n").encode())
                return FetchResult(document=result, status="fetched")
            except (httpx.HTTPError, ValueError, OSError) as error:
                return FetchResult(
                    document=source, status="failed", error=str(error) or type(error).__name__
                )

    return tuple(
        await asyncio.gather(*(fetch_one(d) for d in sorted(documents, key=lambda d: d.id)))
    )


async def with_retries(
    source: SourceDocument,
    host: str,
    client: httpx.AsyncClient,
    settings: IngestionSettings,
    semaphore: asyncio.Semaphore,
    next_request: dict[str, float],
) -> bytes:
    loop = asyncio.get_running_loop()
    for attempt in range(settings.attempts):
        await asyncio.sleep(max(0, next_request.get(host, 0) - loop.time()))
        try:
            async with semaphore:
                next_request[host] = loop.time() + settings.host_interval_seconds
                return await download_pdf(client, source, settings)
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            retry_after = 0.0
            if isinstance(error, httpx.HTTPStatusError):
                if error.response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise
                value = error.response.headers.get("retry-after", "0")
                retry_after = float(value) if value.isdecimal() else 0.0
            if attempt + 1 == settings.attempts:
                raise
            next_request[host] = loop.time() + max(2.0**attempt, retry_after)
    raise RuntimeError("unreachable retry state")


def restore_receipt(path: Path, source: SourceDocument, previous: SourceDocument) -> SourceDocument:
    if path.exists():
        previous = SourceDocument.model_validate_json(path.read_bytes())
    if previous.url != source.url or (source.sha256 and source.sha256 != previous.sha256):
        raise ValueError("download receipt conflicts with pinned source")
    return previous
