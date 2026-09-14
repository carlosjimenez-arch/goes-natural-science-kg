# src/goes_natural_science_kg/corpus/pipeline.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Orchestrate licensed acquisition, structural evidence and real semantic indexing.
"""Each source failure is persisted and does not prevent processing the next source."""

import hashlib
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from pdfplumber.utils.exceptions import PdfminerException

from goes_natural_science_kg.agents.license_workflow import build_license_workflow
from goes_natural_science_kg.corpus.chunk import chunk_document, resolve_chunk
from goes_natural_science_kg.corpus.discover import coverage, discover
from goes_natural_science_kg.corpus.embed import embed_texts, pooled_vector
from goes_natural_science_kg.corpus.fetch import atomic_bytes, blob_path, fetch_documents
from goes_natural_science_kg.corpus.index import build_index
from goes_natural_science_kg.corpus.license_gate import evaluate_license, load_allowlist
from goes_natural_science_kg.corpus.manifest import read_manifest, write_manifest
from goes_natural_science_kg.corpus.parse import parse_pdf
from goes_natural_science_kg.schemas.artifacts import ArtifactDigest, ArtifactManifest
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.ingestion import (
    CorpusCoverage,
    DiscoveryRecord,
    DocumentOutcome,
    EmbeddingReceipt,
    EmbeddingRecord,
    EvidenceChunk,
    IngestionSettings,
    IngestionTimings,
    ParsedDocument,
    parse_plan_hash,
)
from goes_natural_science_kg.schemas.workflow import LicenseGateState


def write_artifact_manifest(
    directory: Path, generated_at: datetime, seed: int, settings_hash: str, inputs: dict[str, str]
) -> None:
    artifacts = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            data = path.read_bytes()
            artifacts[path.name] = ArtifactDigest(
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
            )
    manifest = ArtifactManifest(
        generated_at=generated_at,
        git_sha=None,
        git_dirty=None,
        seed=seed,
        settings_hash=settings_hash,
        inputs=inputs,
        artifacts=artifacts,
        notes="Git metadata unavailable: workspace is not a git repository. Timings are observations, not deterministic products.",
    )
    atomic_bytes(directory / "manifest.json", (canonical_json(manifest) + "\n").encode())


def parse_cached(root: Path, record: DiscoveryRecord) -> ParsedDocument:
    key = parse_plan_hash(record)
    path = root / "interim/parsed" / (key + ".json")
    if path.exists():
        parsed = ParsedDocument.model_validate_json(path.read_bytes())
        if (
            parsed.plan_sha256 != parse_plan_hash(record)
            or parsed.document_sha256 != record.document.sha256
        ):
            raise ValueError("parsed cache metadata mismatch")
        return parsed
    if not record.document.sha256:
        raise ValueError("source has no downloaded bytes")
    parsed = parse_pdf(blob_path(root / "raw", record.document.sha256), record)
    atomic_bytes(path, (canonical_json(parsed) + "\n").encode())
    return parsed


async def ingest(
    discovery_path: Path,
    root: Path,
    settings: IngestionSettings,
    generated_at: datetime,
    seed: int,
    *,
    embedding_client: httpx.AsyncClient | None = None,
    download_client: httpx.AsyncClient,
) -> CorpusCoverage:
    records = discover(discovery_path)
    manifest_path = root / "manifests/corpus.jsonl"
    previous = {d.id: d for d in read_manifest(manifest_path)} if manifest_path.exists() else {}
    policy = load_allowlist()
    # Discovery -> license_gate -> fetch. Explicit reviewed metadata is the input;
    # reuse previously fetched hashes only for the exact same URL/review identity.
    documents = []
    for record in records:
        source = record.document
        old = previous.get(source.id)
        if old and old.url == source.url and old.sha256:
            from goes_natural_science_kg.schemas.corpus import SourceDocument

            source = evaluate_license(source, policy, generated_at)
            source = SourceDocument.model_validate(
                source.model_dump() | old.model_dump(include={"sha256", "bytes", "downloaded_at"})
            )
        documents.append(evaluate_license(source, policy, generated_at))
    gate_state = LicenseGateState(
        documents=tuple(documents),
        allowlist=policy,
        checked_at=generated_at,
        manifest_path=manifest_path,
    )
    gate_result = await build_license_workflow().ainvoke(gate_state)
    documents = list(gate_result["results"])
    results = await fetch_documents(
        tuple(documents), root / "raw", settings, generated_at, download_client
    )
    write_manifest(manifest_path, tuple(r.document for r in results))
    atomic_bytes(
        root / "manifests/fetch-results.jsonl",
        "".join(canonical_json(r) + "\n" for r in results).encode(),
    )
    fetched = {r.document.id: r for r in results}
    outcomes: list[DocumentOutcome] = []
    all_chunks: list[EvidenceChunk] = []
    vectors = []
    updated: list[DiscoveryRecord] = []
    timings: dict[str, dict[str, float | int]] = {}
    receipt_path = root / "manifests/embedding-cache.jsonl"
    expected_hashes = (
        {
            r.request_sha256: r.response_sha256
            for r in (
                EmbeddingReceipt.model_validate_json(line)
                for line in receipt_path.read_text().splitlines()
            )
        }
        if receipt_path.exists()
        else {}
    )
    for record in records:
        result = fetched[record.document.id]
        record = DiscoveryRecord.model_validate(record.model_dump() | {"document": result.document})
        updated.append(record)
        if result.status in {"rejected", "failed"}:
            outcomes.append(
                DocumentOutcome(
                    source_id=record.document.id,
                    status=result.status,
                    error=result.error or result.document.rejection_reason,
                )
            )
            continue
        if not record.segmentation_review:
            outcomes.append(
                DocumentOutcome(
                    source_id=record.document.id,
                    status="needs_review",
                    error="No hash-bound structural review",
                )
            )
            continue
        try:
            started = time.perf_counter()
            parsed = parse_cached(root, record)
            parse_seconds = time.perf_counter() - started
            chunks = chunk_document(parsed, record.boundary_pattern)
            # Each citation remains intact; paragraph views are embedded independently.
            anchors = {p.id: p for p in parsed.paragraphs if p.text.strip()}
            texts = tuple(p.text for p in anchors.values())
            embeddings = await embed_texts(
                texts,
                settings,
                root / "interim/embeddings",
                client=embedding_client,
                expected_hashes=expected_hashes,
            )
            by_id = dict(zip(anchors, embeddings, strict=True))
            local_vectors = []
            for chunk in chunks:
                resolve_chunk(chunk, parsed)
                local_vectors.append(
                    pooled_vector(tuple(by_id[p] for p in chunk.paragraph_ids if p in by_id))
                )
            parsed_digest = content_hash(parsed)
            atomic_bytes(
                root / "interim/evidence" / (parsed_digest + ".json"),
                (canonical_json(parsed) + "\n").encode(),
            )
            all_chunks.extend(chunks)
            vectors.extend(local_vectors)
            pages = len({p.page for p in parsed.paragraphs})
            outcomes.append(
                DocumentOutcome(
                    source_id=record.document.id,
                    status="indexed",
                    parsed_sha256=parsed_digest,
                    chunk_ids=tuple(c.id for c in chunks),
                    pages=pages,
                    paragraphs=len(parsed.paragraphs),
                    tables=len(parsed.tables),
                )
            )
            timings[record.document.id] = {
                "parse_or_cache_seconds": parse_seconds,
                "pages": pages,
                "chunks": len(chunks),
            }
        except (ValueError, OSError, httpx.HTTPError, PdfminerException) as error:
            outcomes.append(
                DocumentOutcome(
                    source_id=record.document.id,
                    status="failed",
                    error=str(error) or type(error).__name__,
                )
            )
        # Durable progress even when a later source or API call fails.
        atomic_bytes(
            root / "manifests/ingestion-outcomes.jsonl",
            "".join(canonical_json(o) + "\n" for o in outcomes).encode(),
        )
    if all_chunks:
        started = time.perf_counter()
        build_index(
            root / "interim/index", tuple(all_chunks), np.asarray(vectors, dtype=np.float64)
        )
        timings["index"] = {
            "seconds": time.perf_counter() - started,
            "chunks": len(all_chunks),
            "dimensions": settings.embedding_dimensions,
        }
    report = coverage(tuple(updated), tuple(outcomes))
    atomic_bytes(
        root / "manifests/ingestion-outcomes.jsonl",
        "".join(canonical_json(o) + "\n" for o in outcomes).encode(),
    )
    atomic_bytes(root / "manifests/coverage.json", (canonical_json(report) + "\n").encode())
    atomic_bytes(
        root / "manifests/run-timings.json",
        (canonical_json(IngestionTimings.model_validate({"operations": timings})) + "\n").encode(),
    )
    write_embedding_receipts(root, receipt_path)
    write_artifact_manifest(
        root / "manifests",
        generated_at,
        seed,
        content_hash(settings),
        {
            str(discovery_path): hashlib.sha256(discovery_path.read_bytes()).hexdigest(),
            "source_allowlist": content_hash(policy),
        },
    )
    return report


def write_embedding_receipts(root: Path, receipt_path: Path) -> None:
    receipts = []
    for path in sorted((root / "interim/embeddings").glob("*.json")):
        if path.name == "manifest.json":
            continue
        embedding = EmbeddingRecord.model_validate_json(path.read_bytes())
        receipts.append(
            EmbeddingReceipt(
                request_sha256=embedding.request_sha256,
                response_sha256=content_hash(embedding),
                model=embedding.model,
                dimensions=len(embedding.values),
            )
        )
    atomic_bytes(receipt_path, "".join(canonical_json(r) + "\n" for r in receipts).encode())
