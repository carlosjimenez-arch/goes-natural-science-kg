# tests/integration/test_ingestion.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify real PDF layout, complete evidence and recorded real embeddings end to end.
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from goes_natural_science_kg.corpus.chunk import chunk_document, resolve_chunk
from goes_natural_science_kg.corpus.discover import discover
from goes_natural_science_kg.corpus.embed import embed_texts, pooled_vector
from goes_natural_science_kg.corpus.index import build_index, load_index, search_index
from goes_natural_science_kg.corpus.parse import parse_pdf
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.ingestion import EvidenceReadyMicroSkill, IngestionSettings

GOLDEN = Path(__file__).resolve().parents[1] / "golden/pdfs"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_real_pdf_pipeline_and_trace(position, tmp_path, before):
    records = discover(GOLDEN / "records.jsonl")
    record = records[position]
    path = GOLDEN / (record.document.identity.key + ".pdf")
    parsed = parse_pdf(path, record)
    assert canonical_json(parsed) + "\n" == path.with_suffix(".pdf.expected.json").read_text()
    assert parsed.tables
    chunks = chunk_document(parsed)
    assert len(chunks) == 1  # These excerpts are complete two-page curricular units.
    assert "".join(c.text for c in chunks) == parsed.text
    assert {p.page for p in parsed.paragraphs} == set(record.excerpt.source_pages)
    settings = IngestionSettings(project_id="recorded-golden-project")
    # Real cached Vertex responses, not invented vectors or a mocked model.
    records = asyncio.run(
        embed_texts(tuple(p.text for p in parsed.paragraphs), settings, GOLDEN / "embeddings")
    )
    vector = pooled_vector(records)
    generation = build_index(tmp_path, chunks, np.asarray([vector]))
    assert build_index(tmp_path, chunks, np.asarray([vector])) == generation
    found, score = search_index(tmp_path, vector, 1)[0]
    assert found == chunks[0] and score == pytest.approx(1.0)
    micro = EvidenceReadyMicroSkill.model_validate(
        before.micro_skills[0].model_dump()
        | {"schema_version": "micro-skill/2.0", "source_refs": [found.id]}
    )
    registry = {c.id: c for c in load_index(tmp_path)[0]}
    anchors = resolve_chunk(registry[micro.source_refs[0]], parsed)
    assert all(a["document_sha256"] == record.document.sha256 for a in anchors)
    assert "".join(a["text"] for a in anchors) == found.text
    assert all(a["page"] in record.excerpt.source_pages for a in anchors)


def test_reviewed_objectives_are_not_interleaved():
    expected = json.loads((GOLDEN / "colombia-dba.pdf.expected.json").read_text())
    text = expected["text"]
    assert (
        "Comprende que las sustancias pueden\nencontrarse en distintos estados (sólido, líquido\ny gaseoso)."
        in text
    )
    assert "del que está hecho." in text
    mined = json.loads((GOLDEN / "el-salvador-science-2026.pdf.expected.json").read_text())
    assert "1.11. Compara las adaptaciones de dos" in mined["text"]
    assert {p["page"] for p in mined["paragraphs"]} == {33, 34}


def test_unreviewed_or_modified_pdf_is_rejected(tmp_path):
    from goes_natural_science_kg.schemas.ingestion import DiscoveryRecord

    record = discover(GOLDEN / "records.jsonl")[0]
    changed = tmp_path / "modified.pdf"
    changed.write_bytes(
        (GOLDEN / (record.document.identity.key + ".pdf")).read_bytes() + b"changed"
    )
    with pytest.raises(ValueError, match="manifest"):
        parse_pdf(changed, record)
    unreviewed = DiscoveryRecord.model_validate(record.model_dump() | {"segmentation_review": None})
    with pytest.raises(ValueError, match="reviewed"):
        parse_pdf(GOLDEN / (record.document.identity.key + ".pdf"), unreviewed)


def test_full_ingestion_network_replay_is_rebuildable(tmp_path):
    import hashlib
    import shutil
    from datetime import UTC, datetime

    import httpx

    from goes_natural_science_kg.corpus.pipeline import ingest
    from goes_natural_science_kg.corpus.trace import trace_references
    from goes_natural_science_kg.schemas.corpus import SourceDocument
    from goes_natural_science_kg.schemas.ingestion import DiscoveryRecord, SectionPlan

    record = discover(GOLDEN / "records.jsonl")[0]
    payload = (GOLDEN / (record.document.identity.key + ".pdf")).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # At the allowed HTTP edge, replay the real excerpt as the served document.
    # This test-only source revision is never written to the real corpus manifest.
    document = SourceDocument.model_validate(
        record.document.model_dump()
        | {
            "sha256": None,
            "bytes": None,
            "downloaded_at": None,
            "title": "Network replay of attributed real PDF excerpt",
        }
    )
    replay = DiscoveryRecord.model_validate(
        record.model_dump()
        | {
            "document": document,
            "excerpt": None,
            "reviewed_sha256": digest,
            "sections": [
                SectionPlan(
                    key="unit", title="Complete test excerpt", first_page=1, last_page=2, grades=[2]
                )
            ],
        }
    )
    rejected = DiscoveryRecord.model_validate(
        replay.model_dump()
        | {
            "document": SourceDocument.model_validate(
                document.model_dump()
                | {
                    "url": "https://unverified.example/curriculum.pdf",
                    "license": "unknown",
                    "license_evidence": None,
                    "license_gate_status": "pending",
                    "checked_at": None,
                    "policy_version": None,
                }
            )
        }
    )
    # Use a different canonical identity for the rejected document.
    from goes_natural_science_kg.schemas.identity import Identity, stable_id

    identity = Identity(namespace="test-replay", key="rejected")
    bad_doc = SourceDocument.model_validate(
        rejected.document.model_dump() | {"identity": identity, "id": stable_id("source", identity)}
    )
    rejected = DiscoveryRecord.model_validate(rejected.model_dump() | {"document": bad_doc})
    selection = tmp_path / "discovery.jsonl"
    selection.write_text(canonical_json(replay) + "\n" + canonical_json(rejected) + "\n")
    shutil.copytree(GOLDEN / "embeddings", tmp_path / "interim/embeddings")
    calls = []

    def handler(request):
        calls.append(str(request.url))
        assert request.url.host != "unverified.example"
        return httpx.Response(200, content=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ingest(
                selection,
                tmp_path,
                IngestionSettings(project_id="recorded-golden-project"),
                datetime(2026, 9, 14, tzinfo=UTC),
                0,
                download_client=client,
            )

    first = asyncio.run(run())
    generation = (tmp_path / "interim/index/current.json").read_bytes()
    from goes_natural_science_kg.corpus.commands import run_ingestion

    second = asyncio.run(
        run_ingestion(
            selection,
            tmp_path,
            IngestionSettings(project_id="recorded-golden-project"),
            datetime(2026, 9, 14, tzinfo=UTC),
            0,
            False,
        )
    )
    assert first == second and len(calls) == 1
    assert (tmp_path / "interim/index/current.json").read_bytes() == generation
    chunks, _ = load_index(tmp_path / "interim/index")
    trace = trace_references(tmp_path, (chunks[0].id,))[0]
    assert trace["source_url"] == str(document.url)
    assert {a["page"] for a in trace["anchors"]} == {1, 2}
    assert not first.complete
    from goes_natural_science_kg.corpus.benchmark import benchmark_corpus

    measured = benchmark_corpus(
        selection, tmp_path, tmp_path / "benchmark-report", datetime(2026, 9, 14, tzinfo=UTC), 0, 1
    )
    assert measured.operations["index-round-1"].chunks == 1
    assert measured.operations["index-round-1"].seconds > 0
    from typer.testing import CliRunner

    from goes_natural_science_kg.cli import app

    result = CliRunner().invoke(app, ["corpus-trace", chunks[0].id, "--data-dir", str(tmp_path)])
    assert result.exit_code == 0 and str(document.url) in result.output


def test_review_timestamp_does_not_change_evidence_ids():
    from datetime import UTC, datetime

    from goes_natural_science_kg.schemas.ingestion import DiscoveryRecord, parse_plan_hash

    record = discover(GOLDEN / "records.jsonl")[0]
    refreshed = DiscoveryRecord.model_validate(
        record.model_dump()
        | {
            "discovered_at": datetime(2026, 9, 15, tzinfo=UTC),
            "segmentation_review": "Same verified geometry; renewed operational review.",
        }
    )
    assert parse_plan_hash(refreshed) == parse_plan_hash(record)
    path = GOLDEN / (record.document.identity.key + ".pdf")
    assert chunk_document(parse_pdf(path, refreshed)) == chunk_document(parse_pdf(path, record))
