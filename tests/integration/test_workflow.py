# tests/integration/test_workflow.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Run the actual LangGraph workflow and persist both accepted and rejected sources.
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from goes_natural_science_kg.agents.license_workflow import build_license_workflow
from goes_natural_science_kg.cli import app
from goes_natural_science_kg.corpus.license_gate import load_allowlist
from goes_natural_science_kg.corpus.manifest import read_manifest, write_manifest
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.workflow import LicenseGateState

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def test_rejection_is_recorded_and_batch_continues(tmp_path):
    docs = TypeAdapter(tuple[SourceDocument, ...]).validate_json(
        (GOLDEN / "source-candidates.json").read_bytes()
    )
    state = LicenseGateState(
        documents=docs,
        allowlist=load_allowlist(),
        checked_at=datetime(2026, 9, 14, tzinfo=UTC),
        manifest_path=tmp_path / "corpus.jsonl",
    )
    graph = build_license_workflow()
    assert "license_gate" in graph.nodes
    result = LicenseGateState.model_validate(graph.invoke(state))
    assert len(result.results) == 3 and len(result.ingestion_ready_ids) == 2
    records = read_manifest(state.manifest_path)
    assert len(records) == 3
    assert len([d for d in records if d.license_gate_status.value == "rejected"]) == 1
    first = state.manifest_path.read_bytes()
    graph.invoke(state)
    assert state.manifest_path.read_bytes() == first
    write_manifest(state.manifest_path, (result.results[0],))
    assert state.manifest_path.read_bytes() == first
    assert not (tmp_path / "raw").exists()


def test_corrupt_manifest_is_not_silently_discarded(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text("{bad json")
    with pytest.raises(ValueError):
        read_manifest(path)
    with pytest.raises(ValueError):
        write_manifest(path, ())
    assert path.read_text() == "{bad json"


def test_cli_diff_and_license_workflow(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app, ["graph-diff", str(GOLDEN / "graph-before.json"), str(GOLDEN / "graph-after.json")]
    )
    assert result.exit_code == 0, result.output
    assert "nodes: +1 -1 ~2" in result.output
    result = runner.invoke(
        app,
        [
            "graph-diff",
            str(GOLDEN / "graph-before.json"),
            str(GOLDEN / "graph-after.json"),
            "--json",
        ],
    )
    assert json.loads(result.output)["schema_version"] == "graph-diff/1.0"
    result = runner.invoke(
        app,
        [
            "corpus-gate",
            str(GOLDEN / "source-candidates.json"),
            "--manifest",
            str(tmp_path / "sources.jsonl"),
            "--checked-at",
            "2026-09-14T00:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "accepted=2 rejected=1" in result.output
