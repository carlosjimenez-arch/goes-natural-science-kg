# tests/benchmarks/test_operations.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure graph comparison and corpus-gate/manifest traversal on real contracts.
import json
from datetime import UTC, datetime
from pathlib import Path

from goes_natural_science_kg.agents.license_workflow import build_license_workflow
from goes_natural_science_kg.corpus.license_gate import evaluate_license, load_allowlist
from goes_natural_science_kg.corpus.manifest import read_manifest, write_manifest
from goes_natural_science_kg.graph.diff import diff_graphs, summarize_diff
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.graph import GraphSnapshot
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.registry import schema_exports
from goes_natural_science_kg.schemas.skills import FrameworkRef, Skill
from goes_natural_science_kg.schemas.workflow import LicenseGateState


def test_diff_1000_nodes(benchmark):
    nodes = []
    for i in range(1000):
        identity = Identity(namespace="benchmark", key=f"skill-{i}")
        nodes.append(
            Skill(
                id=stable_id("skill", identity),
                identity=identity,
                label=f"Skill {i}",
                origin_framework=FrameworkRef(name="test", version="1"),
                domain="matter",
                hierarchy_level=0,
            )
        )
    snapshot = GraphSnapshot(version="v1", skills=tuple(nodes))
    benchmark(diff_graphs, snapshot, snapshot)


def test_diff_summary(benchmark, before, after):
    benchmark(summarize_diff, diff_graphs(before, after))


def test_license_gate(benchmark):
    data = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/source-candidates.json").read_text()
    )
    benchmark(
        evaluate_license,
        SourceDocument.model_validate(data[0]),
        load_allowlist(),
        datetime(2026, 9, 14, tzinfo=UTC),
    )


def test_manifest_roundtrip(benchmark, tmp_path):
    data = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/source-candidates.json").read_text()
    )
    doc = evaluate_license(
        SourceDocument.model_validate(data[0]), load_allowlist(), datetime(2026, 9, 14, tzinfo=UTC)
    )
    path = tmp_path / "sources.jsonl"

    def roundtrip():
        write_manifest(path, (doc,))
        return read_manifest(path)

    benchmark(roundtrip)


def test_gate_workflow(benchmark, tmp_path):
    data = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/source-candidates.json").read_text()
    )
    state = LicenseGateState(
        documents=tuple(SourceDocument.model_validate(d) for d in data),
        allowlist=load_allowlist(),
        checked_at=datetime(2026, 9, 14, tzinfo=UTC),
        manifest_path=tmp_path / "sources.jsonl",
    )
    benchmark(build_license_workflow().invoke, state)


def test_schema_exports(benchmark):
    benchmark(schema_exports)
