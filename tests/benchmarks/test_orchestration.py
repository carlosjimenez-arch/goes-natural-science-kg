# tests/benchmarks/test_orchestration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import asyncio
import json
from pathlib import Path

from goes_natural_science_kg.agents.checks import graph_errors, materialize, verified_quote_span
from goes_natural_science_kg.agents.runner import execute, make_report
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    Assignment,
    Decomposition,
    EvidencePacket,
    MicroProposal,
    OrchestrationInput,
    OrchestrationSettings,
    WorkItem,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/orchestration"


def test_sparse_graph_validation(benchmark):
    request = OrchestrationInput.model_validate_json((GOLDEN / "input.json").read_bytes())
    parent = request.skills[0].skill.id
    task = WorkItem(
        id="bench",
        level="L2",
        instructions="Benchmark",
        assignments=(
            Assignment(
                skill_id=parent,
                grade=2,
                domain="physical-science",
                minutes=9600,
                instructions="Benchmark",
            ),
        ),
    )
    proposals = tuple(
        MicroProposal(
            key=f"task-{i}",
            observable_verb="Compare",
            knowledge_object=f"sample-{i}",
            cognitive_domain="applying",
            grade_band={"minimum": 2, "maximum": 2},
            estimated_minutes=1,
            evidence_of_mastery="Record comparison",
            source_refs=request.skills[0].source_refs,
            confidence=0.8,
            task=f"Compare sample {i}",
            atomicity_reason="Benchmark task",
            prerequisites=({"id": f"task-{i - 1}", "type": "PREREQUISITE"},) if i else (),
        )
        for i in range(1000)
    )
    micros = materialize(
        task, Decomposition(micros=proposals, granularity_rationale="Performance fixture only")
    )
    assert benchmark(graph_errors, micros, {parent}) == ()


def test_quote_offset_verification(benchmark):
    packet = EvidencePacket.model_validate(json.loads((GOLDEN / "evidence.json").read_text())[0])
    anchor = next(a for a in packet.anchors if "fluidez" in a.text)
    assert (
        benchmark(verified_quote_span, anchor.text, "fluidez, viscosidad, transparencia")
        is not None
    )


def test_recorded_hierarchy_with_sqlite(benchmark, tmp_path):
    rt = AgentRuntime(
        request=OrchestrationInput.model_validate_json((GOLDEN / "input.json").read_bytes()),
        settings=OrchestrationSettings.model_validate_json((GOLDEN / "settings.json").read_bytes()),
        evidence=tuple(
            EvidencePacket.model_validate(x)
            for x in json.loads((GOLDEN / "evidence.json").read_text())
        ),
        prompts=ROOT / "prompts",
        cache=GOLDEN / "cassettes",
    )
    calls = []

    def run():
        path = tmp_path / f"run-{len(calls)}.sqlite"
        calls.append(path)
        return make_report(rt, asyncio.run(execute(rt, path)))

    result = benchmark.pedantic(run, rounds=3, iterations=1)
    assert {i.level for i in result.items} == {"L0", "L1", "L2", "L3"}
