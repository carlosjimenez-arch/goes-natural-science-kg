# tests/integration/test_agent_hierarchy.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import asyncio
import json
import shutil
from datetime import UTC
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from goes_natural_science_kg.agents.runner import execute, make_report
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    EvidencePacket,
    OrchestrationInput,
    OrchestrationReport,
    OrchestrationSettings,
    RootPlan,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/orchestration"


def runtime():
    return AgentRuntime(
        request=OrchestrationInput.model_validate_json((GOLDEN / "input.json").read_bytes()),
        settings=OrchestrationSettings.model_validate_json((GOLDEN / "settings.json").read_bytes()),
        evidence=tuple(
            EvidencePacket.model_validate(x)
            for x in json.loads((GOLDEN / "evidence.json").read_text())
        ),
        prompts=ROOT / "prompts",
        cache=GOLDEN / "cassettes",
    )


def test_real_recorded_hierarchy_replays_every_level_offline(tmp_path):
    rt = runtime()
    expected = OrchestrationReport.model_validate_json((GOLDEN / "report.json").read_bytes())
    result = make_report(rt, asyncio.run(execute(rt, tmp_path / "run.sqlite")))
    assert canonical_json(result) == canonical_json(expected)
    assert {i.level for i in result.items} == {"L0", "L1", "L2", "L3"}
    assert all(i.revisions <= 3 and len(i.ballots) == i.revisions + 1 for i in result.items)
    assert all(len(b.verdicts) == 5 for i in result.items for b in i.ballots)
    assert all(c.total_minutes <= 9600 for c in result.approved_curricula)


def test_sqlite_resume_survives_new_graph_and_saver(tmp_path):
    rt = runtime()
    database = tmp_path / "resume.sqlite"
    partial = asyncio.run(execute(rt, database, interrupt_before=["curricularization"]))
    assert not any(v.level == "L3" for v in partial["outcomes"].values())

    async def namespaces():
        async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
            return [
                str(x.config["configurable"].get("checkpoint_ns", ""))
                async for x in saver.alist(None)
            ]

    assert any("worker" in ns for ns in asyncio.run(namespaces()))
    resumed = make_report(rt, asyncio.run(execute(rt, database, resume=True)))
    expected = OrchestrationReport.model_validate_json((GOLDEN / "report.json").read_bytes())
    assert canonical_json(resumed) == canonical_json(expected)
    with pytest.raises(ValueError, match="checkpoint exists"):
        asyncio.run(execute(rt, database))
    with pytest.raises(ValueError, match="no matching checkpoint"):
        asyncio.run(
            execute(
                rt.model_copy(update={"settings": rt.settings.model_copy(update={"seed": 1})}),
                database,
                resume=True,
            )
        )


def test_cache_miss_inside_child_is_resumable_without_fabrication(tmp_path):
    rt = runtime()
    cache = tmp_path / "cassettes"
    shutil.copytree(rt.cache, cache)
    expected = OrchestrationReport.model_validate_json((GOLDEN / "report.json").read_bytes())
    # Remove all evidence-judge records: this necessarily interrupts the child panel.
    removed = {}
    for p in cache.glob("[0-9a-f]*.json"):
        if json.loads(p.read_text())["request"]["prompt_id"] == "evidence-judge":
            removed[p.name] = p.read_bytes()
            p.unlink()
    rt = rt.model_copy(update={"cache": cache})
    database = tmp_path / "interrupted.sqlite"
    with pytest.raises(FileNotFoundError, match="offline generation cache miss"):
        asyncio.run(execute(rt, database))
    for name, data in removed.items():
        (cache / name).write_bytes(data)
    resumed = make_report(rt, asyncio.run(execute(rt, database, resume=True)))
    assert canonical_json(resumed) == canonical_json(expected)


def test_real_invalid_response_is_rejected_by_contract():
    records = [json.loads(p.read_text()) for p in (GOLDEN / "cassettes").glob("[0-9a-f]*.json")]
    from goes_natural_science_kg.schemas.orchestration import CachedGeneration

    for record in records:
        CachedGeneration.model_validate(record)
    invalid = next(
        r
        for r in records
        if r["request"]["prompt_id"] == "optimizer-l0"
        and json.loads(r["response"]).get("schema_version") == "v1.0"
    )
    with pytest.raises(ValueError):
        RootPlan.model_validate_json(invalid["response"])


def test_report_writer_preserves_every_revision_and_hash(tmp_path):
    import hashlib
    from datetime import datetime

    from goes_natural_science_kg.agents.runner import write_report

    report = OrchestrationReport.model_validate_json((GOLDEN / "report.json").read_bytes())
    write_report(tmp_path, report, datetime(2026, 9, 14, tzinfo=UTC), 0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (
        manifest["artifacts"]["report.json"]["sha256"]
        == hashlib.sha256((tmp_path / "report.json").read_bytes()).hexdigest()
    )
    assert (tmp_path / "report.json").read_bytes() == (GOLDEN / "report.json").read_bytes()
