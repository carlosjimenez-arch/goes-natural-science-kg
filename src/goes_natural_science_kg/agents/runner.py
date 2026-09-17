# src/goes_natural_science_kg/agents/runner.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import ast
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from google import genai
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from goes_natural_science_kg.agents.hierarchy import build_hierarchy
from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.corpus.index import load_index
from goes_natural_science_kg.corpus.manifest import read_manifest
from goes_natural_science_kg.corpus.trace import trace_references
from goes_natural_science_kg.eval.calibration import confidence_report
from goes_natural_science_kg.eval.prompts import read_prompt
from goes_natural_science_kg.schemas.artifacts import ArtifactDigest, ArtifactManifest
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.calibration import CalibrationRecord
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    EvidenceAnchor,
    EvidencePacket,
    ItemOutcome,
    OrchestrationInput,
    OrchestrationReport,
    OrchestrationSettings,
    WorkItem,
)


def load_evidence(root: Path, request: OrchestrationInput) -> tuple[EvidencePacket, ...]:
    refs = tuple(sorted({r for s in request.skills for r in s.source_refs}))
    chunks = {c.id: c for c in load_index(root / "interim/index")[0]}
    docs = {d.id: d for d in read_manifest(root / "manifests/corpus.jsonl")}
    return tuple(
        EvidencePacket(
            chunk_id=str(record["chunk_id"]),
            source_url=str(record["source_url"]),
            source_type=docs[chunks[str(record["chunk_id"])].source_id].type.value,
            anchors=tuple(
                EvidenceAnchor(page=a["page"], paragraph_id=a["paragraph"], text=a["text"])
                for a in cast(tuple[dict[str, Any], ...], record["anchors"])
            ),
        )
        for record in trace_references(root, refs)
    )


def imported_modules(path: Path, package: str) -> set[str]:
    """Dotted names of this package that the module imports, including inside functions."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(node.module + "." + alias.name for alias in node.names)
    return {name for name in names if name == package or name.startswith(package + ".")}


def module_file(package_root: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")[1:]
    if not parts:
        candidate = package_root / "__init__.py"
        return candidate if candidate.is_file() else None
    module = package_root.joinpath(*parts).with_suffix(".py")
    if module.is_file():
        return module
    package = package_root.joinpath(*parts) / "__init__.py"
    return package if package.is_file() else None


def import_closure(entry: Path, package_root: Path) -> tuple[Path, ...]:
    """Modules this entry point can reach through static imports, transitively.

    Hashing every file in the package made an edit to an unrelated module change
    execution_sha256 and invalidate recorded runs. The closure keeps the guarantee that
    any module able to affect the run is covered, without the false positives.
    """
    package = package_root.name
    seen: set[Path] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for dotted in imported_modules(path, package):
            target = module_file(package_root, dotted)
            if target is not None and target not in seen:
                queue.append(target)
    return tuple(sorted(seen))


def execution_id(runtime: AgentRuntime) -> str:
    prompts = {
        p.name: content_hash(p.read_text()) for p in sorted(runtime.prompts.glob("*.prompt"))
    }
    package_root = Path(__file__).parents[1]
    return content_hash(
        {
            "input": runtime.request.model_dump(mode="json"),
            "settings": runtime.settings.model_dump(mode="json"),
            "evidence": [p.model_dump(mode="json") for p in runtime.evidence],
            "prompts": prompts,
            "workflow": "4.2",
            "implementation_fingerprint": "import-closure/1.0",
            "implementation": {
                str(path.relative_to(package_root)): content_hash(path.read_text())
                for path in import_closure(Path(__file__).resolve(), package_root)
            },
        }
    )


async def execute(
    runtime: AgentRuntime,
    checkpoints: Path,
    *,
    resume: bool = False,
    interrupt_before: list[str] | None = None,
) -> dict[str, Any]:
    checkpoints.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoints)) as saver:
        graph = build_hierarchy(runtime, saver, interrupt_before=interrupt_before)
        config: RunnableConfig = {
            "configurable": {"thread_id": execution_id(runtime)},
            "recursion_limit": 10000,
            "max_concurrency": runtime.settings.max_concurrency,
        }
        snapshot = await graph.aget_state(config)
        if resume and not snapshot.values:
            raise ValueError("no matching checkpoint for input/settings/evidence/prompts")
        if not resume and snapshot.values:
            raise ValueError("checkpoint exists; use resume or a different checkpoint file")
        result = await graph.ainvoke(
            None
            if resume
            else {
                "task": WorkItem(
                    id="L0/root",
                    level="L0",
                    instructions="Plan the input skill map by grade within each explicit minute budget.",
                )
            },
            config,
        )
        return dict(result)


def make_report(runtime: AgentRuntime, state: dict[str, Any]) -> OrchestrationReport:
    items = tuple(ItemOutcome.model_validate(v) for _, v in sorted(state["outcomes"].items()))
    curricula = tuple(
        i.curriculum
        for i in items
        if i.level == "L3" and i.status == "approved" and i.curriculum is not None
    )
    return OrchestrationReport(
        input_sha256=content_hash(runtime.request),
        execution_sha256=execution_id(runtime),
        prompt_versions={
            read_prompt(p)[0].id: read_prompt(p)[0].version
            for p in sorted(runtime.prompts.glob("*.prompt"))
        },
        settings_sha256=content_hash(runtime.settings),
        items=items,
        approved_curricula=curricula,
        complete={c.grade for c in curricula} == set(runtime.request.budget_minutes),
    )


def write_report(
    output: Path,
    report: OrchestrationReport,
    generated_at: datetime,
    seed: int,
    calibration: CalibrationRecord | None = None,
) -> None:
    """Write the report and its panel-derived confidence estimates under one manifest."""
    import hashlib

    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "report.json": (canonical_json(report) + "\n").encode(),
        "confidence.json": (canonical_json(confidence_report(report, calibration)) + "\n").encode(),
    }
    for name, payload in payloads.items():
        atomic_bytes(output / name, payload)
    inputs = {"orchestration-input": report.input_sha256}
    if calibration is not None:
        inputs["confidence-calibration"] = calibration.id
    manifest = ArtifactManifest(
        generated_at=generated_at,
        git_sha=None,
        git_dirty=None,
        seed=seed,
        settings_hash=report.settings_sha256,
        inputs=inputs,
        artifacts={
            name: ArtifactDigest(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
            for name, payload in payloads.items()
        },
        notes="Every item records initial ballot plus at most three revisions. Provider cache is required for deterministic replay. Confidence is recomputed from ballots; self-reported values are never published as the estimate.",
    )
    atomic_bytes(output / "manifest.json", (canonical_json(manifest) + "\n").encode())


async def run_orchestration(
    request: OrchestrationInput,
    settings: OrchestrationSettings,
    data: Path,
    prompts: Path,
    cache: Path,
    checkpoints: Path,
    output: Path,
    generated_at: datetime,
    *,
    online: bool = False,
    resume: bool = False,
    calibration: CalibrationRecord | None = None,
) -> OrchestrationReport:
    client = (
        genai.Client(
            vertexai=True,
            project=settings.project,
            location=settings.location,
            http_options={"timeout": 120000, "retry_options": {"attempts": 3}},
        )
        if online
        else None
    )
    runtime = AgentRuntime(
        settings=settings,
        request=request,
        evidence=load_evidence(data, request),
        prompts=prompts,
        cache=cache,
        client=client,
        limiter=asyncio.Semaphore(settings.max_concurrency),
    )
    try:
        report = make_report(runtime, await execute(runtime, checkpoints, resume=resume))
        write_report(output, report, generated_at, settings.seed, calibration)
        return report
    finally:
        if client is not None:
            await client.aio.aclose()
