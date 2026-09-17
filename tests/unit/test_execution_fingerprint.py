# tests/unit/test_execution_fingerprint.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify the execution fingerprint covers what can affect a run and nothing else.
from pathlib import Path

from goes_natural_science_kg.agents.runner import (
    import_closure,
    imported_modules,
    module_file,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/goes_natural_science_kg"
ENTRY = PACKAGE / "agents/runner.py"


def relative(paths: tuple[Path, ...]) -> set[str]:
    return {str(p.relative_to(PACKAGE)) for p in paths}


def test_closure_covers_every_module_orchestration_can_reach():
    covered = relative(import_closure(ENTRY, PACKAGE))
    # The agent graph, its transport, its checks and every contract they validate.
    assert {
        "agents/runner.py",
        "agents/hierarchy.py",
        "agents/checks.py",
        "agents/transport.py",
        "schemas/orchestration.py",
        "schemas/agent_state.py",
        "schemas/base.py",
        "schemas/skills.py",
        "schemas/ingestion.py",
        "eval/prompts.py",
        "eval/registry.py",
        "eval/calibration.py",
        "corpus/index.py",
        "corpus/trace.py",
    } <= covered


def test_closure_excludes_modules_that_cannot_affect_a_run():
    covered = relative(import_closure(ENTRY, PACKAGE))
    # Evaluation studies, the solver and the CLI never run inside an orchestration.
    assert covered.isdisjoint(
        {
            "cli.py",
            "config.py",
            "eval/rescore.py",
            "eval/reviewer_agreement.py",
            "eval/harness.py",
            "eval/experiment.py",
            "eval/metrics.py",
            "curriculum/solver.py",
            "curriculum/artifacts.py",
            "graph/diff.py",
            "schemas/rescoring.py",
            "schemas/sequencing.py",
        }
    )
    assert len(covered) < len(list(PACKAGE.rglob("*.py")))


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_closure_follows_transitive_and_function_local_imports(tmp_path):
    package = tmp_path / "goes_natural_science_kg"
    write(package / "__init__.py", "")
    write(
        package / "agents/entry.py",
        "from goes_natural_science_kg.agents.middle import step\n"
        "def later():\n"
        "    from goes_natural_science_kg.deferred import helper\n"
        "    return helper(step)\n",
    )
    write(package / "agents/middle.py", "import goes_natural_science_kg.leaf\nstep = 1\n")
    write(package / "leaf.py", "value = 1\n")
    write(package / "deferred.py", "def helper(x):\n    return x\n")
    write(package / "unrelated.py", "import json\n")
    covered = {
        str(p.relative_to(package)) for p in import_closure(package / "agents/entry.py", package)
    }
    assert covered == {"agents/entry.py", "agents/middle.py", "leaf.py", "deferred.py"}
    assert "unrelated.py" not in covered


def test_module_resolution_handles_packages_and_foreign_names():
    assert (
        module_file(PACKAGE, "goes_natural_science_kg.schemas.base") == PACKAGE / "schemas/base.py"
    )
    assert (
        module_file(PACKAGE, "goes_natural_science_kg.schemas") == PACKAGE / "schemas/__init__.py"
    )
    assert module_file(PACKAGE, "goes_natural_science_kg.does_not_exist") is None
    names = imported_modules(ENTRY, "goes_natural_science_kg")
    assert "goes_natural_science_kg.agents.hierarchy" in names
    assert not any(name.startswith(("google", "langgraph", "asyncio")) for name in names)
