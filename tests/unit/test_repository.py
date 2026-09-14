# tests/unit/test_repository.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Enforce frozen exports, decision test links, source policy and repository hygiene.
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import yaml

from goes_natural_science_kg.config import Settings
from goes_natural_science_kg.eval.prompts import read_prompt
from goes_natural_science_kg.schemas.decisions import ArchitectureDecision
from goes_natural_science_kg.schemas.registry import schema_exports

ROOT = Path(__file__).resolve().parents[2]


def test_schema_exports_are_frozen():
    directory = ROOT / "data/processed/contracts"
    exports = schema_exports()
    assert {p.name for p in directory.glob("*.json")} - {"manifest.json"} == set(exports)
    for name, expected in exports.items():
        assert (directory / name).read_text() == expected, (
            f"{name}: contract changed; review its version before refreshing exports"
        )


def test_decisions_validate_and_enforcement_exists():
    for path in sorted((ROOT / "decisions").glob("*.yaml")):
        decision = ArchitectureDecision.model_validate(yaml.safe_load(path.read_text()))
        assert path.name.startswith(decision.id + "-")
        for reference in decision.enforced_by:
            file, function = reference.split("::")
            module = ast.parse((ROOT / file).read_text())
            assert any(
                isinstance(node, ast.FunctionDef) and node.name == function
                for node in ast.walk(module)
            ), reference


def test_every_schema_and_golden_digest_matches():
    for directory in [ROOT / "tests/golden", ROOT / "data/processed/contracts"]:
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, record in manifest["artifacts"].items():
            content = (directory / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == record["sha256"]
            assert len(content) == record["bytes"]


def test_gitignore_actual_behavior(tmp_path):
    # The real workspace need not be a git repository to test git's ignore semantics.
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ignored = [
        ".env",
        ".env.backup",
        "private.key",
        "service-account-test.json",
        "data/raw/book.pdf",
        "data/interim/chunk.json",
        "reports/test.json",
        ".venv/bin/python",
        "var/output.json",
        "credentials.json",
        "application_default_credentials.json",
        "client_secret-example.json",
        "service_account-example.json",
        "session.sqlite-wal",
        "session.sqlite3-shm",
        "data/processed/curriculum/previous/build/grade_2.json",
        "data/processed/curriculum/examples/160h/previous/build/report.html",
        ".venv-backup/bin/python",
        ".gcloud/token.json",
        "state.tfstate.backup",
        "local.log",
    ]
    kept = [
        ".env.example",
        "data/processed/graph.json",
        "data/manifests/corpus.jsonl",
        "decisions/0002-contracts.yaml",
        "prompts/evidence-review.v1.prompt",
        "tests/golden/pdfs/colombia-dba.pdf",
        "uv.lock",
        "findings.yaml",
        "data/manifests/corpus.jsonl",
        "data/processed/contracts/skill.json",
        "data/processed/curriculum/grade_2.json",
        "data/processed/prompt-evaluation/observations/record.json.gz",
    ]
    for path in ignored + kept:
        result = subprocess.run(
            ["git", "-C", str(tmp_path), "check-ignore", "--no-index", "-q", path], check=False
        )
        assert (result.returncode == 0) == (path in ignored), path


def test_all_direct_dependency_versions_are_exact():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = (
        config["project"]["dependencies"]
        + config["dependency-groups"]["dev"]
        + config["build-system"]["requires"]
    )
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.-]*", r) for r in requirements)
    assert (ROOT / "uv.lock").is_file()


def test_data_contracts_are_centralized():
    for path in (ROOT / "src").rglob("*.py"):
        if "schemas" in path.parts or path.name == "config.py":
            continue
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)), path


def test_prompt_frontmatter_and_spanish_body():
    prompts = list((ROOT / "prompts").glob("*.prompt"))
    assert prompts
    for path in prompts:
        metadata, body = read_prompt(path)
        assert metadata.language == "es-SV" and "Revisá" in body
        assert "{{criterion}}" in body or "{{input}}" in body


def test_config_namespacing_and_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("GOES_NATURAL_SCIENCE_KG_GCP__PROJECT_ID", "example-project")
    monkeypatch.setenv("GOES_MATH_KG_DATA_DIR", "wrong")
    cfg = Settings(_env_file=None)
    assert cfg.gcp.project_id == "example-project" and cfg.data_dir == Path("data")
    assert cfg.session_minutes is None
    assert not any("key" in name or "token" in name for name in type(cfg).model_fields)


def test_module_headers_and_only_expected_markdown():
    for base in ["src", "tests"]:
        for path in (ROOT / base).rglob("*.py"):
            lines = path.read_text().splitlines()
            assert lines[0] == f"# {path.relative_to(ROOT)}"
            assert "# SPDX-License-Identifier: Apache-2.0" in lines[:6]
    assert {p.name for p in ROOT.glob("*.md")} == {"README.md", "CLAUDE.md"}


def test_python_and_uv_versions_match_tooling():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["requires-python"] == ">=3.13,<3.14"
    assert (ROOT / ".python-version").read_text().strip().startswith("3.13.")
    assert config["tool"]["ruff"]["target-version"] == "py313"
    assert config["tool"]["mypy"]["python_version"] == "3.13"
    version = config["tool"]["uv"]["required-version"].removeprefix("==")
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert version in (ROOT / "README.md").read_text()
    # No hosted workflow is configured; the local gate is `make ci`.
    assert not (ROOT / ".github").exists()
