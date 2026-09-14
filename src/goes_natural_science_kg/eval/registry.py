# src/goes_natural_science_kg/eval/registry.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
import re
from functools import lru_cache
from pathlib import Path

import yaml

from goes_natural_science_kg.eval.prompts import read_prompt
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.prompt_evaluation import (
    ChangeLogEntry,
    PromptArtifact,
    PromptRegistry,
    VersionedPrompt,
)


def load_registry(directory: Path, *, recursive: bool = True) -> PromptRegistry:
    paths = directory.rglob("*.prompt") if recursive else directory.glob("*.prompt")
    files = tuple((str(p), p.read_text()) for p in sorted(paths))
    return registry_from_files(files)


@lru_cache(maxsize=8)
def registry_from_files(files: tuple[tuple[str, str], ...]) -> PromptRegistry:
    """Cache parsing by full file content, so edits cannot reuse a stale artifact."""
    artifacts = []
    for name, text in files:
        path = Path(name)
        parts = text.split("---\n", 2)
        if len(parts) != 3 or parts[0]:
            raise ValueError("missing prompt front-matter")
        raw = yaml.safe_load(parts[1])
        legacy = raw.get("schema_version") != "versioned-prompt/1.0"
        if legacy:
            old, body = read_prompt(path)
            if isinstance(old, VersionedPrompt):
                raise ValueError("inconsistent legacy prompt metadata")
            metadata = VersionedPrompt(
                id=old.id,
                version=old.version,
                role=old.task,
                technique="structured",
                variables=old.input_variables,
                changelog=(
                    ChangeLogEntry(version=old.version, change="Frozen historical prompt import"),
                ),
                output_contract=old.output_contract,
            )
        else:
            metadata = VersionedPrompt.model_validate(raw)
            body = parts[2].strip()
            if set(re.findall(r"\{\{([a-z][a-z0-9_]*)\}\}", body)) != set(metadata.variables):
                raise ValueError("undeclared or missing template variables")
        artifacts.append(
            PromptArtifact(
                metadata=metadata,
                body=body,
                path=str(path),
                legacy=legacy,
                sha256=content_hash({"metadata": metadata.model_dump(mode="json"), "body": body}),
            )
        )
    return PromptRegistry(artifacts=tuple(artifacts))
