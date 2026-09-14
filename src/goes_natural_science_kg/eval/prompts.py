# src/goes_natural_science_kg/eval/prompts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Validate prompt front-matter and exact template variables offline.
"""Only format validation; no semantic judge or model call in Phase 2."""

from pathlib import Path
from string import Formatter

import yaml
from pydantic import TypeAdapter

from goes_natural_science_kg.schemas.prompt_evaluation import VersionedPrompt
from goes_natural_science_kg.schemas.prompts import AgentPromptMetadata, PromptMetadata


def read_prompt(path: Path) -> tuple[PromptMetadata | AgentPromptMetadata | VersionedPrompt, str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    if len(parts) != 3 or parts[0]:
        raise ValueError("prompt requires YAML front-matter delimited by ---")
    raw = yaml.safe_load(parts[1])
    if raw.get("schema_version") == "versioned-prompt/1.0":
        import re

        modern = VersionedPrompt.model_validate(raw)
        body = parts[2].strip()
        if set(re.findall(r"\{\{([a-z][a-z0-9_]*)\}\}", body)) != set(modern.variables):
            raise ValueError("prompt variables do not match front-matter")
        return modern, body
    metadata: PromptMetadata | AgentPromptMetadata = TypeAdapter(
        PromptMetadata | AgentPromptMetadata
    ).validate_python(yaml.safe_load(parts[1]))
    body = parts[2].strip()
    if not body:
        raise ValueError("prompt body is empty")
    variables = {field for _, field, _, _ in Formatter().parse(body) if field is not None}
    if variables != set(metadata.input_variables):
        raise ValueError("prompt variables do not match front-matter")
    return metadata, body
