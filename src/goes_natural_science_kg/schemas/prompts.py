# src/goes_natural_science_kg/schemas/prompts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Freeze versioned Spanish prompt metadata and evaluation output.
"""Prompt bodies are data; no LLM invocation is implemented."""

from typing import Annotated, Literal

from pydantic import Field

from goes_natural_science_kg.schemas.base import Contract, Slug, Text


class PromptMetadata(Contract):
    schema_version: Literal["prompt/1.0"] = "prompt/1.0"
    id: Slug
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    language: Literal["es-SV"]
    status: Literal["draft", "reviewed"]
    task: Text
    output_contract: Literal["evidence-review/1.0"]
    input_variables: tuple[Slug, ...]


class EvidenceReview(Contract):
    schema_version: Literal["evidence-review/1.0"] = "evidence-review/1.0"
    supported: bool
    rationale: Text
    missing_evidence: tuple[Text, ...] = ()


class AgentPromptMetadata(Contract):
    schema_version: Literal["agent-prompt/1.0"] = "agent-prompt/1.0"
    id: Slug
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    language: Literal["es-SV"]
    status: Literal["draft", "reviewed"]
    task: Text
    output_contract: Literal[
        "root-plan/1.0",
        "domain-plan/1.0",
        "decomposition/1.0",
        "curriculum-proposal/1.0",
        "judge-verdict/1.0",
    ]
    input_variables: tuple[Slug, ...]
