# src/goes_natural_science_kg/schemas/decisions.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Validate both research proposals and implemented architecture decisions.
"""Evidence extensions retain the Phase 1 decision without silently accepting it."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Text
from goes_natural_science_kg.schemas.corpus import SourceUrl


class DecisionOption(Contract):
    name: Text
    pros: tuple[Text, ...]
    cons: tuple[Text, ...]


class DecisionEvidence(Contract):
    findings: Text
    claim_ids: tuple[Text, ...]
    source_url: tuple[SourceUrl, ...]
    retrieved_at: date


class ProfileTradeoffs(Contract):
    curriculum_designer: Text
    graph_theory_researcher: Text
    ai_software_engineer: Text


class ContractChange(Contract):
    location: Text
    change: Text


class ArchitectureDecision(Contract):
    # The existing research decision predates a schema_version field.
    schema_version: Literal["architecture-decision/1.0"] = "architecture-decision/1.0"
    id: Annotated[str, Field(pattern=r"^\d{4}$")]
    title: Text
    status: Literal["proposed", "accepted", "superseded", "pending"]
    date: date
    supersedes: Annotated[str, Field(pattern=r"^\d{4}$")] | None
    invariants: tuple[Annotated[int, Field(ge=1, le=8)], ...]
    context: Text
    options: Annotated[tuple[DecisionOption, ...], Field(min_length=1)]
    criteria: Annotated[tuple[Text, ...], Field(min_length=1)]
    decision: Text
    consequences: tuple[Text, ...]
    enforced_by: tuple[Text, ...]
    evidence: DecisionEvidence | None = None
    profile_tradeoffs: ProfileTradeoffs | None = None
    contract_changes_proposed_not_applied: tuple[ContractChange, ...] = ()
    future_acceptance_checks: tuple[Text, ...] = ()
    implementation_status: Text | None = None

    @model_validator(mode="after")
    def enforcement_required(self) -> Self:
        if self.status != "proposed" and not self.enforced_by:
            raise ValueError("non-proposed decisions require enforcement tests")
        return self
