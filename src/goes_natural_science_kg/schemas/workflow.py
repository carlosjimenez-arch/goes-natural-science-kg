# src/goes_natural_science_kg/schemas/workflow.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Centralize the typed license workflow state.
"""No agent state contract is declared outside schemas/."""

from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, model_validator

from goes_natural_science_kg.schemas.base import Contract, References
from goes_natural_science_kg.schemas.corpus import SourceAllowlist, SourceDocument


class LicenseGateState(Contract):
    schema_version: Literal["license-gate-state/1.0"] = "license-gate-state/1.0"
    documents: tuple[SourceDocument, ...]
    allowlist: SourceAllowlist
    checked_at: AwareDatetime
    manifest_path: Path
    results: tuple[SourceDocument, ...] = ()
    ingestion_ready_ids: References = ()

    @model_validator(mode="after")
    def unique_input(self) -> Self:
        if len({doc.id for doc in self.documents}) != len(self.documents):
            raise ValueError("duplicate source IDs in one batch")
        return self
