# src/goes_natural_science_kg/schemas/artifacts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Describe reproducible artifacts without inventing missing repository metadata.
"""Manifest timestamps and seeds are explicit inputs."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from goes_natural_science_kg.schemas.base import Contract, Digest, Minutes, Text


class ArtifactDigest(Contract):
    sha256: Digest
    bytes: Minutes


class ArtifactManifest(Contract):
    schema_version: Literal["artifact-manifest/1.0"] = "artifact-manifest/1.0"
    generated_at: AwareDatetime
    git_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")] | None
    git_dirty: bool | None
    seed: Annotated[int, Field(strict=True, ge=0)]
    settings_hash: Digest
    inputs: dict[Text, Digest]
    artifacts: dict[Text, ArtifactDigest]
    notes: Text | None = None
