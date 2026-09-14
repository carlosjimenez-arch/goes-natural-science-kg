# src/goes_natural_science_kg/schemas/identity.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Mint stable content-derived identities without coupling them to grade or label.
"""The immutable identity payload is distinct from editable revision content."""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import model_validator

from goes_natural_science_kg.schemas.base import Contract, Slug, StableId, content_hash


class Identity(Contract):
    """Persistent editorial namespace/key; changing it denotes a different entity."""

    namespace: Slug
    key: Slug


def stable_id(kind: str, identity: Identity) -> str:
    """Use readable key plus SHA-256 of the canonical, immutable identity content."""
    if kind not in {"skill", "micro", "edge", "source", "unit", "grade"}:
        raise ValueError("unsupported entity kind")
    payload = {"identity_version": 1, "kind": kind, **identity.model_dump(mode="json")}
    return f"{kind}-{identity.key}-{content_hash(payload)[:16]}"


class Entity(Contract):
    """Require IDs to match identity; callers use stable_id when creating records."""

    identity_kind: ClassVar[str]
    id: StableId
    identity: Identity

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if self.id != stable_id(self.identity_kind, self.identity):
            raise ValueError("id does not match canonical identity")
        return self
