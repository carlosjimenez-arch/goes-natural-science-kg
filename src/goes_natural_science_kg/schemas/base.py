# src/goes_natural_science_kg/schemas/base.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Define immutable, closed data contracts and canonical serialization.
"""Shared primitives; JSON arrays become immutable tuples inside contracts."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Annotated, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, StringConstraints


def normalize_text(value: str) -> str:
    """Normalize Unicode and boundary whitespace, preserving internal wording."""
    return unicodedata.normalize("NFC", value.strip())


Text = Annotated[str, AfterValidator(normalize_text), Field(min_length=1)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StableId = Annotated[
    str,
    StringConstraints(pattern=r"^(skill|micro|edge|source|unit|grade)-[a-z0-9-]+-[0-9a-f]{16}$"),
]
Grade = Annotated[StrictInt, Field(ge=2, le=6)]
Minutes = Annotated[StrictInt, Field(ge=0)]
PositiveMinutes = Annotated[StrictInt, Field(gt=0)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
T = TypeVar("T", bound=str)


def unique_sorted[T: str](values: tuple[T, ...]) -> tuple[T, ...]:
    """Reject duplicate references and canonicalize order-insensitive collections."""
    if len(values) != len(set(values)):
        raise ValueError("duplicate values are not permitted")
    return tuple(sorted(values))


References = Annotated[tuple[StableId, ...], AfterValidator(unique_sorted)]
Texts = Annotated[tuple[Text, ...], AfterValidator(unique_sorted)]


class Contract(BaseModel):
    """Closed, frozen contract with validated defaults and nested model revalidation."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, revalidate_instances="always"
    )


def canonical_json(value: object) -> str:
    """Serialize validated JSON-compatible values; reject non-finite numbers."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def content_hash(value: object) -> str:
    """Hash the full canonical revision, independently of entity identity."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
