# src/goes_natural_science_kg/schemas/skills.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Freeze context-free skill, micro-skill and edge contracts.
"""No ContextTag or contextualization fields belong in this module."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import (
    Contract,
    Grade,
    PositiveMinutes,
    Probability,
    References,
    StableId,
    Text,
    Texts,
)
from goes_natural_science_kg.schemas.identity import Entity


class CognitiveDomain(StrEnum):
    KNOWING = "knowing"
    APPLYING = "applying"
    REASONING = "reasoning"


class EdgeType(StrEnum):
    PREREQUISITE = "PREREQUISITE"
    CO_REQUISITE = "CO_REQUISITE"
    REFINES = "REFINES"
    TRANSFERS_TO = "TRANSFERS_TO"


class FrameworkRef(Contract):
    """Versioned origin; local codes never replace canonical identity."""

    name: Text
    version: Text
    source_code: Text | None = None


class GradeBand(Contract):
    minimum: Grade
    maximum: Grade

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.minimum > self.maximum:
            raise ValueError("grade band minimum exceeds maximum")
        return self


class Skill(Entity):
    schema_version: Literal["skill/1.0"] = "skill/1.0"
    kind: Literal["skill"] = "skill"
    identity_kind: ClassVar[str] = "skill"
    label: Text
    origin_framework: FrameworkRef
    domain: Text
    hierarchy_level: Annotated[int, Field(strict=True, ge=0)]


class PrerequisiteRef(Contract):
    """Inline view: id is a prerequisite or co-requisite of the containing micro."""

    id: StableId
    type: Literal[EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE]


class MicroSkill(Entity):
    schema_version: Literal["micro-skill/1.0"] = "micro-skill/1.0"
    kind: Literal["micro_skill"] = "micro_skill"
    identity_kind: ClassVar[str] = "micro"
    parent_skill_id: StableId
    observable_verb: Text
    knowledge_object: Text
    cognitive_domain: CognitiveDomain
    grade_band: GradeBand
    estimated_minutes: PositiveMinutes
    prerequisites: tuple[PrerequisiteRef, ...] = ()
    misconceptions: Texts = ()
    evidence_of_mastery: Text
    source_refs: Annotated[References, Field(min_length=1)]
    confidence: Probability

    @model_validator(mode="after")
    def reference_shape(self) -> Self:
        if not self.parent_skill_id.startswith("skill-"):
            raise ValueError("parent_skill_id must reference a Skill")
        keys = [(p.type.value, p.id) for p in self.prerequisites]
        if len(keys) != len(set(keys)) or any(p.id == self.id for p in self.prerequisites):
            raise ValueError("duplicate or self prerequisite")
        if any(not ref.startswith("source-") for ref in self.source_refs):
            raise ValueError("source_refs must reference corpus documents")
        object.__setattr__(
            self,
            "prerequisites",
            tuple(sorted(self.prerequisites, key=lambda p: (p.type.value, p.id))),
        )
        return self


class Edge(Entity):
    """Directed source→target; CO_REQUISITE endpoints are canonicalized as unordered."""

    schema_version: Literal["edge/1.0"] = "edge/1.0"
    kind: Literal["edge"] = "edge"
    identity_kind: ClassVar[str] = "edge"
    source: StableId
    target: StableId
    type: EdgeType
    strength: Probability
    justification: Text
    source_refs: Annotated[References, Field(min_length=1)]

    @model_validator(mode="after")
    def reference_shape(self) -> Self:
        if self.source == self.target:
            raise ValueError("self edges are not permitted")
        if any(not r.startswith(("skill-", "micro-")) for r in (self.source, self.target)):
            raise ValueError("edge endpoints must be graph nodes")
        if any(not ref.startswith("source-") for ref in self.source_refs):
            raise ValueError("source_refs must reference corpus documents")
        if self.type == EdgeType.CO_REQUISITE and self.source > self.target:
            source, target = self.target, self.source
            object.__setattr__(self, "source", source)
            object.__setattr__(self, "target", target)
        return self
