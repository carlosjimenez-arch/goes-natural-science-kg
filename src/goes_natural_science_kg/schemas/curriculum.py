# src/goes_natural_science_kg/schemas/curriculum.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Keep contextualization and auditable minute totals outside the graph.
"""Contracts validate arithmetic only; they do not solve or allocate instruction."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from goes_natural_science_kg.schemas.base import Contract, Grade, Minutes, References, Text
from goes_natural_science_kg.schemas.identity import Entity

BUDGET_MINUTES_PER_GRADE = 9600


class ContextType(StrEnum):
    EVERYDAY = "everyday"
    DISCIPLINARY = "disciplinary"
    INTERDISCIPLINARY = "interdisciplinary"


class ContextDomain(StrEnum):
    POLITICAL = "political"
    CULTURAL = "cultural"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"


class ContextTag(Contract):
    schema_version: Literal["context-tag/1.0"] = "context-tag/1.0"
    type: ContextType
    domain: ContextDomain


class CurriculumUnit(Entity):
    schema_version: Literal["curriculum-unit/1.0"] = "curriculum-unit/1.0"
    identity_kind: ClassVar[str] = "unit"
    label: Text
    micro_skill_ids: Annotated[References, Field(min_length=1)]
    contexts: tuple[ContextTag, ...] = ()
    instruction_minutes: Minutes
    assessment_minutes: Minutes
    review_minutes: Minutes
    setup_minutes: Minutes
    total_minutes: Annotated[Minutes, Field(gt=0)]

    @model_validator(mode="after")
    def consistent_total(self) -> Self:
        if any(not r.startswith("micro-") for r in self.micro_skill_ids):
            raise ValueError("units must reference micro-skills")
        parts = self.instruction_minutes + self.assessment_minutes
        parts += self.review_minutes + self.setup_minutes
        if self.total_minutes != parts:
            raise ValueError("unit total_minutes must equal all minute components")
        keys = [(c.type.value, c.domain.value) for c in self.contexts]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate context tags")
        object.__setattr__(
            self,
            "contexts",
            tuple(sorted(self.contexts, key=lambda c: (c.type.value, c.domain.value))),
        )
        return self


class GradeCurriculum(Entity):
    schema_version: Literal["grade-curriculum/1.0"] = "grade-curriculum/1.0"
    identity_kind: ClassVar[str] = "grade"
    grade: Grade
    units: tuple[CurriculumUnit, ...]
    total_minutes: Minutes
    budget_minutes: Annotated[Minutes, Field(le=BUDGET_MINUTES_PER_GRADE)] = 9600

    @model_validator(mode="after")
    def consistent_total(self) -> Self:
        if len({u.id for u in self.units}) != len(self.units):
            raise ValueError("duplicate units would double-count minutes")
        if self.total_minutes != sum(u.total_minutes for u in self.units):
            raise ValueError("grade total_minutes must equal unit totals")
        if self.total_minutes > self.budget_minutes:
            raise ValueError("grade minute budget exceeded")
        return self
