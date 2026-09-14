# src/goes_natural_science_kg/schemas/__init__.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Define the schemas namespace.

from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.curriculum import ContextTag, CurriculumUnit, GradeCurriculum
from goes_natural_science_kg.schemas.graph import GraphDiff, GraphSnapshot
from goes_natural_science_kg.schemas.skills import Edge, MicroSkill, Skill

__all__ = [
    "ContextTag",
    "CurriculumUnit",
    "Edge",
    "GradeCurriculum",
    "GraphDiff",
    "GraphSnapshot",
    "MicroSkill",
    "Skill",
    "SourceDocument",
]
