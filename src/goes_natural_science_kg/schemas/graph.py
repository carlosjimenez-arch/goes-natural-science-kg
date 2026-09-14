# src/goes_natural_science_kg/schemas/graph.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Freeze graph snapshots and semantic diff output contracts.
"""Structural integrity only; no prerequisite inference or curriculum solver."""

from __future__ import annotations

from typing import Literal, Self, TypeVar

from pydantic import model_validator

from goes_natural_science_kg.schemas.base import Contract, Digest, References, StableId, Text
from goes_natural_science_kg.schemas.ingestion import ChunkId, EvidenceReadyMicroSkill
from goes_natural_science_kg.schemas.skills import Edge, EdgeType, MicroSkill, Skill


class GraphSnapshot(Contract):
    schema_version: Literal["graph-snapshot/1.0"] = "graph-snapshot/1.0"
    version: Text
    skills: tuple[Skill, ...] = ()
    micro_skills: tuple[MicroSkill, ...] = ()
    edges: tuple[Edge, ...] = ()
    source_ids: References = ()

    @model_validator(mode="after")
    def integrity(self) -> Self:
        for field in ("skills", "micro_skills", "edges"):
            values = getattr(self, field)
            if len({value.id for value in values}) != len(values):
                raise ValueError(f"duplicate {field} IDs")
            object.__setattr__(self, field, tuple(sorted(values, key=lambda v: v.id)))
        nodes = {n.id for n in (*self.skills, *self.micro_skills)}
        skills = {n.id for n in self.skills}
        if any(m.parent_skill_id not in skills for m in self.micro_skills):
            raise ValueError("dangling parent_skill_id")
        if any(e.source not in nodes or e.target not in nodes for e in self.edges):
            raise ValueError("dangling edge endpoint")
        if any(not r.startswith("source-") for r in self.source_ids):
            raise ValueError("source_ids must be corpus document IDs")
        refs = {r for micro in self.micro_skills for r in micro.source_refs}
        refs.update(r for edge in self.edges for r in edge.source_refs)
        if not refs.issubset(self.source_ids):
            raise ValueError("unregistered corpus source reference")
        self._check_inline_prerequisites()
        return self

    def _check_inline_prerequisites(self) -> None:
        expected: dict[str, set[tuple[EdgeType, str]]] = {m.id: set() for m in self.micro_skills}
        for edge in self.edges:
            if (
                edge.type in {EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE}
                and edge.target in expected
            ):
                expected[edge.target].add((edge.type, edge.source))
            if edge.type == EdgeType.CO_REQUISITE and edge.source in expected:
                expected[edge.source].add((edge.type, edge.target))
        for micro in self.micro_skills:
            actual = {(p.type, p.id) for p in micro.prerequisites}
            if actual != expected[micro.id]:
                raise ValueError("inline prerequisites must exactly mirror snapshot edges")


T = TypeVar("T", Skill, MicroSkill, Edge, Skill | MicroSkill)


class Modification[T: (Skill, MicroSkill, Edge, Skill | MicroSkill)](Contract):
    id: StableId
    changed_fields: tuple[Text, ...]
    before: T
    after: T


class EntityDelta[T: (Skill, MicroSkill, Edge, Skill | MicroSkill)](Contract):
    added: tuple[T, ...] = ()
    removed: tuple[T, ...] = ()
    modified: tuple[Modification[T], ...] = ()


class GraphDiff(Contract):
    schema_version: Literal["graph-diff/1.0"] = "graph-diff/1.0"
    before_version: Text
    after_version: Text
    before_sha256: Digest
    after_sha256: Digest
    nodes: EntityDelta[Skill | MicroSkill]
    edges: EntityDelta[Edge]
    added_source_ids: References = ()
    removed_source_ids: References = ()


class EvidenceGraphSnapshot(GraphSnapshot):
    """Version 2 keeps Phase 3 chunk references through scheduling and interchange."""

    schema_version: Literal["graph-snapshot/2.0"] = "graph-snapshot/2.0"  # type: ignore[assignment]
    micro_skills: tuple[EvidenceReadyMicroSkill, ...] = ()
    chunk_sources: dict[ChunkId, StableId]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        for field in ("skills", "micro_skills", "edges"):
            values = getattr(self, field)
            if len({value.id for value in values}) != len(values):
                raise ValueError("duplicate graph entity")
            object.__setattr__(self, field, tuple(sorted(values, key=lambda value: value.id)))
        skills = {s.id for s in self.skills}
        nodes = skills | {m.id for m in self.micro_skills}
        if any(m.parent_skill_id not in skills for m in self.micro_skills):
            raise ValueError("dangling parent skill")
        if any(e.source not in nodes or e.target not in nodes for e in self.edges):
            raise ValueError("dangling edge endpoint")
        if any(not key.startswith("source-") for key in self.source_ids):
            raise ValueError("source_ids must reference documents")
        if not set(self.chunk_sources.values()) <= set(self.source_ids):
            raise ValueError("chunk references unknown source document")
        if any(ref not in self.chunk_sources for m in self.micro_skills for ref in m.source_refs):
            raise ValueError("unregistered chunk reference")
        if any(ref not in self.source_ids for e in self.edges for ref in e.source_refs):
            raise ValueError("unregistered edge source document")
        self._check_inline_prerequisites()
        return self
