# src/goes_natural_science_kg/graph/diff.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Compare versioned graphs by stable identity and canonical semantic content.
"""O(n) indexing plus sorted output, without pairwise graph comparisons."""

from __future__ import annotations

from typing import TypeVar

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.graph import (
    EntityDelta,
    GraphDiff,
    GraphSnapshot,
    Modification,
)
from goes_natural_science_kg.schemas.skills import Edge, MicroSkill, Skill

T = TypeVar("T", Skill, MicroSkill, Edge, Skill | MicroSkill)


def _delta[T: (Skill, MicroSkill, Edge, Skill | MicroSkill)](
    before: tuple[T, ...], after: tuple[T, ...]
) -> EntityDelta[T]:
    old = {entity.id: entity for entity in before}
    new = {entity.id: entity for entity in after}
    modified = []
    for key in sorted(old.keys() & new.keys()):
        left, right = old[key].model_dump(mode="json"), new[key].model_dump(mode="json")
        changed = tuple(
            sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
        )
        if changed:
            modified.append(
                Modification[T](id=key, changed_fields=changed, before=old[key], after=new[key])
            )
    return EntityDelta[T](
        added=tuple(new[k] for k in sorted(new.keys() - old.keys())),
        removed=tuple(old[k] for k in sorted(old.keys() - new.keys())),
        modified=tuple(modified),
    )


def diff_graphs(before: GraphSnapshot, after: GraphSnapshot) -> GraphDiff:
    """Classify edits without ID churn; array ordering in snapshots is immaterial."""
    return GraphDiff(
        before_version=before.version,
        after_version=after.version,
        before_sha256=content_hash(before),
        after_sha256=content_hash(after),
        nodes=_delta((*before.skills, *before.micro_skills), (*after.skills, *after.micro_skills)),
        edges=_delta(before.edges, after.edges),
        added_source_ids=tuple(sorted(set(after.source_ids) - set(before.source_ids))),
        removed_source_ids=tuple(sorted(set(before.source_ids) - set(after.source_ids))),
    )


def summarize_diff(diff: GraphDiff) -> str:
    """Render a compact review, keeping the full typed JSON available separately."""
    lines = [f"Graph {diff.before_version} -> {diff.after_version}"]
    for name, group in (("nodes", diff.nodes), ("edges", diff.edges)):
        lines.append(f"{name}: +{len(group.added)} -{len(group.removed)} ~{len(group.modified)}")
        lines.extend(f"  + {item.id}" for item in group.added)
        lines.extend(f"  - {item.id}" for item in group.removed)
        lines.extend(f"  ~ {item.id}: {', '.join(item.changed_fields)}" for item in group.modified)
    lines.append(f"sources: +{len(diff.added_source_ids)} -{len(diff.removed_source_ids)}")
    return "\n".join(lines) + "\n"
