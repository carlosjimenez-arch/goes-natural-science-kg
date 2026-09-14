# src/goes_natural_science_kg/schemas/agent_state.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
from __future__ import annotations

from typing import Annotated, TypedDict

from goes_natural_science_kg.schemas.orchestration import (
    Ballot,
    Candidate,
    EvaluableMicro,
    ItemOutcome,
    Verdict,
    WorkItem,
)


def merge_keyed[T](left: dict[str, T], right: dict[str, T]) -> dict[str, T]:
    """Commutative, associative, idempotent union; reject conflicting concurrent writes."""
    for key in left.keys() & right.keys():
        if left[key] != right[key]:
            raise ValueError("conflicting concurrent result: " + key)
    return dict(sorted((left | right).items()))


class WorkInput(TypedDict):
    task: WorkItem


class WorkOutput(TypedDict):
    outcomes: Annotated[dict[str, ItemOutcome], merge_keyed]


class WorkState(WorkInput, WorkOutput, total=False):
    revision: int
    candidate: Candidate | None
    errors: tuple[str, ...]
    micros: tuple[EvaluableMicro, ...]
    expected_children: tuple[str, ...]
    verdicts: Annotated[dict[str, Verdict], merge_keyed]
    ballots: Annotated[dict[str, Ballot], merge_keyed]
    accepted: bool


class JudgeState(TypedDict):
    state: WorkState
    role: str
