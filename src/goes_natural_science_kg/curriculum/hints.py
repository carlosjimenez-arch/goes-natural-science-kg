# src/goes_natural_science_kg/curriculum/hints.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Seed large fixed-grade CP-SAT searches with finite-bank feasible placements.
from __future__ import annotations

from graphlib import TopologicalSorter

from ortools.sat.python import cp_model

from goes_natural_science_kg.schemas.release import ScheduleHint
from goes_natural_science_kg.schemas.sequencing import (
    LearningActivity,
    SequencingInput,
    SequencingSettings,
)
from goes_natural_science_kg.schemas.skills import EdgeType


def fill(
    activities: tuple[LearningActivity, ...], minutes: int, seed: int
) -> tuple[LearningActivity, ...] | None:
    """Solve a small bounded subset-sum; never invent or stretch an activity."""
    if minutes == 0:
        return ()
    model = cp_model.CpModel()
    chosen = [model.new_bool_var(a.id) for a in activities]
    model.add(sum(a.minutes * z for a, z in zip(activities, chosen, strict=True)) == minutes)
    model.maximize(
        sum(
            a.minutes
            * (
                3
                if a.cognitive_domain.value == "reasoning"
                else 2
                if a.cognitive_domain.value == "applying"
                else 1
            )
            * z
            for a, z in zip(activities, chosen, strict=True)
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_deterministic_time = 0.2
    if solver.solve(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return tuple(a for a, z in zip(activities, chosen, strict=True) if solver.value(z))


def schedule_hint(data: SequencingInput, settings: SequencingSettings) -> tuple[ScheduleHint, ...]:  # noqa: C901 - guarded constructive schedule, no acceptance bypass
    """Hint only the fixed-grade, atomic-teach case; the full solver and audit remain authoritative."""
    if any(len(a.allowed_grades) != 1 for a in data.activities) or any(
        e.type == EdgeType.CO_REQUISITE for e in data.graph.edges
    ):
        return ()
    teaches = [a for a in data.activities if a.kind == "teach"]
    if any(len(a.micro_skill_ids) != 1 for a in teaches):
        return ()
    by_micro = {a.micro_skill_ids[0]: a for a in teaches}
    if len(by_micro) != len(teaches):
        return ()
    predecessors: dict[str, set[str]] = {m.id: set() for m in data.graph.micro_skills}
    for edge in data.graph.edges:
        if edge.type == EdgeType.PREREQUISITE:
            predecessors[edge.target].add(edge.source)
    order = tuple(
        TopologicalSorter(
            {k: tuple(sorted(v)) for k, v in sorted(predecessors.items())}
        ).static_order()
    )
    by_id = {a.id: a for a in data.activities}
    hints: list[ScheduleHint] = []
    used: set[str] = set()
    taught: set[str] = set()
    budget = settings.budget_hours * 60
    for grade in range(2, 7):
        required = [
            a
            for a in data.activities
            if a.mandatory and a.kind != "teach" and a.allowed_grades == (grade,)
        ]
        required += [by_micro[k] for k in order if by_micro[k].allowed_grades == (grade,)]
        pending = list(required)
        cursor = (grade - 2) * budget
        for period in range(settings.periods):
            period_hint_start = len(hints)
            end = (grade - 2) * budget + (period + 1) * budget // settings.periods
            while pending and cursor + pending[0].minutes <= end:
                a = pending.pop(0)
                if a.kind == "teach" and not predecessors[a.micro_skill_ids[0]] <= taught:
                    return ()
                if a.kind != "teach" and not set(a.micro_skill_ids) <= taught:
                    return ()
                hints.append(
                    ScheduleHint(
                        activity_id=a.id, grade=grade, start=cursor, end=cursor + a.minutes
                    )
                )
                cursor += a.minutes
                used.add(a.id)
                if a.kind == "teach":
                    taught.update(a.micro_skill_ids)
            extra = None
            for _ in range(9):
                eligible = tuple(
                    a
                    for a in data.activities
                    if a.id not in used
                    and a.kind != "teach"
                    and a.allowed_grades == (grade,)
                    and set(a.micro_skill_ids) <= taught
                    and a not in pending
                )
                extra = fill(eligible, end - cursor, settings.seed)
                if extra is not None:
                    break
                if len(hints) <= period_hint_start:
                    return ()
                previous = by_id[hints[-1].activity_id]
                if previous.kind != "teach":
                    return ()
                hints.pop()
                pending.insert(0, previous)
                used.remove(previous.id)
                taught.difference_update(previous.micro_skill_ids)
                cursor -= previous.minutes
            if extra is None:
                return ()
            for a in extra:
                hints.append(
                    ScheduleHint(
                        activity_id=a.id, grade=grade, start=cursor, end=cursor + a.minutes
                    )
                )
                cursor += a.minutes
                used.add(a.id)
        if pending:
            return ()
    return tuple(hints)


def context_hint(
    data: SequencingInput, settings: SequencingSettings, hints: tuple[ScheduleHint, ...]
) -> set[tuple[str, str, int]]:
    """Seed one authored example per available context, subject to both caps and variety."""
    activities = {a.id: a for a in data.activities}
    result: set[tuple[str, str, int]] = set()
    for grade in range(2, 7):
        selected = tuple(h for h in hints if h.grade == grade)
        used: set[str] = set()
        minutes = 0
        grade_choices: set[tuple[str, str, int]] = set()
        for context in sorted(data.contexts, key=lambda c: c.id):
            for hint in selected:
                activity = activities[hint.activity_id]
                if activity.id in used or context.id not in activity.context_tasks:
                    continue
                if (
                    100 * (minutes + activity.minutes)
                    > settings.context_max_percent * settings.budget_hours * 60
                    or 100 * (len(used) + 1) > settings.context_max_percent * len(selected)
                    or 100 * activity.minutes
                    > settings.context_single_max_percent * settings.budget_hours * 60
                ):
                    continue
                grade_choices.add((activity.id, context.id, grade))
                used.add(activity.id)
                minutes += activity.minutes
                break
        if len(grade_choices) >= settings.context_min_variety:
            result.update(grade_choices)
    return result
