# src/goes_natural_science_kg/curriculum/validation.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Independently audit the solver calendar against graph and authored activities.
from typing import Any

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.sequencing import (
    ConstraintModel,
    ScheduledUnit,
    SequencingInput,
    SequencingSettings,
    SolvedGradeCurriculum,
)
from goes_natural_science_kg.schemas.skills import EdgeType


def preflight(data: SequencingInput, settings: SequencingSettings) -> tuple[str, ...]:
    import numpy as np
    from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
    from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

    errors = []
    micros = {m.id: m for m in data.graph.micro_skills}
    for s in data.graph.skills:
        if not any(m.parent_skill_id == s.id for m in micros.values()):
            errors.append("Uncovered input skill: " + s.id)
        if s.domain not in settings.domain_target_percent:
            errors.append("No configured content-domain target: " + s.domain)
    for key in micros:
        if not any(a.kind == "teach" and key in a.micro_skill_ids for a in data.activities):
            errors.append("No initial teaching activity: " + key)
    ids = {key: i for i, key in enumerate(sorted(micros))}
    pairs = [
        (ids[e.source], ids[e.target]) for e in data.graph.edges if e.type == EdgeType.PREREQUISITE
    ]
    if pairs:
        rows, cols = zip(*pairs, strict=True)
        matrix = csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(len(ids), len(ids)))
        _, labels = connected_components(matrix, directed=True, connection="strong")
        if np.any(np.bincount(labels) > 1):
            errors.append("Prerequisite cycle")
    for g in range(2, 7):
        available = sum(
            a.minutes
            for a in data.activities
            if g in a.allowed_grades
            and (
                a.kind != "teach"
                or all(
                    micros[k].grade_band.minimum <= g <= micros[k].grade_band.maximum
                    for k in a.micro_skill_ids
                )
            )
        )
        if available < settings.budget_hours * 60:
            errors.append(
                f"Grade {g}: at most {available} authored minutes available; {settings.budget_hours * 60} required. Add justified activities; padding is forbidden."
            )
    return tuple(errors)


def extract(
    data: SequencingInput, settings: SequencingSettings, c: ConstraintModel, solver: Any
) -> tuple[SolvedGradeCurriculum, ...]:
    contexts = {v.id: v for v in data.contexts}
    units = []
    for a in data.activities:
        if not solver.value(c.selected[a.id]):
            continue
        _, g, t = next(key for key, z in c.placements.items() if key[0] == a.id and solver.value(z))
        context_id = next(
            (key[1] for key, z in c.contexts.items() if key[0] == a.id and solver.value(z)), None
        )
        start = solver.value(c.starts[a.id]) - (g - 2) * settings.budget_hours * 60
        units.append(
            ScheduledUnit(
                id=stable_id("unit", Identity(namespace="solver-activity", key=a.id)),
                activity_id=a.id,
                micro_skill_ids=a.micro_skill_ids,
                kind=a.kind,
                grade=g,
                period=t,
                start_minute=start,
                end_minute=start + a.minutes,
                total_minutes=a.minutes,
                cognitive_domain=a.cognitive_domain,
                inquiry=a.inquiry,
                theme=a.theme,
                task=a.context_tasks[context_id] if context_id else a.task,
                mastery_criterion=a.mastery_criterion,
                context_id=context_id,
                contexts=contexts[context_id].tags if context_id else (),
            )
        )
    return tuple(
        SolvedGradeCurriculum(
            grade=g,
            approval=data.approval,
            input_sha256=content_hash(data),
            settings_sha256=content_hash(settings),
            total_minutes=settings.budget_hours * 60,
            budget_minutes=settings.budget_hours * 60,
            units=tuple(
                sorted((u for u in units if u.grade == g), key=lambda u: (u.start_minute, u.id))
            ),
        )
        for g in range(2, 7)
    )


def audit(  # noqa: C901 - independent certificate audit enumerates distinct hard invariants
    data: SequencingInput, settings: SequencingSettings, grades: tuple[SolvedGradeCurriculum, ...]
) -> tuple[tuple[str, ...], dict[str, bool], dict[int, dict[str, Any]]]:
    from goes_natural_science_kg.curriculum.solver import domain_minutes

    errors = []
    budget = settings.budget_hours * 60
    all_units = [u for g in grades for u in g.units]
    if {g.grade for g in grades} != set(range(2, 7)) or len(grades) != 5:
        errors.append("All five distinct grades required")
    if len({u.activity_id for u in all_units}) != len(all_units):
        errors.append("An authored activity was reused")
    activities = {a.id: a for a in data.activities}
    unknown = [u.activity_id for u in all_units if u.activity_id not in activities]
    if unknown:
        return tuple([*errors, *("Unknown activity: " + key for key in unknown)]), {}, {}
    units_by_id = {u.activity_id: u for u in all_units}
    for required in data.activities:
        if required.mandatory and required.id not in units_by_id:
            errors.append("Mandatory activity omitted: " + required.id)
    teaching = {}
    for micro in data.graph.micro_skills:
        taught = [u for u in all_units if u.kind == "teach" and micro.id in u.micro_skill_ids]
        if len(taught) != 1:
            errors.append("Exactly one initial teaching required: " + micro.id)
        else:
            teaching[micro.id] = taught[0]
            if not micro.grade_band.minimum <= taught[0].grade <= micro.grade_band.maximum:
                errors.append("Grade-band violation: " + micro.id)
    for u in all_units:
        a = activities.get(u.activity_id)
        if a is None:
            errors.append("Unknown activity: " + u.activity_id)
            continue
        if (
            u.id != stable_id("unit", Identity(namespace="solver-activity", key=a.id))
            or u.grade not in a.allowed_grades
            or u.total_minutes != a.minutes
            or u.micro_skill_ids != a.micro_skill_ids
            or u.cognitive_domain != a.cognitive_domain
            or u.kind != a.kind
            or u.inquiry != a.inquiry
            or u.theme != a.theme
        ):
            errors.append("Output differs from authored activity: " + u.activity_id)
        if (
            u.task != (a.context_tasks.get(u.context_id, "") if u.context_id else a.task)
            or u.mastery_criterion != a.mastery_criterion
        ):
            errors.append("Unauthored task text: " + u.activity_id)
        if u.kind != "teach":
            for key in u.micro_skill_ids:
                first = teaching.get(key)
                if (
                    first
                    and (first.grade - 2) * budget + first.end_minute
                    > (u.grade - 2) * budget + u.start_minute
                ):
                    errors.append("Practice precedes initial teaching: " + u.activity_id)
    errors.extend(dependency_errors(data, settings, teaching, all_units))
    for thread in data.continuity:
        first, last = (
            units_by_id.get(thread.opening_activity_id),
            units_by_id.get(thread.continuation_activity_id),
        )
        if first is None or last is None or last.grade != first.grade + 1 or last.period != 0:
            errors.append("Continuity thread not resumed: " + thread.id)
    coverage = {
        s.id: any(m.parent_skill_id == s.id and m.id in teaching for m in data.graph.micro_skills)
        for s in data.graph.skills
    }
    if not all(coverage.values()):
        errors.append("Input skill coverage incomplete")
    metrics = {}
    context_map = {c.id: c for c in data.contexts}
    for grade in grades:
        if (
            grade.input_sha256 != content_hash(data)
            or grade.settings_sha256 != content_hash(settings)
            or grade.approval != data.approval
        ):
            errors.append("Grade provenance differs from build input/settings")
        errors.extend(grade_errors(grade, settings, context_map))
        domain = {k: 0 for k in settings.domain_target_percent}
        cognition = {k: 0 for k in ("knowing", "applying", "reasoning")}
        for u in grade.units:
            for key, value in domain_minutes(data, u.activity_id).items():
                domain[key] += value
            cognition[u.cognitive_domain.value] += u.total_minutes
        spacing_counts = {}
        for key in data.key_micro_ids:
            first = teaching.get(key)
            if first and first.grade == grade.grade:
                spacing_counts[key] = len(
                    {
                        u.period
                        for u in grade.units
                        if u.kind == "retrieval"
                        and key in u.micro_skill_ids
                        and u.start_minute >= first.end_minute + settings.spacing_minutes
                    }
                )
        metrics[grade.grade] = {
            "total_minutes": grade.total_minutes,
            "domain_minutes": domain,
            "cognitive_minutes": cognition,
            "inquiry_minutes": sum(u.total_minutes for u in grade.units if u.inquiry),
            "context_minutes": sum(u.total_minutes for u in grade.units if u.context_id),
            "context_ids": sorted({u.context_id for u in grade.units if u.context_id}),
            "spaced_retrieval_periods": spacing_counts,
            "reasoning_target_percent": settings.reasoning_min_percent[grade.grade],
            "knowing_max_percent": settings.knowing_max_percent,
            "theme_count_per_period": {
                t: len({u.theme for u in grade.units if u.period == t})
                for t in range(settings.periods)
            },
        }
    return tuple(errors), coverage, metrics


def dependency_errors(
    data: SequencingInput,
    settings: SequencingSettings,
    teaching: dict[str, ScheduledUnit],
    units: list[ScheduledUnit],
) -> tuple[str, ...]:
    errors = []
    budget = settings.budget_hours * 60
    for edge in data.graph.edges:
        if edge.type not in (EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE):
            continue
        first, last = teaching.get(edge.source), teaching.get(edge.target)
        if first is None or last is None:
            continue
        if edge.type == EdgeType.CO_REQUISITE:
            if first.activity_id != last.activity_id:
                errors.append("Co-requisites require a joint authored activity")
            continue
        if (first.grade - 2) * budget + first.end_minute > (
            last.grade - 2
        ) * budget + last.start_minute:
            errors.append("Prerequisite taught after dependent")
        for g in range(first.grade + 1, last.grade + 1):
            bridge = [
                u
                for u in units
                if u.kind == "bridge"
                and edge.source in u.micro_skill_ids
                and u.grade == g
                and u.period == 0
                and (u.grade - 2) * budget + u.end_minute
                <= (last.grade - 2) * budget + last.start_minute
            ]
            if not bridge:
                errors.append(f"Missing continuity bridge for {edge.source} in grade {g}")
    return tuple(errors)


def grade_errors(  # noqa: C901 - independently audit each calendar invariant
    grade: SolvedGradeCurriculum, settings: SequencingSettings, contexts: dict[str, Any]
) -> tuple[str, ...]:
    errors = []
    budget = settings.budget_hours * 60
    if grade.total_minutes != budget or grade.budget_minutes != budget:
        errors.append("Scenario budget violated")
    previous = 0
    used: dict[str, int] = {}
    for u in grade.units:
        if u.grade != grade.grade:
            errors.append("Unit grade differs from calendar grade")
        if u.start_minute != previous or u.end_minute - u.start_minute != u.total_minutes:
            errors.append("Calendar gap or overlap")
        previous = u.end_minute
        if (
            not 0 <= u.period < settings.periods
            or u.start_minute < u.period * budget // settings.periods
            or u.end_minute > (u.period + 1) * budget // settings.periods
        ):
            errors.append("Unit crosses a period boundary")
        if u.context_id:
            if u.context_id not in contexts or u.contexts != contexts[u.context_id].tags:
                errors.append("Invalid context tags")
            used[u.context_id] = used.get(u.context_id, 0) + u.total_minutes
    if previous != budget or sum(u.total_minutes for u in grade.units) != budget:
        errors.append("Calendar does not fill the scenario budget")
    if sum(used.values()) * 100 > settings.context_max_percent * budget or sum(
        bool(u.context_id) for u in grade.units
    ) * 100 > settings.context_max_percent * len(grade.units):
        errors.append("Contextualization cap exceeded")
    if used and len(used) < settings.context_min_variety:
        errors.append("Context variety insufficient")
    if any(value * 100 > settings.context_single_max_percent * budget for value in used.values()):
        errors.append("Single context dominates grade")
    return tuple(errors)
