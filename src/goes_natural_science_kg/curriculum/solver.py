# src/goes_natural_science_kg/curriculum/solver.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Solve finite educational activity choices with integer temporal constraints.
from __future__ import annotations

from importlib.metadata import version
from typing import Any

from ortools.sat.python import cp_model

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.sequencing import (
    ConstraintModel,
    SequencingInput,
    SequencingSettings,
    SolverReport,
)
from goes_natural_science_kg.schemas.skills import EdgeType


def period_bounds(grade: int, period: int, settings: SequencingSettings) -> tuple[int, int]:
    budget = settings.budget_hours * 60
    offset = (grade - 2) * budget
    return offset + period * budget // settings.periods, offset + (
        period + 1
    ) * budget // settings.periods


def placements(data: SequencingInput, settings: SequencingSettings) -> ConstraintModel:
    model = cp_model.CpModel()
    selected, starts, grades, choices = {}, {}, {}, {}
    intervals = []
    horizon = 5 * settings.budget_hours * 60
    micros = {m.id: m for m in data.graph.micro_skills}
    for a in sorted(data.activities, key=lambda a: a.id):
        selected[a.id] = model.new_bool_var(a.id + ":selected")
        starts[a.id] = model.new_int_var(0, horizon, a.id + ":start")
        grades[a.id] = model.new_int_var(2, 6, a.id + ":grade")
        allowed = [
            g
            for g in sorted(a.allowed_grades)
            if a.kind != "teach"
            or all(
                micros[k].grade_band.minimum <= g <= micros[k].grade_band.maximum
                for k in a.micro_skill_ids
            )
        ]
        for g in allowed:
            for t in range(settings.periods):
                lower, upper = period_bounds(g, t, settings)
                if a.minutes > upper - lower:
                    continue
                z = model.new_bool_var(f"{a.id}:{g}:{t}")
                choices[a.id, g, t] = z
                model.add(starts[a.id] >= lower).only_enforce_if(z)
                model.add(starts[a.id] + a.minutes <= upper).only_enforce_if(z)
        own = [(key, z) for key, z in choices.items() if key[0] == a.id]
        model.add(sum(z for _, z in own) == selected[a.id])
        model.add(grades[a.id] == sum(key[1] * z for key, z in own) + 2 * (1 - selected[a.id]))
        model.add(starts[a.id] == 0).only_enforce_if(selected[a.id].Not())
        if a.mandatory:
            model.add(selected[a.id] == 1)
        intervals.append(
            model.new_optional_fixed_size_interval_var(
                starts[a.id], a.minutes, selected[a.id], a.id
            )
        )
    model.add_no_overlap(intervals)
    for g in range(2, 7):
        model.add(
            sum(
                a.minutes * z
                for a in data.activities
                for (key, grade, _), z in choices.items()
                if key == a.id and grade == g
            )
            == settings.budget_hours * 60
        )
    return ConstraintModel(
        model=model,
        selected=selected,
        starts=starts,
        grades=grades,
        placements=choices,
        contexts={},
        components={},
    )


def dependencies(data: SequencingInput, settings: SequencingSettings, c: ConstraintModel) -> None:
    model = c.model
    horizon = 5 * settings.budget_hours * 60
    for micro in data.graph.micro_skills:
        initial = [
            a for a in data.activities if a.kind == "teach" and micro.id in a.micro_skill_ids
        ]
        model.add(sum(c.selected[a.id] for a in initial) == 1)
        key = "micro:" + micro.id
        c.starts[key] = model.new_int_var(0, horizon, key)
        c.starts[key + ":end"] = model.new_int_var(0, horizon, key + ":end")
        c.grades[key] = model.new_int_var(2, 6, key + ":grade")
        for a in initial:
            model.add(c.starts[key] == c.starts[a.id]).only_enforce_if(c.selected[a.id])
            model.add(c.starts[key + ":end"] == c.starts[a.id] + a.minutes).only_enforce_if(
                c.selected[a.id]
            )
            model.add(c.grades[key] == c.grades[a.id]).only_enforce_if(c.selected[a.id])
        for a in data.activities:
            if a.kind != "teach" and micro.id in a.micro_skill_ids:
                model.add(c.starts[a.id] >= c.starts[key + ":end"]).only_enforce_if(
                    c.selected[a.id]
                )
    for edge in data.graph.edges:
        source, target = "micro:" + edge.source, "micro:" + edge.target
        if edge.type == EdgeType.PREREQUISITE:
            model.add(c.starts[source + ":end"] <= c.starts[target])
            add_bridges(data, settings, c, edge.source, edge.target)
        elif edge.type == EdgeType.CO_REQUISITE:
            model.add(c.starts[source] == c.starts[target])
            model.add(c.starts[source + ":end"] == c.starts[target + ":end"])
    for thread in data.continuity:
        first, last = thread.opening_activity_id, thread.continuation_activity_id
        model.add(c.selected[first] == 1)
        model.add(c.selected[last] == 1)
        model.add(c.grades[last] == c.grades[first] + 1)
        model.add(sum(z for (key, _, t), z in c.placements.items() if key == last and t == 0) == 1)


def add_bridges(
    data: SequencingInput,
    settings: SequencingSettings,
    c: ConstraintModel,
    source: str,
    target: str,
) -> None:
    for g in range(3, 7):
        earlier = c.model.new_bool_var(f"{source}:{target}:{g}:earlier")
        later = c.model.new_bool_var(f"{source}:{target}:{g}:later")
        c.model.add(c.grades["micro:" + source] < g).only_enforce_if(earlier)
        c.model.add(c.grades["micro:" + source] >= g).only_enforce_if(earlier.Not())
        c.model.add(c.grades["micro:" + target] >= g).only_enforce_if(later)
        c.model.add(c.grades["micro:" + target] < g).only_enforce_if(later.Not())
        bridges = [
            a
            for a in data.activities
            if a.kind == "bridge" and source in a.micro_skill_ids and (a.id, g, 0) in c.placements
        ]
        c.model.add(sum(c.placements[a.id, g, 0] for a in bridges) >= 1).only_enforce_if(
            [earlier, later]
        )
        for a in bridges:
            c.model.add(c.starts[a.id] + a.minutes <= c.starts["micro:" + target]).only_enforce_if(
                [earlier, later, c.placements[a.id, g, 0]]
            )


def contexts(data: SequencingInput, settings: SequencingSettings, c: ConstraintModel) -> None:
    for a in data.activities:
        own = []
        for context_id in sorted(a.context_tasks):
            for g in range(2, 7):
                z = c.model.new_bool_var(f"context:{a.id}:{context_id}:{g}")
                c.contexts[a.id, context_id, g] = z
                c.model.add(
                    z
                    <= sum(
                        v
                        for (key, grade, _), v in c.placements.items()
                        if key == a.id and grade == g
                    )
                )
                own.append(z)
        c.model.add(sum(own) <= c.selected[a.id])
    acts = {a.id: a for a in data.activities}
    diversity = []
    for g in range(2, 7):
        chosen = [(key, z) for key, z in c.contexts.items() if key[2] == g]
        c.model.add(
            100 * sum(acts[key[0]].minutes * z for key, z in chosen)
            <= settings.context_max_percent * settings.budget_hours * 60
        )
        grade_count = sum(z for (_, grade, _), z in c.placements.items() if grade == g)
        c.model.add(100 * sum(z for _, z in chosen) <= settings.context_max_percent * grade_count)
        used = []
        for context in data.contexts:
            group = [(key, z) for key, z in chosen if key[1] == context.id]
            present = c.model.new_bool_var(f"context-used:{context.id}:{g}")
            c.model.add_max_equality(present, [z for _, z in group] + [0])
            c.model.add(
                100 * sum(acts[key[0]].minutes * z for key, z in group)
                <= settings.context_single_max_percent * settings.budget_hours * 60
            )
            used.append(present)
        any_used = c.model.new_bool_var(f"context-any:{g}")
        c.model.add_max_equality(any_used, [*used, 0])
        c.model.add(sum(used) >= settings.context_min_variety * any_used)
        diversity.extend(used)
    c.components["context_variety"] = -60 * sum(diversity)


def domain_minutes(data: SequencingInput, activity_id: str) -> dict[str, int]:
    a = next(a for a in data.activities if a.id == activity_id)
    micros = {m.id: m for m in data.graph.micro_skills}
    skills = {s.id: s for s in data.graph.skills}
    domains = sorted({skills[micros[k].parent_skill_id].domain for k in a.micro_skill_ids})
    return {
        domain: a.minutes // len(domains) + (i < a.minutes % len(domains))
        for i, domain in enumerate(domains)
    }


def absolute_minutes(c: ConstraintModel, expression: Any, upper: int, label: str) -> Any:
    deviation = c.model.new_int_var(0, upper, label + ":scaled")
    result = c.model.new_int_var(0, upper, label)
    c.model.add_abs_equality(deviation, expression)
    c.model.add_division_equality(result, deviation + 99, 100)
    return result


def objectives(data: SequencingInput, settings: SequencingSettings, c: ConstraintModel) -> None:
    budget = settings.budget_hours * 60
    balance, cognition, theme_flags = [], [], []
    allocations = {a.id: domain_minutes(data, a.id) for a in data.activities}
    for g in range(2, 7):
        for domain, percent in settings.domain_target_percent.items():
            minutes = sum(
                allocations[key].get(domain, 0) * z
                for (key, grade, _), z in c.placements.items()
                if grade == g
            )
            balance.append(
                absolute_minutes(
                    c, 100 * minutes - percent * budget, 100 * budget, f"domain:{domain}:{g}"
                )
            )
        knowing = sum(
            a.minutes * z
            for a in data.activities
            if a.cognitive_domain.value == "knowing"
            for (key, grade, _), z in c.placements.items()
            if key == a.id and grade == g
        )
        reasoning = sum(
            a.minutes * z
            for a in data.activities
            if a.cognitive_domain.value == "reasoning"
            for (key, grade, _), z in c.placements.items()
            if key == a.id and grade == g
        )
        for expression, label in [
            (100 * knowing - settings.knowing_max_percent * budget, "knowing"),
            (settings.reasoning_min_percent[g] * budget - 100 * reasoning, "reasoning"),
        ]:
            excess = c.model.new_int_var(0, 100 * budget, label + str(g))
            c.model.add_max_equality(excess, [expression, 0])
            rounded = c.model.new_int_var(0, budget, label + str(g) + ":minutes")
            c.model.add_division_equality(rounded, excess + 99, 100)
            cognition.append(rounded)
        for t in range(settings.periods):
            for theme in sorted({a.theme for a in data.activities}):
                active = c.model.new_bool_var(f"theme:{g}:{t}:{theme}")
                c.model.add_max_equality(
                    active,
                    [
                        z
                        for (key, grade, period), z in c.placements.items()
                        if grade == g
                        and period == t
                        and next(a.theme for a in data.activities if a.id == key) == theme
                    ]
                    + [0],
                )
                theme_flags.append(active)
    c.components["domain_balance"] = sum(balance)
    c.components["cognitive_balance"] = sum(cognition)
    c.components["inquiry"] = -sum(
        a.minutes * c.selected[a.id] for a in data.activities if a.inquiry
    )
    c.components["thematic_coherence"] = 60 * sum(theme_flags)
    c.components["spacing"] = spacing(data, settings, c)
    c.model.minimize(
        sum(getattr(settings.weights, name) * value for name, value in c.components.items())
    )


def spacing(data: SequencingInput, settings: SequencingSettings, c: ConstraintModel) -> Any:
    losses = []
    for micro_id in data.key_micro_ids:
        periods = []
        for g in range(2, 7):
            same_grade = c.model.new_bool_var(f"same-grade:{micro_id}:{g}")
            c.model.add(c.grades["micro:" + micro_id] == g).only_enforce_if(same_grade)
            c.model.add(c.grades["micro:" + micro_id] != g).only_enforce_if(same_grade.Not())
            for t in range(settings.periods):
                flags = []
                for a in data.activities:
                    if (
                        a.kind != "retrieval"
                        or micro_id not in a.micro_skill_ids
                        or (a.id, g, t) not in c.placements
                    ):
                        continue
                    enough = c.model.new_bool_var(f"gap:{micro_id}:{a.id}:{g}:{t}")
                    c.model.add(
                        c.starts[a.id]
                        >= c.starts["micro:" + micro_id + ":end"] + settings.spacing_minutes
                    ).only_enforce_if(enough)
                    c.model.add(
                        c.starts[a.id]
                        < c.starts["micro:" + micro_id + ":end"] + settings.spacing_minutes
                    ).only_enforce_if(enough.Not())
                    spaced = c.model.new_bool_var(f"spaced:{micro_id}:{a.id}:{g}:{t}")
                    c.model.add_min_equality(spaced, [enough, same_grade, c.placements[a.id, g, t]])
                    flags.append(spaced)
                used = c.model.new_bool_var(f"retrieved:{micro_id}:{g}:{t}")
                c.model.add_max_equality(used, [*flags, 0])
                periods.append(used)
        missing = c.model.new_int_var(0, settings.retrieval_target, "missing:" + micro_id)
        c.model.add_max_equality(missing, [settings.retrieval_target - sum(periods), 0])
        losses.append(missing)
    return 60 * sum(losses)


def solve(data: SequencingInput, settings: SequencingSettings) -> SolverReport:
    from goes_natural_science_kg.curriculum.validation import audit, extract, preflight

    errors = preflight(data, settings)
    common = dict(
        input_sha256=content_hash(data),
        settings_sha256=content_hash(settings),
        solver_version=version("ortools"),
    )
    if errors:
        return SolverReport(status="invalid", hard_errors=errors, **common)
    c = placements(data, settings)
    dependencies(data, settings, c)
    contexts(data, settings, c)
    objectives(data, settings, c)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = settings.seed
    solver.parameters.max_deterministic_time = settings.max_deterministic_time
    status = solver.solve(c.model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolverReport(
            status="infeasible" if status == cp_model.INFEASIBLE else "unknown",
            hard_errors=("No certified schedule: " + solver.status_name(status),),
            **common,
        )
    grades = extract(data, settings, c, solver)
    hard_errors, coverage, metrics = audit(data, settings, grades)
    return SolverReport(
        status="invalid"
        if hard_errors
        else "optimal"
        if status == cp_model.OPTIMAL
        else "feasible",
        hard_errors=hard_errors,
        coverage=coverage,
        grade_metrics=metrics,
        grades=grades if not hard_errors else (),
        objective_value=solver.objective_value,
        best_bound=solver.best_objective_bound,
        objective_components={
            name: int(solver.value(value)) for name, value in c.components.items()
        },
        **common,
    )
