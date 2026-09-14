# tests/unit/test_sequencing.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Exercise real CP-SAT solutions and adversarial calendar certificates.
from pathlib import Path

import pytest

from goes_natural_science_kg.curriculum.artifacts import graph_input, read_input, report_html
from goes_natural_science_kg.curriculum.solver import solve
from goes_natural_science_kg.curriculum.validation import audit
from goes_natural_science_kg.graph.linked_data import export_jsonld, import_jsonld
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.sequencing import SequencingInput, SequencingSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data():
    return read_input(ROOT / "tests/golden/sequencing/engineering.json")


@pytest.fixture(scope="module")
def solution(data):
    settings = SequencingSettings(max_deterministic_time=1)
    result = solve(data, settings)
    assert result.status in {"optimal", "feasible"}
    return settings, result


def test_exact_budget_coverage_and_independent_audit(data, solution):
    settings, report = solution
    assert len(report.grades) == 5
    assert all(g.total_minutes == 9600 for g in report.grades)
    assert all(report.coverage.values())
    assert not audit(data, settings, report.grades)[0]
    assert all("context" not in type(m).model_fields for m in data.graph.micro_skills)


def test_repeated_solver_is_deterministic(data, solution):
    settings, report = solution
    assert canonical_json(solve(data, settings)) == canonical_json(report)


def test_missing_minutes_and_uncovered_skills_fail_without_padding(data):
    minimal = graph_input(data.graph)
    report = solve(minimal, SequencingSettings())
    assert report.status == "invalid" and not report.grades
    assert any("padding is forbidden" in error for error in report.hard_errors)


def test_jsonld_roundtrip_preserves_graph_and_excludes_context(data):
    document = export_jsonld(data.graph)
    assert import_jsonld(document) == data.graph
    assert "context_tasks" not in canonical_json(document)
    assert "water-sanitation" not in canonical_json(document)


def test_140_hour_scenario_uses_same_activity_bank(data):
    result = solve(data, SequencingSettings(budget_hours=140, max_deterministic_time=1))
    assert result.status in {"optimal", "feasible"}
    assert {g.total_minutes for g in result.grades} == {8400}


def test_cognitive_relabeling_is_detected(data, solution):
    settings, report = solution
    grade = report.grades[0]
    first = grade.units[0]
    from goes_natural_science_kg.schemas.skills import CognitiveDomain

    changed = (
        CognitiveDomain.REASONING
        if first.cognitive_domain != CognitiveDomain.REASONING
        else CognitiveDomain.KNOWING
    )
    tampered = first.model_copy(update={"cognitive_domain": changed})
    grades = (grade.model_copy(update={"units": (tampered, *grade.units[1:])}), *report.grades[1:])
    assert any("authored activity" in error for error in audit(data, settings, grades)[0])


def test_context_caps_and_abstract_units(data, solution):
    _, report = solution
    for grade in report.grades:
        contextual = [u for u in grade.units if u.context_id]
        assert len(contextual) * 100 <= 60 * len(grade.units)
        assert sum(u.total_minutes for u in contextual) * 100 <= 60 * grade.total_minutes
        assert not contextual or len({u.context_id for u in contextual}) >= 3
        assert any(not u.contexts for u in grade.units)


def test_html_escapes_authored_content(data, solution):
    settings, report = solution
    dangerous = data.model_copy(update={"approval": "<script>alert(1)</script>"})
    result = report_html(dangerous, settings, report)
    assert "<script>alert(1)</script>" not in result
    assert "&lt;script&gt;" in result


def test_provisional_graph_keeps_chunk_evidence():
    data = read_input(ROOT / "data/processed/curriculum/provisional-graph.json")
    assert data.graph.schema_version == "graph-snapshot/2.0"
    assert all(ref.startswith("chunk-") for m in data.graph.micro_skills for ref in m.source_refs)
    assert import_jsonld(export_jsonld(data.graph)) == data.graph
    with pytest.raises(ValueError, match="reviewer"):
        SequencingInput.model_validate(data.model_dump() | {"approval": "approved"})


def with_cross_grade_edge(data):
    from goes_natural_science_kg.schemas.identity import Identity, stable_id
    from goes_natural_science_kg.schemas.skills import Edge, EdgeType, PrerequisiteRef

    source = next(a.micro_skill_ids[0] for a in data.activities if a.id == "teach-2-0")
    target = next(a.micro_skill_ids[0] for a in data.activities if a.id == "teach-3-0")
    ident = Identity(namespace="test", key="cross-grade")
    edge = Edge(
        id=stable_id("edge", ident),
        identity=ident,
        source=source,
        target=target,
        type=EdgeType.PREREQUISITE,
        strength=1.0,
        justification="Synthetic continuity test.",
        source_refs=data.graph.source_ids,
    )
    micros = tuple(
        m.model_copy(
            update={
                "prerequisites": (
                    *m.prerequisites,
                    PrerequisiteRef(id=source, type=EdgeType.PREREQUISITE),
                )
            }
        )
        if m.id == target
        else m
        for m in data.graph.micro_skills
    )
    graph = type(data.graph).model_validate(
        data.graph.model_dump() | {"micro_skills": micros, "edges": (*data.graph.edges, edge)}
    )
    return data.model_copy(update={"graph": graph}), source


def test_cross_grade_dependency_requires_bridge(data):
    changed, _ = with_cross_grade_edge(data)
    result = solve(changed, SequencingSettings(max_deterministic_time=1))
    assert result.status == "infeasible" and not result.grades


def test_explicit_bridge_resumes_next_grade(data):
    from goes_natural_science_kg.schemas.sequencing import ContinuityThread, LearningActivity

    changed, source = with_cross_grade_edge(data)
    bridge = LearningActivity(
        id="bridge-grade-3",
        kind="bridge",
        micro_skill_ids=(source,),
        minutes=300,
        cognitive_domain="knowing",
        theme="physical-science",
        task="Synthetic bridge task.",
        mastery_criterion="Recall the synthetic prior observation.",
        allowed_grades=(3,),
    )
    changed = changed.model_copy(
        update={
            "activities": (*changed.activities, bridge),
            "continuity": (
                ContinuityThread(
                    id="resume", opening_activity_id="teach-2-0", continuation_activity_id=bridge.id
                ),
            ),
        }
    )
    report = solve(changed, SequencingSettings(max_deterministic_time=1))
    assert report.status in {"optimal", "feasible"}
    unit = next(u for g in report.grades for u in g.units if u.activity_id == bridge.id)
    assert unit.grade == 3 and unit.period == 0


def test_objective_weights_are_applied_explicitly(solution):
    settings, report = solution
    assert report.objective_value == sum(
        getattr(settings.weights, key) * value for key, value in report.objective_components.items()
    )


def test_prerequisite_cycle_is_rejected(data):
    from goes_natural_science_kg.schemas.identity import Identity, stable_id
    from goes_natural_science_kg.schemas.skills import Edge, EdgeType, PrerequisiteRef

    existing = data.graph.edges[0]
    identity = Identity(namespace="test", key="reverse-edge")
    reverse = Edge(
        id=stable_id("edge", identity),
        identity=identity,
        source=existing.target,
        target=existing.source,
        type=EdgeType.PREREQUISITE,
        strength=1.0,
        justification="Synthetic cycle.",
        source_refs=data.graph.source_ids,
    )
    micros = tuple(
        m.model_copy(
            update={
                "prerequisites": (
                    *m.prerequisites,
                    PrerequisiteRef(id=reverse.source, type=EdgeType.PREREQUISITE),
                )
            }
        )
        if m.id == reverse.target
        else m
        for m in data.graph.micro_skills
    )
    graph = type(data.graph).model_validate(
        data.graph.model_dump() | {"micro_skills": micros, "edges": (*data.graph.edges, reverse)}
    )
    result = solve(data.model_copy(update={"graph": graph}), SequencingSettings())
    assert result.status == "invalid" and "Prerequisite cycle" in result.hard_errors


def test_changed_budget_certificate_is_rejected(data, solution):
    _, report = solution
    errors, _, _ = audit(data, SequencingSettings(budget_hours=140), report.grades)
    assert "Scenario budget violated" in errors
    assert "Grade provenance differs from build input/settings" in errors


def test_failed_build_archives_previous_calendars(tmp_path, data, solution):
    from goes_natural_science_kg.curriculum.artifacts import write_build

    settings, report = solution
    write_build(tmp_path, data, settings, report)
    assert len(list(tmp_path.glob("grade_*.json"))) == 5
    failed_data = graph_input(data.graph)
    failed = solve(failed_data, settings)
    write_build(tmp_path, failed_data, settings, failed)
    assert not list(tmp_path.glob("grade_*.json"))
    assert len(list((tmp_path / "previous").rglob("grade_*.json"))) == 5
    assert "Build failed" in (tmp_path / "report.html").read_text()


def test_cli_failure_returns_nonzero_and_report(tmp_path):
    from typer.testing import CliRunner

    from goes_natural_science_kg.cli import app

    result = CliRunner().invoke(
        app,
        [
            "build",
            "--skills-map",
            str(ROOT / "data/processed/curriculum/provisional-graph.json"),
            "--budget-hours",
            "160",
            "--seed",
            "42",
            "--output",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert (tmp_path / "build.json").is_file()
    assert not list(tmp_path.glob("grade_*.json"))


def test_corequisites_need_joint_authored_teaching(data):
    from goes_natural_science_kg.schemas.skills import EdgeType, PrerequisiteRef

    edge = data.graph.edges[0]
    changed_edge = edge.model_copy(update={"type": EdgeType.CO_REQUISITE})
    micros = tuple(
        m.model_copy(
            update={
                "prerequisites": tuple(
                    PrerequisiteRef(id=p.id, type=EdgeType.CO_REQUISITE)
                    if m.id == edge.target and p.id == edge.source
                    else p
                    for p in m.prerequisites
                )
            }
        )
        for m in data.graph.micro_skills
    )
    micros = tuple(
        m.model_copy(
            update={
                "prerequisites": (
                    *m.prerequisites,
                    PrerequisiteRef(id=edge.target, type=EdgeType.CO_REQUISITE),
                )
            }
        )
        if m.id == edge.source
        else m
        for m in micros
    )
    graph = type(data.graph).model_validate(
        data.graph.model_dump()
        | {"micro_skills": micros, "edges": (changed_edge, *data.graph.edges[1:])}
    )
    changed = data.model_copy(update={"graph": graph})
    settings = SequencingSettings(max_deterministic_time=1)
    assert solve(changed, settings).status == "infeasible"
    source = next(
        a for a in data.activities if a.kind == "teach" and edge.source in a.micro_skill_ids
    )
    target = next(
        a for a in data.activities if a.kind == "teach" and edge.target in a.micro_skill_ids
    )
    joint = source.model_copy(
        update={
            "id": "joint-teaching",
            "micro_skill_ids": (edge.source, edge.target),
            "minutes": source.minutes + target.minutes,
            "context_tasks": {},
        }
    )
    changed = changed.model_copy(
        update={
            "activities": (
                *(a for a in data.activities if a.id not in (source.id, target.id)),
                joint,
            )
        }
    )
    result = solve(changed, settings)
    assert result.status in ("feasible", "optimal")
    assert not audit(changed, settings, result.grades)[0]


def test_activity_permutations_preserve_canonical_input(data):
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=8, deadline=None)
    @given(st.permutations(tuple(range(len(data.activities)))))
    def check(order):
        permuted = SequencingInput.model_validate(
            data.model_dump() | {"activities": tuple(data.activities[i] for i in order)}
        )
        assert canonical_json(permuted) == canonical_json(data)
        assert import_jsonld(export_jsonld(permuted.graph)) == data.graph

    check()
