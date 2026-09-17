# tests/integration/test_proposal_snapshot.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Audit the published real proposal independently of its model review outcomes.
from itertools import pairwise
from pathlib import Path

from goes_natural_science_kg.curriculum.hints import schedule_hint
from goes_natural_science_kg.curriculum.validation import audit
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.release import BindingAudit, DraftAttempt, ProposalMeasurement
from goes_natural_science_kg.schemas.sequencing import (
    SequencingInput,
    SequencingSettings,
    SolverReport,
)

ROOT = Path("data/processed/proposals/2026-09-17")


def test_real_proposal_has_audited_calendars_and_retains_rejected_draft():
    draft = DraftAttempt.model_validate_json((ROOT / "v1/attempt.json").read_bytes())
    assert any(draft.structural_errors.values())
    data = SequencingInput.model_validate_json((ROOT / "v5/skills-map.json").read_bytes())
    settings = SequencingSettings.model_validate_json((ROOT / "v5/settings.json").read_bytes())
    report = SolverReport.model_validate_json((ROOT / "v5/build.json").read_bytes())
    assert report.status in {"feasible", "optimal"}
    errors, coverage, metrics = audit(data, settings, report.grades)
    assert not errors and all(coverage.values())
    assert {g.total_minutes for g in report.grades} == {9600}
    assert len(data.continuity) == 4 and len(metrics) == 5
    assert report.input_sha256 == content_hash(data)
    assert data.approval == "provisional"
    bindings = BindingAudit.model_validate_json((ROOT / "v5/binding-audit.json").read_bytes())
    assert bindings.checked >= len(data.graph.micro_skills) and not bindings.errors
    assert not bindings.semantic_entailment_verified
    measurement = ProposalMeasurement.model_validate_json(
        (ROOT / "solver-measurement.json").read_bytes()
    )
    assert measurement.byte_identical_report


def test_warm_start_is_finite_topological_and_deterministic():
    data = SequencingInput.model_validate_json((ROOT / "v5/skills-map.json").read_bytes())
    settings = SequencingSettings.model_validate_json((ROOT / "v5/settings.json").read_bytes())
    hints = schedule_hint(data, settings)
    assert hints and hints == schedule_hint(data, settings)
    activities = {a.id: a for a in data.activities}
    teaching = {
        m: h
        for h in hints
        if activities[h.activity_id].kind == "teach"
        for m in activities[h.activity_id].micro_skill_ids
    }
    assert set(teaching) == {m.id for m in data.graph.micro_skills}
    for edge in data.graph.edges:
        if edge.type.value == "PREREQUISITE":
            assert teaching[edge.source].end <= teaching[edge.target].start
    for grade in range(2, 7):
        calendar = sorted((h for h in hints if h.grade == grade), key=lambda h: h.start)
        assert calendar[0].start == (grade - 2) * 9600 and calendar[-1].end == (grade - 1) * 9600
        assert all(a.end == b.start for a, b in pairwise(calendar))
        assert all(h.end - h.start == activities[h.activity_id].minutes for h in calendar)


def test_context_seed_uses_authored_alternatives_and_obeys_caps():
    from goes_natural_science_kg.curriculum.hints import context_hint

    data = SequencingInput.model_validate_json((ROOT / "v5/skills-map.json").read_bytes())
    settings = SequencingSettings.model_validate_json((ROOT / "v5/settings.json").read_bytes())
    hints = schedule_hint(data, settings)
    selected = context_hint(data, settings, hints)
    activities = {a.id: a for a in data.activities}
    for grade in range(2, 7):
        choices = [(key, context) for key, context, g in selected if g == grade]
        assert len({c for _, c in choices}) >= settings.context_min_variety
        assert all(c in activities[k].context_tasks for k, c in choices)
        assert (
            100 * sum(activities[k].minutes for k, _ in choices)
            <= settings.context_max_percent * settings.budget_hours * 60
        )
    assert not context_hint(data, settings.model_copy(update={"context_max_percent": 0}), hints)
