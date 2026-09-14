# tests/unit/test_contracts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Reject invalid contracts and preserve context-free stable graph identity.
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.curriculum import GradeCurriculum
from goes_natural_science_kg.schemas.graph import GraphSnapshot
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.skills import GradeBand, MicroSkill, Skill

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def test_ids_are_stable_and_revision_hashes_change(before, after):
    old = {m.identity.key: m for m in before.micro_skills}
    new = {m.identity.key: m for m in after.micro_skills}
    assert old["explain-flow"].id == new["explain-flow"].id
    assert content_hash(old["explain-flow"]) != content_hash(new["explain-flow"])
    assert old["explain-flow"].grade_band.minimum == 4
    assert new["explain-flow"].grade_band.minimum == 3


def test_frozen_and_closed(before):
    micro = before.micro_skills[0]
    with pytest.raises(ValidationError, match="frozen"):
        micro.estimated_minutes = 1
    for model in [before, before.skills[0], micro, before.edges[0]]:
        with pytest.raises(ValidationError, match="Extra inputs"):
            type(model).model_validate({**model.model_dump(), "contexts": []})


@pytest.mark.parametrize(
    "field,value",
    [
        ("estimated_minutes", -1),
        ("estimated_minutes", 0),
        ("estimated_minutes", 1.5),
        ("estimated_minutes", True),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("observable_verb", "   "),
        ("knowledge_object", ""),
        ("evidence_of_mastery", ""),
        ("cognitive_domain", "memorizing"),
        ("source_refs", []),
    ],
)
def test_invalid_micro_fields(before, field, value):
    with pytest.raises(ValidationError):
        MicroSkill.model_validate({**before.micro_skills[0].model_dump(), field: value})


@pytest.mark.parametrize(
    "values",
    [{"minimum": 4, "maximum": 3}, {"minimum": 1, "maximum": 6}, {"minimum": 2, "maximum": 7}],
)
def test_grade_bands(values):
    with pytest.raises(ValidationError):
        GradeBand.model_validate(values)


def test_identity_mismatch_rejected(before):
    data = before.skills[0].model_dump()
    data["identity"] = {"namespace": "test", "key": "different-concept"}
    with pytest.raises(ValidationError, match="canonical identity"):
        Skill.model_validate(data)


def test_deterministic_id_fixture():
    vector = json.loads((GOLDEN / "identity-vector.json").read_text())
    assert (
        stable_id(vector["kind"], Identity.model_validate(vector["identity"]))
        == vector["expected_id"]
    )


def test_context_in_curriculum_and_exact_arithmetic():
    data = json.loads((GOLDEN / "grade-curriculum.json").read_text())
    grade = GradeCurriculum.model_validate(data)
    assert grade.units[0].contexts[0].domain.value == "environmental"
    assert grade.total_minutes == 50
    for total in [49, 51]:
        with pytest.raises(ValidationError, match="unit totals"):
            GradeCurriculum.model_validate({**data, "total_minutes": total})
    data["units"][0]["total_minutes"] += 1
    with pytest.raises(ValidationError, match="minute components"):
        GradeCurriculum.model_validate(data)


def test_budget_and_shared_skill_accounting():
    data = json.loads((GOLDEN / "grade-curriculum.json").read_text())
    data["units"][0].update(instruction_minutes=9590, total_minutes=9620)
    data["total_minutes"] = 9620
    with pytest.raises(ValidationError, match="budget exceeded"):
        GradeCurriculum.model_validate(data)
    data["budget_minutes"] = 20000
    with pytest.raises(ValidationError):
        GradeCurriculum.model_validate(data)


def test_no_duplicate_units():
    data = json.loads((GOLDEN / "grade-curriculum.json").read_text())
    data["units"] *= 2
    data["total_minutes"] *= 2
    with pytest.raises(ValidationError, match="duplicate units"):
        GradeCurriculum.model_validate(data)


def test_graph_integrity(before):
    data = before.model_dump(mode="json")
    with pytest.raises(ValidationError, match="duplicate skills"):
        GraphSnapshot.model_validate({**data, "skills": data["skills"] * 2})
    with pytest.raises(ValidationError, match="parent_skill_id"):
        GraphSnapshot.model_validate({**data, "skills": []})
    with pytest.raises(ValidationError, match="unregistered"):
        GraphSnapshot.model_validate({**data, "source_ids": []})
    with pytest.raises(ValidationError, match="inline prerequisites"):
        GraphSnapshot.model_validate({**data, "edges": []})
    data["edges"][0]["target"] = stable_id("micro", Identity(namespace="test", key="missing"))
    with pytest.raises(ValidationError, match="dangling edge"):
        GraphSnapshot.model_validate(data)


def test_round_trip(before):
    assert GraphSnapshot.model_validate_json(canonical_json(before)) == before


def test_source_lifecycle():
    source = json.loads((GOLDEN / "source-candidates.json").read_text())[0]
    for updates in [
        dict(sha256="0" * 64),
        dict(license_gate_status="rejected"),
        dict(license_gate_status="accepted"),
        dict(url="file:///etc/passwd"),
    ]:
        with pytest.raises(ValidationError):
            SourceDocument.model_validate({**source, **updates})
