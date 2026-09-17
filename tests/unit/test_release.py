# tests/unit/test_release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify proposal review gates without simulating provider responses.
from goes_natural_science_kg.agents.release import design_errors, panel_accepts
from goes_natural_science_kg.schemas.release import (
    ReleaseActivity,
    ReleaseAnchor,
    ReleaseIssue,
    ReleaseMicro,
    ReleaseSettings,
    ReleaseSkill,
    ReleaseVerdict,
    SupportedClaim,
    UnitAssignment,
    UnitDesign,
)


def sample_design():
    anchor = ReleaseAnchor(
        key="test-anchor",
        chunk_id="test-chunk",
        source_id="test-source",
        source_url="https://example.org/synthetic",
        page=1,
        paragraph_id="p1",
        start=0,
        end=40,
        text="A measured comparison needs a common unit.",
    )
    assignment = UnitAssignment(
        key="test-unit",
        grade=2,
        title="Synthetic contract fixture",
        weeks=1,
        anchors=(anchor,),
        continuity="Synthetic test only.",
    )
    skills = []
    activities = []
    for i in range(2):
        micros = []
        for j in range(4):
            key = f"micro-{i}-{j}"
            micros.append(
                ReleaseMicro(
                    key=key,
                    verb="compares",
                    knowledge_object="Two synthetic measurements.",
                    cognitive_domain="applying",
                    estimated_minutes=15,
                    misconceptions=("Synthetic misconception.",),
                    mastery="Select the larger record.",
                    assessment_task="Compare 2 and 3.",
                    expected_response="3 is larger.",
                    scoring_rule="One point for 3.",
                    support=(
                        SupportedClaim(
                            anchor_key=anchor.key,
                            quote="a common unit",
                            claim="Comparison uses a common unit.",
                        ),
                    ),
                )
            )
            for kind, minutes in [("teach", 15), ("practice", 30)]:
                activities.append(
                    ReleaseActivity(
                        key=f"{kind}-{key}",
                        micro_keys=(key,),
                        kind=kind,
                        minutes=minutes,
                        cognitive_domain="applying",
                        inquiry=False,
                        theme="measurement",
                        task="Compare two synthetic records.",
                        mastery_criterion="Select the larger record.",
                        materials=("Cards",),
                        safety="Paper only.",
                        differentiation="Read aloud.",
                        rationale_for_minutes="Demonstration and individual practice.",
                    )
                )
        skills.append(
            ReleaseSkill(
                key=f"skill-{i}",
                label="Synthetic comparison",
                domain="physical-science",
                pedagogical_rationale="Test fixture only.",
                micros=tuple(micros),
            )
        )
    return assignment, UnitDesign(
        unit_key=assignment.key,
        skills=tuple(skills),
        activities=tuple(activities),
        progression="Compare then apply.",
        limitations=("Synthetic test only.",),
    )


def test_panel_requires_evidence_and_age_and_rejects_critical_issues():
    settings = ReleaseSettings(project="test-project")
    votes = tuple(
        ReleaseVerdict(
            criterion=k, passed=True, score=0.9, issues=(), rationale="Synthetic gate fixture."
        )
        for k in settings.judge_models
    )
    assert panel_accepts(votes, (), settings)
    for mandatory in ("evidence", "age"):
        changed = tuple(
            v.model_copy(update={"passed": False}) if v.criterion == mandatory else v for v in votes
        )
        assert not panel_accepts(changed, (), settings)
    assert not panel_accepts(votes, ("invalid quote",), settings)
    issue = ReleaseIssue(
        entity_key="test",
        severity="critical",
        finding="Unsafe task.",
        required_change="Use safe materials.",
    )
    assert not panel_accepts(
        (votes[0].model_copy(update={"issues": (issue,)}), *votes[1:]), (), settings
    )


def test_quote_and_dependency_validation_detects_corruption():
    assignment, design = sample_design()
    assert not design_errors(design, assignment)
    micro = design.skills[0].micros[0]
    claim = micro.support[0].model_copy(update={"quote": "This quotation is invented."})
    changed = micro.model_copy(update={"support": (claim,), "prerequisites": (micro.key,)})
    first = design.skills[0].model_copy(update={"micros": (changed, *design.skills[0].micros[1:])})
    invalid = design.model_copy(update={"skills": (first, *design.skills[1:])})
    errors = design_errors(invalid, assignment)
    assert any("unverified exact quotation" in e for e in errors)
    assert any("invalid prerequisite" in e for e in errors)
    assert "prerequisite cycle" in errors


def test_compiled_proposal_preserves_ids_and_publishes_quote_hashes():
    from goes_natural_science_kg.curriculum.release import compile_proposal, proposal_diff
    from goes_natural_science_kg.schemas.identity import Identity, stable_id
    from goes_natural_science_kg.schemas.release import ReleaseUnitResult

    assignment, design = sample_design()
    anchor = assignment.anchors[0].model_copy(
        update={
            "source_id": stable_id("source", Identity(namespace="synthetic", key="evidence")),
            "chunk_id": "chunk-synthetic-0123456789abcdef",
        }
    )
    assignment = assignment.model_copy(update={"anchors": (anchor,)})
    result = ReleaseUnitResult(
        assignment=assignment,
        initial=design,
        final=design,
        panels=(),
        revisions=0,
        status="needs_human_review",
        request_ids=(),
    )
    before, pedagogy, bindings = compile_proposal((result,), (), "draft")
    assert before.approval == "provisional"
    assert len(pedagogy) == 8 and len(bindings) == 8
    assert all(b.location_verified for b in bindings)
    assert all("quote" not in b.model_dump() for b in bindings)
    changed_micro = (
        design.skills[0]
        .micros[0]
        .model_copy(update={"knowledge_object": "Compare two measurements using the same unit."})
    )
    skill = design.skills[0].model_copy(
        update={"micros": (changed_micro, *design.skills[0].micros[1:])}
    )
    changed = design.model_copy(update={"skills": (skill, *design.skills[1:])})
    after, _, _ = compile_proposal((result.model_copy(update={"final": changed}),), (), "proposal")
    diff = proposal_diff(before.graph, after.graph)
    assert len(diff.changes) == 1
    assert diff.changes[0].operation == "modified"
    assert diff.changes[0].changed_fields == ("knowledge_object",)


def test_transport_schema_keeps_types_without_weakening_local_contract():
    import pytest
    from pydantic import ValidationError

    from goes_natural_science_kg.agents.release import transport_schema

    schema = transport_schema(UnitDesign.model_json_schema())
    assert schema["type"] == "object"
    assert "maxItems" not in str(schema)
    with pytest.raises(ValidationError):
        UnitDesign(unit_key="unit", skills=(), activities=(), progression="Test", limitations=())


def test_optimizer_patch_preserves_unaffected_content():
    from goes_natural_science_kg.agents.release import apply_patch
    from goes_natural_science_kg.schemas.release import UnitPatch

    assignment, design = sample_design()
    changed = design.activities[0].model_copy(update={"task": "Revised observable comparison."})
    result = apply_patch(design, UnitPatch(unit_key=assignment.key, activities=(changed,)))
    assert result.activities[0].task == changed.task
    assert result.activities[1:] == design.activities[1:]
    assert result.skills == design.skills


def test_continuity_is_explicit_and_does_not_invent_graph_edges():
    import pytest

    from goes_natural_science_kg.curriculum.continuity import add_continuity
    from goes_natural_science_kg.curriculum.release import compile_proposal
    from goes_natural_science_kg.schemas.identity import Identity, stable_id
    from goes_natural_science_kg.schemas.release import ContinuityProposal, ReleaseUnitResult

    assignment, design = sample_design()
    anchor = assignment.anchors[0].model_copy(
        update={
            "source_id": stable_id("source", Identity(namespace="synthetic", key="evidence")),
            "chunk_id": "chunk-synthetic-0123456789abcdef",
        }
    )
    assignment = assignment.model_copy(update={"anchors": (anchor,)})
    results = tuple(
        ReleaseUnitResult(
            assignment=assignment.model_copy(update={"key": f"grade-{g}", "grade": g}),
            initial=design,
            final=design,
            panels=(),
            revisions=0,
            status="needs_human_review",
            request_ids=(),
        )
        for g in (2, 3)
    )
    data, _, _ = compile_proposal(results, (), "synthetic")
    source = next(m for m in data.graph.micro_skills if m.grade_band.minimum == 2)
    target = next(m for m in data.graph.micro_skills if m.grade_band.minimum == 3)
    handoff = ContinuityProposal(
        id="measurement-transfer",
        source_micro_id=source.id,
        target_micro_id=target.id,
        target_grade=3,
        task="Compare two records using a shared unit.",
        expected_response="3 exceeds 2 in the same unit.",
        scoring_rule="Select 3 and state the common unit.",
        minutes=15,
        rationale="Recall the established comparison before the next measurement task.",
    )
    revised = add_continuity(data, (handoff,))
    assert revised.graph == data.graph
    assert len(revised.continuity) == 1
    assert next(a for a in revised.activities if a.id.startswith("continuity-")).mandatory
    with pytest.raises(ValueError, match="adjacent"):
        add_continuity(data, (handoff.model_copy(update={"target_grade": 4}),))


def test_specialist_packet_separates_source_scope():
    from goes_natural_science_kg.agents.release_review import judge_packet
    from goes_natural_science_kg.schemas.release import ReleaseUnitResult

    assignment, design = sample_design()
    unrelated = assignment.anchors[0].model_copy(
        update={"key": "uncited", "source_id": "different-source", "text": "Unrelated page."}
    )
    assignment = assignment.model_copy(update={"anchors": (*assignment.anchors, unrelated)})
    result = ReleaseUnitResult(
        assignment=assignment,
        initial=design,
        final=design,
        panels=(),
        revisions=0,
        status="needs_human_review",
        request_ids=(),
    )
    assert not judge_packet(result, "cognitive").evidence
    assert len(judge_packet(result, "evidence").evidence) == 1
    assert len(judge_packet(result, "curricular").evidence) == 2


def test_real_pdf_binding_audit_detects_offset_and_source_corruption():
    import hashlib
    from pathlib import Path

    from goes_natural_science_kg.corpus.chunk import chunk_document
    from goes_natural_science_kg.eval.bindings import audit_bindings
    from goes_natural_science_kg.schemas.ingestion import ParsedDocument
    from goes_natural_science_kg.schemas.release import PublishedBinding

    path = Path(__file__).parents[1] / "golden/pdfs/el-salvador-science-2026.pdf.expected.json"
    document = ParsedDocument.model_validate_json(path.read_bytes())
    chunks = chunk_document(document)
    chunk = chunks[0]
    paragraph = next(
        p for p in document.paragraphs if p.id in chunk.paragraph_ids and len(p.text) > 30
    )
    binding = PublishedBinding(
        entity_id="contract-test",
        source_id=document.source_id,
        source_url="https://www.mined.gob.sv/",
        chunk_id=chunk.id,
        page=paragraph.page,
        paragraph_id=paragraph.id,
        quote_start=paragraph.start,
        quote_end=paragraph.end,
        quote_sha256=hashlib.sha256(paragraph.text.encode()).hexdigest(),
        claim="Fixture location check only.",
        location_verified=True,
    )
    args = (
        {document.source_id: document},
        {c.id: c for c in chunks},
        {document.source_id: document.document_sha256},
    )
    assert not audit_bindings((binding,), *args).errors
    assert audit_bindings(
        (binding.model_copy(update={"quote_start": binding.quote_start + 1}),), *args
    ).errors
    assert audit_bindings((binding,), args[0], args[1], {document.source_id: "0" * 64}).errors


def test_import_parse_cache_never_hides_file_edits(tmp_path):
    from goes_natural_science_kg.agents.runner import imported_modules

    module = tmp_path / "module.py"
    module.write_text("import example.first\n")
    assert imported_modules(module, "example") == {"example.first"}
    module.write_text("import example.second\n")
    assert imported_modules(module, "example") == {"example.second"}
    result = imported_modules(module, "example")
    result.clear()
    assert imported_modules(module, "example") == {"example.second"}


def test_identical_index_generation_is_reused_and_corruption_repaired(tmp_path):
    from pathlib import Path

    import numpy as np

    from goes_natural_science_kg.corpus.chunk import chunk_document
    from goes_natural_science_kg.corpus.index import build_index, load_index
    from goes_natural_science_kg.schemas.ingestion import ParsedDocument

    source = Path(__file__).parents[1] / "golden/pdfs/el-salvador-science-2026.pdf.expected.json"
    chunks = chunk_document(ParsedDocument.model_validate_json(source.read_bytes()))
    vectors = np.ones((len(chunks), 3))
    digest = build_index(tmp_path, chunks, vectors)
    files = tuple(p for p in tmp_path.rglob("*") if p.is_file())
    before = {p: p.stat().st_mtime_ns for p in files}
    assert build_index(tmp_path, chunks, vectors) == digest
    assert {p: p.stat().st_mtime_ns for p in files} == before
    target = tmp_path / digest / "vectors.npy"
    target.write_bytes(b"corrupted cache")
    assert build_index(tmp_path, chunks, vectors) == digest
    restored, matrix = load_index(tmp_path)
    assert len(restored) == len(chunks) and np.isfinite(matrix).all()
