# tests/benchmarks/test_release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure proposal schema preparation and stable identity transformation.
from goes_natural_science_kg.agents.release import transport_schema
from goes_natural_science_kg.curriculum.release import entity_identity
from goes_natural_science_kg.schemas.release import UnitDesign


def test_release_provider_grammar(benchmark):
    schema = UnitDesign.model_json_schema()
    assert benchmark(transport_schema, schema)["type"] == "object"


def test_release_bounded_identity(benchmark):
    assert (
        len(benchmark(entity_identity, "grade-two-unit-one", "observable-operation-" * 3).key) <= 80
    )


def test_release_graph_compilation_and_checks(benchmark):
    import json
    from pathlib import Path

    from goes_natural_science_kg.agents.release import design_errors
    from goes_natural_science_kg.curriculum.release import compile_proposal
    from goes_natural_science_kg.schemas.release import (
        ReleaseUnitResult,
        UnitAssignment,
        UnitDesign,
    )

    raw = json.loads(
        (Path(__file__).parents[1] / "golden/release/contract-fixture.json").read_text()
    )
    assignment, design = (
        UnitAssignment.model_validate(raw["assignment"]),
        UnitDesign.model_validate(raw["design"]),
    )
    assert not design_errors(design, assignment)
    result = ReleaseUnitResult(
        assignment=assignment,
        initial=design,
        final=design,
        panels=(),
        revisions=0,
        status="needs_human_review",
        request_ids=(),
    )
    data, _, _ = benchmark(compile_proposal, (result,), (), "benchmark")
    assert len(data.graph.micro_skills) == 8


def test_source_quote_location(benchmark):
    from goes_natural_science_kg.corpus.quotation import locate_quote

    text = "Measured\n observations\t support\n bounded claims."
    start, end = benchmark(locate_quote, text, "observations support bounded claims.")
    assert " ".join(text[start:end].split()) == "observations support bounded claims."


def test_fixed_grade_schedule_hint(benchmark):
    from pathlib import Path

    from goes_natural_science_kg.curriculum.hints import schedule_hint
    from goes_natural_science_kg.schemas.sequencing import SequencingInput, SequencingSettings

    root = Path("data/processed/proposals/2026-09-17/v5")
    data = SequencingInput.model_validate_json((root / "skills-map.json").read_bytes())
    settings = SequencingSettings.model_validate_json((root / "settings.json").read_bytes())
    hints = benchmark(schedule_hint, data, settings)
    assert len(hints) > 300
    assert sum(h.end - h.start for h in hints) == 5 * 9600
