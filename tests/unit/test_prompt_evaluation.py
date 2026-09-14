# tests/unit/test_prompt_evaluation.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Enforce versioned prompts, disjoint evidence examples and bounded evaluation records.
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_natural_science_kg.eval.cases import ROLES, load_cases
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.prompt_evaluation import EvaluationCell, EvaluationObservation

ROOT = Path(__file__).resolve().parents[2]


def test_registry_variants_and_disjoint_examples():
    registry = load_registry(ROOT / "prompts")
    cases = load_cases(ROOT / "tests/golden/prompt-evaluation/cases.jsonl")
    held_out = {c.id for c in cases if c.split == "evaluation"}
    examples = {c.id for c in cases if c.split == "demonstration"}
    assert len(held_out) == 40 and len(examples) == 6 and not held_out & examples
    assert {
        g: sum(c.skill.suggested_grade == g for c in cases if c.id in held_out) for g in range(2, 7)
    } == dict.fromkeys(range(2, 7), 8)
    for role in ROLES:
        for technique in ("cot", "few_shot", "cot_few_shot"):
            artifact = registry.get(
                role.replace("_", "-") + "-" + technique.replace("_", "-"), "1.0.0"
            )
            assert artifact.metadata.technique == technique
            assert not set(artifact.metadata.example_ids) & held_out
            if technique != "cot":
                assert set(artifact.metadata.example_ids) == examples
                assert "NEGATIVO" in artifact.body and "CORRECCIÓN" in artifact.body
                assert len(artifact.body) > 20000
            rendered = registry.render(
                artifact.metadata.id, "1.0.0", {"input": '{"literal":"{{input}}"}'}
            )
            assert '{"literal":"{{input}}"}' in rendered
    assert all(not a.legacy for a in registry.artifacts)


def test_unversioned_prompt_is_rejected(tmp_path):
    source = (ROOT / "prompts/variants/decomposition-cot.v1.prompt").read_text()
    (tmp_path / "bad.prompt").write_text(source.replace("version: 1.0.0\n", "", 1))
    with pytest.raises(ValidationError):
        load_registry(tmp_path)


def test_cell_revisions_are_bounded():
    with pytest.raises(ValidationError, match="missing case metrics"):
        EvaluationCell(
            id="a" * 64,
            case_ids=("case",),
            role="decomposition",
            technique="cot",
            replicate=0,
            observations=(),
            metrics=(),
        )


def test_observation_digest_rejects_tampering():
    request = {"seed": 0}
    with pytest.raises(ValidationError, match="request digest mismatch"):
        EvaluationObservation(
            request=request,
            request_sha256=content_hash({"seed": 1}),
            response=None,
            response_sha256=None,
            status="failed",
            error="transport unavailable",
            model_version=None,
            latency_seconds=1,
            input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            cached_input_tokens=None,
            estimated_usd=None,
        )


def test_authored_references_never_impersonate_human_review():
    cases = load_cases(ROOT / "tests/golden/prompt-evaluation/cases.jsonl")
    for case in cases:
        assert case.annotation_status == "agent_authored_pending_human_review"
        assert case.reviewer is None
        assert all(m.source_refs for m in case.expected.micros)
        assert set(r for m in case.expected.micros for r in m.source_refs) <= {
            p.chunk_id for p in case.evidence
        }
        with pytest.raises(ValidationError, match="human reviewer required"):
            type(case).model_validate(case.model_dump() | {"annotation_status": "human_reviewed"})


def test_matching_is_one_to_one_and_references_need_exact_quotes():
    from goes_natural_science_kg.eval.metrics import decomposition_metrics, matched_keys
    from goes_natural_science_kg.schemas.prompt_evaluation import (
        ReferenceReview,
        SemanticMatch,
        SourceSupport,
    )

    case = load_cases(ROOT / "tests/golden/prompt-evaluation/cases.jsonl")[0]
    ambiguous = ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=case.expected.micros[0].key,
                expected_key=m.key,
                equivalent=True,
                rationale="Synthetic ambiguity exercises matching, not a provider cassette.",
            )
            for m in case.expected.micros
        ),
        support=(),
    )
    assert len(matched_keys(case.expected, case.expected, ambiguous)) == 1
    packet = case.evidence[0]
    review = ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=m.key,
                expected_key=m.key,
                equivalent=True,
                rationale="Identity match for a domain-function test.",
            )
            for m in case.expected.micros
        ),
        support=tuple(
            SourceSupport(
                candidate_key=m.key,
                source_ref=packet.chunk_id,
                paragraph_id=packet.anchors[0].paragraph_id,
                quote="This quotation does not occur in the document.",
                supported=True,
                rationale="Deliberately invalid quote for integrity test.",
            )
            for m in case.expected.micros
        ),
    )
    metrics = decomposition_metrics(case, case.expected, review)
    assert metrics["coverage"] == 1 and metrics["prerequisite_precision"] == 1
    assert metrics["invalid_reference_rate"] == 0
    assert metrics["unsupported_claim_rate"] == 1


def test_prerequisite_omission_is_not_perfect_precision():
    from goes_natural_science_kg.eval.metrics import decomposition_metrics
    from goes_natural_science_kg.schemas.prompt_evaluation import ReferenceReview, SemanticMatch

    case = load_cases(ROOT / "tests/golden/prompt-evaluation/cases.jsonl")[0]
    candidate = case.expected.model_copy(
        update={
            "micros": tuple(
                m.model_copy(update={"prerequisites": ()}) for m in case.expected.micros
            )
        }
    )
    review = ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=m.key,
                expected_key=m.key,
                equivalent=True,
                rationale="Identity match.",
            )
            for m in case.expected.micros
        ),
        support=(),
    )
    metrics = decomposition_metrics(case, candidate, review)
    assert metrics["prerequisite_precision"] is None
    assert metrics["prerequisite_recall"] == 0


def test_fifth_attempt_and_uncensored_failure_are_rejected():
    from goes_natural_science_kg.schemas.prompt_evaluation import EvaluationMetric

    base = dict(
        case_id="case",
        role="decomposition",
        technique="cot",
        replicate=0,
        schema_valid=False,
        coverage=None,
        prerequisite_precision=None,
        prerequisite_recall=None,
        invalid_reference_rate=None,
        unsupported_claim_rate=None,
        judge_correct=None,
        passed=False,
    )
    metrics = tuple(EvaluationMetric(**base, revisions=i, censored=i == 4) for i in range(5))
    with pytest.raises(ValidationError, match="unbounded"):
        EvaluationCell(
            id="a" * 64,
            case_ids=("case",),
            role="decomposition",
            technique="cot",
            replicate=0,
            observations=(),
            metrics=metrics,
        )
    with pytest.raises(ValidationError, match="censored"):
        EvaluationCell(
            id="a" * 64,
            case_ids=("case",),
            role="decomposition",
            technique="cot",
            replicate=0,
            observations=(),
            metrics=metrics[:1],
        )


def test_high_variance_is_rejected_from_recorded_metric_scenarios():
    from goes_natural_science_kg.eval.reporting import summarize_variant

    folder = ROOT / "tests/golden/prompt-evaluation"
    measured = EvaluationCell.model_validate_json((folder / "replay-cell.json").read_bytes())
    cells = []
    for replica in range(3):
        metrics = []
        for original in measured.metrics:
            # A statistical counterexample, not an invented model response.
            for revision in range(4 if replica == 0 else 1):
                metrics.append(
                    original.model_copy(
                        update={
                            "replicate": replica,
                            "revisions": revision,
                            "passed": replica != 0,
                            "censored": replica == 0 and revision == 3,
                        }
                    )
                )
        cells.append(measured.model_copy(update={"replicate": replica, "metrics": tuple(metrics)}))
    summary = summarize_variant(cells, folder / "observations")
    assert summary.final_pass_sd > 0.5 and not summary.eligible
    assert any("standard deviation" in reason for reason in summary.rejection_reasons)


def test_production_selection_requires_identified_human_review():
    from goes_natural_science_kg.schemas.prompt_evaluation import EvaluationReport

    with pytest.raises(ValidationError, match="human review"):
        EvaluationReport(
            dataset_sha256="a" * 64,
            human_reviewed=False,
            complete=True,
            estimated_total_usd=0,
            missing_usage_requests=0,
            provider_requests=0,
            expected_cells=630,
            completed_cells=630,
            variants=(),
            provisional_choices={"decomposition": "cot"},
            production_choices={"decomposition": "cot"},
            limitations=(),
        )
