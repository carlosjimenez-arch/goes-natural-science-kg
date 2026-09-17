# tests/benchmarks/test_rescore_benchmarks.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure the revised decomposition score on real recorded decompositions.
from pathlib import Path

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.rescore import (
    design_quality,
    reference_agreement,
    revised_pass,
)
from goes_natural_science_kg.schemas.prompt_evaluation import ReferenceReview, SemanticMatch

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/golden/prompt-evaluation/cases.jsonl"


def evaluation_cases():
    return [c for c in load_cases(CASES) if c.split == "evaluation"]


def full_review(case) -> ReferenceReview:
    return ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=m.key, expected_key=m.key, equivalent=True, rationale="identical"
            )
            for m in case.expected.micros
        ),
        support=(),
    )


def test_rescore_forty_recorded_cases(benchmark):
    cases = evaluation_cases()
    reviews = {c.id: full_review(c) for c in cases}

    def rescore_all() -> int:
        passed = 0
        for case in cases:
            agreement = reference_agreement(case, case.expected, reviews[case.id])
            design = design_quality(case, case.expected)
            passed += revised_pass(agreement, design)
        return passed

    assert benchmark(rescore_all) == len(cases)


def test_reference_agreement_single_case(benchmark):
    case = evaluation_cases()[0]
    review = full_review(case)
    benchmark(reference_agreement, case, case.expected, review)
