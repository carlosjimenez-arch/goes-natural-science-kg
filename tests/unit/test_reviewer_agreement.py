# tests/unit/test_reviewer_agreement.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify the agreement statistics used to judge the reviewers that gate decomposition scores.
import pytest

from goes_natural_science_kg.eval.reviewer_agreement import (
    cohen_kappa,
    krippendorff_alpha_nominal,
    pair_labels,
)
from goes_natural_science_kg.schemas.prompt_evaluation import ReferenceReview, SemanticMatch
from goes_natural_science_kg.schemas.reviewer_agreement import ReviewerArm, ReviewerProbePlan


def test_cohen_kappa_matches_hand_computed_values():
    perfect = [True, False, True, False]
    observed, kappa, counts = cohen_kappa(perfect, perfect)
    assert observed == 1.0 and kappa == 1.0 and counts[True, True] == 2
    # Chance-level agreement on a balanced pair of raters scores zero.
    a = [True, True, False, False]
    b = [True, False, True, False]
    observed, kappa, counts = cohen_kappa(a, b)
    assert observed == 0.5 and kappa == pytest.approx(0.0)
    assert counts[True, False] == 1 and counts[False, True] == 1
    # High raw agreement on a rare label still yields a modest kappa.
    rare_a = [True] + [False] * 19
    rare_b = [False] * 20
    observed, kappa, _ = cohen_kappa(rare_a, rare_b)
    assert observed == 0.95 and kappa == 0.0


def test_krippendorff_alpha_handles_three_raters_and_missing_labels():
    agreeing = [[True, True, True], [False, False, False]] * 5
    assert krippendorff_alpha_nominal(agreeing) == pytest.approx(1.0)
    mixed = [[True, False, True], [False, True, False]] * 5
    assert krippendorff_alpha_nominal(mixed) < 0.5
    sparse = [[True, None, True], [False, None, False]] * 5
    assert krippendorff_alpha_nominal(sparse) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal([[True, None, None]]) is None
    assert krippendorff_alpha_nominal([]) is None


def test_pair_labels_cover_the_full_grid_and_default_to_not_equivalent():
    from pathlib import Path

    from goes_natural_science_kg.eval.cases import load_cases

    root = Path(__file__).resolve().parents[2]
    case = next(
        c
        for c in load_cases(root / "tests/golden/prompt-evaluation/cases.jsonl")
        if c.id == "length"
    )
    review = ReferenceReview(
        matches=(
            SemanticMatch(
                candidate_key="length-1",
                expected_key="length-1",
                equivalent=True,
                rationale="same",
            ),
            SemanticMatch(
                candidate_key="length-2",
                expected_key="length-2",
                equivalent=False,
                rationale="different product",
            ),
        ),
        support=(),
    )
    labels = pair_labels("arm", 0, {"length": case.expected}, {"length": case}, {"length": review})
    assert len(labels) == 9
    assert sum(label.equivalent for label in labels) == 1
    # A pair the reviewer never mentioned is recorded as not equivalent, not as missing.
    unmentioned = next(
        label
        for label in labels
        if label.candidate_key == "length-3" and label.expected_key == "length-1"
    )
    assert unmentioned.equivalent is False


def reviewer(key: str, recorded: bool = False) -> ReviewerArm:
    return ReviewerArm(key=key, model="gemini-2.5-pro", recorded=recorded)


def test_probe_plan_requires_exactly_one_recorded_reviewer():
    with pytest.raises(ValueError, match="exactly one reviewer replays"):
        ReviewerProbePlan(
            name="probe",
            project="p",
            generator_arms=("a",),
            reviewers=(reviewer("x"), reviewer("y")),
            reference_reviewer="x",
        )
    with pytest.raises(ValueError, match="reference reviewer"):
        ReviewerProbePlan(
            name="probe",
            project="p",
            generator_arms=("a",),
            reviewers=(reviewer("x", True), reviewer("y")),
            reference_reviewer="z",
        )
    plan = ReviewerProbePlan(
        name="probe",
        project="p",
        generator_arms=("a",),
        reviewers=(reviewer("x", True), reviewer("y")),
        reference_reviewer="x",
    )
    assert plan.replicates == 3 and plan.prompt_id == "reference-review"
