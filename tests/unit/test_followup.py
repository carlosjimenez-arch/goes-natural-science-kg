# tests/unit/test_followup.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify follow-up experiment contracts, pricing, slices and paired bootstrap.
from pathlib import Path
from typing import get_args

import pytest

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.experiment import MODEL_LIST_PRICES, estimate_cost
from goes_natural_science_kg.eval.reporting import paired_bootstrap, slice_rows
from goes_natural_science_kg.schemas.prompt_evaluation import (
    EvaluationMetric,
    ExperimentArm,
    FollowUpPlan,
    GeneratorModel,
    PromptVariantRef,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/golden/prompt-evaluation/cases.jsonl"


def test_every_generator_model_has_a_list_price_and_legacy_rates_are_unchanged():
    assert set(get_args(GeneratorModel)) <= set(MODEL_LIST_PRICES)
    # Legacy experiment rates (decision 0006) must reproduce byte-identical cost estimates.
    assert estimate_cost("gemini-2.5-flash", 100_000, 10_000, 5_000, 0) == pytest.approx(
        0.1 * 0.30 + 0.015 * 2.5
    )
    assert estimate_cost("gemini-2.5-pro", 100_000, 0, 0, 50_000) == pytest.approx(
        0.05 * 1.25 + 0.05 * 0.125
    )
    assert estimate_cost("gemini-2.5-flash", None, 10, None, None) is None
    with pytest.raises(ValueError, match="no list price"):
        estimate_cost("unknown-model", 10, 10, None, None)
    with pytest.raises(ValueError, match="pricing tier"):
        estimate_cost("gemini-2.5-flash", 300_000, 10, None, None)


def arm(key: str, prompt_id: str = "decomposition-few-shot") -> ExperimentArm:
    return ExperimentArm(key=key, prompt=PromptVariantRef(prompt_id=prompt_id))


def test_followup_plan_requires_baseline_and_unique_arms():
    with pytest.raises(ValueError, match="baseline arm"):
        FollowUpPlan(
            name="plan",
            role="decomposition",
            project="p",
            arms=(arm("a"), arm("b")),
            baseline_arm="c",
        )
    with pytest.raises(ValueError, match="duplicate arm"):
        FollowUpPlan(
            name="plan",
            role="decomposition",
            project="p",
            arms=(arm("a"), arm("a")),
            baseline_arm="a",
        )
    plan = FollowUpPlan(
        name="plan",
        role="decomposition",
        project="p",
        arms=(arm("a"), arm("b", "decomposition-anchored-few-shot")),
        baseline_arm="a",
        tuning_case_ids=("balance",),
    )
    assert plan.arms[1].location == "us-central1" and plan.bootstrap_resamples == 2000


def synthetic_final(passed_ids: set[str], replicates: int = 3) -> list[EvaluationMetric]:
    cases = [c for c in load_cases(CASES) if c.split == "evaluation"]
    return [
        EvaluationMetric(
            case_id=c.id,
            role="decomposition",
            technique="few_shot",
            replicate=r,
            schema_valid=True,
            coverage=1.0 if c.id in passed_ids else 0.5,
            prerequisite_precision=None,
            prerequisite_recall=1.0 if c.id in passed_ids else 0.0,
            invalid_reference_rate=0.0,
            unsupported_claim_rate=0.0,
            judge_correct=None,
            passed=c.id in passed_ids,
            revisions=0,
            censored=c.id not in passed_ids,
        )
        for c in cases
        for r in range(replicates)
    ]


def test_slice_rows_expose_grade_domain_and_split_with_insufficient_flags():
    cases = {c.id: c for c in load_cases(CASES)}
    grade_two = {c.id for c in cases.values() if c.skill.suggested_grade == 2}
    rows = slice_rows(synthetic_final(grade_two), cases, tuning={"balance", "day-night"})
    by_dim = {(r.dimension, r.key): r for r in rows}
    assert by_dim[("grade", "2")].final_pass_rate == 1.0
    assert by_dim[("grade", "6")].final_pass_rate == 0.0
    assert all(r.insufficient for r in rows if r.dimension == "grade")
    assert by_dim[("split", "tuning")].cases == 2
    assert by_dim[("split", "holdout")].cases == 38
    assert not by_dim[("split", "holdout")].insufficient
    assert {r.key for r in rows if r.dimension == "domain"} == {
        "physical-science",
        "life-science",
        "earth-space-science",
    }
    assert all(r.case_replicates == 3 * r.cases for r in rows)


def test_paired_bootstrap_separates_real_improvement_from_noise():
    cases = [c.id for c in load_cases(CASES) if c.split == "evaluation"]
    baseline = synthetic_final(set(cases[:10]))
    better = synthetic_final(set(cases[:30]))
    delta, low, high = paired_bootstrap(better, baseline, resamples=500, seed=0)
    assert delta == pytest.approx(0.5) and low > 0 and high <= 1.0
    same_delta, same_low, same_high = paired_bootstrap(baseline, baseline, resamples=500, seed=0)
    assert same_delta == 0 and same_low <= 0 <= same_high
    # Deterministic under the same seed.
    assert paired_bootstrap(better, baseline, resamples=500, seed=0) == (delta, low, high)
