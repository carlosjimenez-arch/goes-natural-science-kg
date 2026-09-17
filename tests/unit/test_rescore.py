# tests/unit/test_rescore.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify the revised decomposition score separates node matching from dependency order.
from pathlib import Path

import pytest

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.metrics import decomposition_metrics
from goes_natural_science_kg.eval.rescore import (
    design_quality,
    reference_agreement,
    revised_pass,
)
from goes_natural_science_kg.schemas.orchestration import Decomposition
from goes_natural_science_kg.schemas.prompt_evaluation import (
    ReferenceReview,
    SemanticMatch,
)

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/golden/prompt-evaluation/cases.jsonl"


def case(case_id: str = "length"):
    return next(c for c in load_cases(CASES) if c.id == case_id)


def renamed(expected: Decomposition, rename: dict[str, str]) -> Decomposition:
    """The same decomposition under different editorial keys and wording."""
    data = expected.model_dump(mode="json")
    for micro in data["micros"]:
        micro["key"] = rename.get(micro["key"], micro["key"])
        for prerequisite in micro["prerequisites"]:
            prerequisite["id"] = rename.get(prerequisite["id"], prerequisite["id"])
    return Decomposition.model_validate(data)


def review_for(candidate: Decomposition, pairs: dict[str, str]) -> ReferenceReview:
    return ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=left, expected_key=right, equivalent=True, rationale="same product"
            )
            for left, right in pairs.items()
        ),
        support=(),
    )


def test_one_unmatched_node_no_longer_zeroes_dependency_metrics():
    target = case()
    rename = {"length-1": "ruler-parts", "length-2": "length-2", "length-3": "length-3"}
    candidate = renamed(target.expected, rename)
    # The reviewer accepts two of three nodes; the renamed first step is judged different.
    partial = review_for(candidate, {"length-2": "length-2", "length-3": "length-3"})
    frozen = decomposition_metrics(target, candidate, partial)
    revised = reference_agreement(target, candidate, partial)
    # The frozen metric charges the candidate for the edge that touches the unmatched node,
    # and its pass rule demands exactly 1.0, so this candidate fails on an unreachable edge.
    assert frozen["prerequisite_recall"] == 0.5 and frozen["prerequisite_precision"] == 0.5
    # Between the two nodes both sides matched, the single shared edge is reproduced exactly.
    assert revised.node_match_rate == pytest.approx(2 / 3)
    assert revised.edge_recall_matched == 1.0 and revised.edge_precision_matched == 1.0
    assert revised.comparable_expected_edges == 1


def test_full_agreement_scores_one_and_reordered_dependencies_are_caught():
    target = case()
    candidate = target.expected
    full = review_for(candidate, {m.key: m.key for m in candidate.micros})
    agreement = reference_agreement(target, candidate, full)
    assert agreement.node_match_rate == 1.0
    assert agreement.edge_recall_matched == 1.0 and agreement.edge_precision_matched == 1.0
    data = candidate.model_dump(mode="json")
    for micro in data["micros"]:
        micro["prerequisites"] = []
    stripped = Decomposition.model_validate(data)
    dropped = reference_agreement(
        target, stripped, review_for(stripped, {m.key: m.key for m in stripped.micros})
    )
    assert dropped.node_match_rate == 1.0
    assert dropped.edge_recall_matched == 0.0 and dropped.edge_precision_matched is None


def test_design_quality_is_reference_free_and_catches_inflated_labels():
    target = case()
    good = design_quality(target, target.expected)
    assert good.has_reasoning_step and good.monotone_cognitive_chain
    assert good.grade_band_contains_grade and good.minutes_within_assignment
    assert good.citations_resolve and not good.structural_errors
    assert good.distinct_cognitive_domains == 3
    data = target.expected.model_dump(mode="json")
    # A reasoning label on the first step and knowing on the last inverts the progression.
    data["micros"][0]["cognitive_domain"] = "reasoning"
    data["micros"][-1]["cognitive_domain"] = "knowing"
    inverted = design_quality(target, Decomposition.model_validate(data))
    assert not inverted.monotone_cognitive_chain


def test_revised_pass_requires_design_checks_and_most_reference_nodes():
    target = case()
    candidate = target.expected
    full = review_for(candidate, {m.key: m.key for m in candidate.micros})
    agreement = reference_agreement(target, candidate, full)
    design = design_quality(target, candidate)
    assert revised_pass(agreement, design)
    # Two of three matched nodes still passes; one of three does not.
    two = reference_agreement(
        target, candidate, review_for(candidate, {"length-2": "length-2", "length-3": "length-3"})
    )
    assert revised_pass(two, design)
    one = reference_agreement(target, candidate, review_for(candidate, {"length-3": "length-3"}))
    assert not revised_pass(one, design)
    broken = design.model_copy(update={"citations_resolve": False})
    assert not revised_pass(agreement, broken)
    flat = design.model_copy(update={"has_reasoning_step": False})
    assert not revised_pass(agreement, flat)
