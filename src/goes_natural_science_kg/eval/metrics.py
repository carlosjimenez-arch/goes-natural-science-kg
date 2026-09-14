# src/goes_natural_science_kg/eval/metrics.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from pathlib import Path
from typing import Any

from goes_natural_science_kg.agents.checks import (
    claims_for,
    curriculum_errors,
    evidence_errors,
    graph_errors,
    materialize,
    verified_quote_span,
)
from goes_natural_science_kg.eval.cases import curriculum_task, work_item
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    CurriculumProposal,
    Decomposition,
    OrchestrationSettings,
    Verdict,
)
from goes_natural_science_kg.schemas.prompt_evaluation import (
    AnnotatedCase,
    ReferenceReview,
)


def matched_keys(
    expected: Decomposition, candidate: Decomposition, review: ReferenceReview
) -> dict[str, str]:
    """Maximum cardinality bipartite matching; one generated node cannot cover many targets."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    left = sorted({m.key for m in candidate.micros})
    right = sorted({m.key for m in expected.micros})
    matrix = np.zeros((len(left), len(right)), dtype=int)
    for match in review.matches:
        if match.equivalent and match.candidate_key in left and match.expected_key in right:
            matrix[left.index(match.candidate_key), right.index(match.expected_key)] = 1
    rows, cols = linear_sum_assignment(-matrix)
    return {left[i]: right[j] for i, j in zip(rows, cols, strict=True) if matrix[i, j]}


def decomposition_metrics(
    case: AnnotatedCase, candidate: Decomposition, review: ReferenceReview
) -> dict[str, Any]:
    mapping = matched_keys(case.expected, candidate, review)
    expected = {(p.id, m.key, p.type.value) for m in case.expected.micros for p in m.prerequisites}
    actual = {
        (
            mapping.get(p.id, "unmatched:" + p.id),
            mapping.get(m.key, "unmatched:" + m.key),
            p.type.value,
        )
        for m in candidate.micros
        for p in m.prerequisites
    }
    correct = len(expected & actual)
    refs = [r for m in candidate.micros for r in m.source_refs]
    known = {p.chunk_id: p for p in case.evidence}
    bad = sum(r not in known for r in refs)
    supported = set()
    for check in review.support:
        packet = known.get(check.source_ref)
        node = next((m for m in candidate.micros if m.key == check.candidate_key), None)
        if (
            packet is None
            or node is None
            or check.source_ref not in node.source_refs
            or not check.supported
        ):
            continue
        anchor = next((a for a in packet.anchors if a.paragraph_id == check.paragraph_id), None)
        if (
            anchor
            and verified_quote_span(anchor.text, check.quote) is not None
            and not node.misconceptions
        ):
            supported.add(node.key)
    errors = []
    try:
        errors = list(graph_errors(materialize(work_item(case), candidate), {case.skill.skill.id}))
    except ValueError as error:
        errors = [str(error)]
    return {
        "coverage": len(mapping) / len(case.expected.micros),
        "prerequisite_precision": correct / len(actual) if actual else None,
        "prerequisite_recall": correct / len(expected) if expected else None,
        "invalid_reference_rate": bad / len(refs) if refs else 1.0,
        "unsupported_claim_rate": 1 - len(supported) / len(candidate.micros),
        "notes": tuple(errors),
    }


def score_output(
    case: AnnotatedCase, role: str, output: Any, review: ReferenceReview | None, negative: bool
) -> dict[str, Any]:
    if role == "decomposition":
        if not isinstance(output, Decomposition) or review is None:
            raise ValueError("missing decomposition or independent reference review")
        metrics = decomposition_metrics(case, output, review)
        metrics["passed"] = (
            metrics["coverage"] >= 0.8
            and metrics["prerequisite_recall"] == 1.0
            and metrics["prerequisite_precision"] == 1.0
            and metrics["invalid_reference_rate"] == 0
            and metrics["unsupported_claim_rate"] == 0
            and not metrics["notes"]
        )
        return metrics
    if role == "curricularization":
        if not isinstance(output, CurriculumProposal):
            raise ValueError("wrong curriculum schema")
        if review is None:
            raise ValueError("missing independent curriculum evidence review")
        errors = curriculum_errors(curriculum_task(case), output)
        expected = {m.skill.id for m in curriculum_task(case).micros}
        actual = {i for u in output.units for i in u.micro_skill_ids}
        known = {p.chunk_id for p in case.evidence}
        invalid = (
            sum(r not in known for r in output.source_refs) / len(output.source_refs)
            if output.source_refs
            else 1.0
        )
        unsupported = curriculum_support_rate(case, output, review)
        return {
            "coverage": len(expected & actual) / len(expected),
            "invalid_reference_rate": invalid,
            "unsupported_claim_rate": unsupported,
            "passed": not errors and invalid == 0 and unsupported == 0,
            "notes": errors,
        }
    if not isinstance(output, Verdict) or output.role.value != role:
        raise ValueError("wrong judge schema or role")
    predicted = output.passed and output.score >= 0.8
    correct = predicted != negative
    judge_errors = []
    if role == "evidence_judge":
        from goes_natural_science_kg.eval.cases import judge_candidate

        candidate = judge_candidate(case, role, negative)
        rt = AgentRuntime(
            request={"skills": [case.skill], "budget_minutes": {case.skill.suggested_grade: 9600}},
            settings=OrchestrationSettings(project="offline", seed=0),
            evidence=case.evidence,
            cache=Path("."),
            prompts=Path("."),
        )
        if predicted:
            judge_errors = list(
                evidence_errors(
                    rt, claims_for(candidate, materialize(work_item(case), candidate)), output
                )
            )
        elif not output.claims:
            judge_errors = ["evidence rejection lacks claim inspection"]
    return {
        "judge_correct": correct,
        "passed": correct and not judge_errors,
        "notes": tuple(judge_errors),
    }


def curriculum_support_rate(
    case: AnnotatedCase, output: CurriculumProposal, review: ReferenceReview
) -> float:
    supported = set()
    for check in review.support:
        packet = next((p for p in case.evidence if p.chunk_id == check.source_ref), None)
        if packet is None or check.source_ref not in output.source_refs or not check.supported:
            continue
        anchor = next((a for a in packet.anchors if a.paragraph_id == check.paragraph_id), None)
        if anchor and verified_quote_span(anchor.text, check.quote) is not None:
            supported.add(check.candidate_key)
    return (
        1 - sum(u.key in supported for u in output.units) / len(output.units)
        if output.units
        else 1.0
    )
