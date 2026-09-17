# src/goes_natural_science_kg/eval/rescore.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Re-read recorded decomposition responses under a revised, less brittle score.
"""Reads only recorded observations: no provider call, no change to the frozen scorer."""

from __future__ import annotations

from pathlib import Path
from statistics import mean

from goes_natural_science_kg.agents.checks import graph_errors, materialize
from goes_natural_science_kg.eval.cases import load_cases, work_item
from goes_natural_science_kg.eval.experiment import read_observation
from goes_natural_science_kg.eval.metrics import matched_keys
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.orchestration import Decomposition
from goes_natural_science_kg.schemas.prompt_evaluation import (
    AnnotatedCase,
    BatchOutput,
    FollowUpCell,
    ReferenceReview,
)
from goes_natural_science_kg.schemas.rescoring import (
    DesignQuality,
    ReferenceAgreement,
    RescoreArmSummary,
    RescoreReport,
    RescoreSliceRow,
    RevisedMetric,
)
from goes_natural_science_kg.schemas.skills import CognitiveDomain

COGNITIVE_ORDER = {
    CognitiveDomain.KNOWING: 0,
    CognitiveDomain.APPLYING: 1,
    CognitiveDomain.REASONING: 2,
}
MINIMUM_SLICE_CASES = 15
# Declared policy for the revised rule; see decision 0014. Not an empirically optimal cut.
MINIMUM_NODE_MATCH = 0.6
MINIMUM_EDGE_RECALL = 0.5


def recorded_attempt(
    cell: FollowUpCell, observations: Path, case_id: str, revision: int
) -> tuple[Decomposition | None, ReferenceReview | None]:
    """Recover the candidate and its blind reference review for one case at one revision."""
    candidate: Decomposition | None = None
    review: ReferenceReview | None = None
    for key in cell.observations:
        record = read_observation(observations, key)
        if record.request["revision"] != revision or record.response is None:
            continue
        if case_id not in {c["case_id"] for c in record.request["input"]["cases"]}:
            continue
        if record.request["prompt_id"] == "reference-review":
            try:
                batch = BatchOutput[ReferenceReview].model_validate_json(record.response)
            except ValueError:
                continue
            review = next((r.output for r in batch.results if r.case_id == case_id), None)
        else:
            try:
                decoded = BatchOutput[Decomposition].model_validate_json(record.response)
            except ValueError:
                continue
            candidate = next((r.output for r in decoded.results if r.case_id == case_id), None)
    return candidate, review


def reference_agreement(
    case: AnnotatedCase, candidate: Decomposition, review: ReferenceReview
) -> ReferenceAgreement:
    """Edge metrics are computed only between nodes both sides agree exist.

    The frozen scorer maps an unmatched node to a sentinel, so a single naming
    disagreement zeroes precision and recall. Restricting to matched endpoints
    separates "named the same steps" from "ordered the steps the same way".
    """
    mapping = matched_keys(case.expected, candidate, review)
    covered = set(mapping.values())
    expected_edges = {
        (p.id, m.key, p.type.value)
        for m in case.expected.micros
        for p in m.prerequisites
        if p.id in covered and m.key in covered
    }
    candidate_edges = {
        (mapping[p.id], mapping[m.key], p.type.value)
        for m in candidate.micros
        for p in m.prerequisites
        if p.id in mapping and m.key in mapping
    }
    hits = len(expected_edges & candidate_edges)
    return ReferenceAgreement(
        node_match_rate=len(mapping) / len(case.expected.micros),
        matched_nodes=len(mapping),
        expected_nodes=len(case.expected.micros),
        candidate_nodes=len(candidate.micros),
        edge_precision_matched=hits / len(candidate_edges) if candidate_edges else None,
        edge_recall_matched=hits / len(expected_edges) if expected_edges else None,
        comparable_expected_edges=len(expected_edges),
        comparable_candidate_edges=len(candidate_edges),
    )


def design_quality(case: AnnotatedCase, candidate: Decomposition) -> DesignQuality:
    """Checks that hold whatever the draft reference says; all are contract-level facts."""
    task = work_item(case)
    grade = case.skill.suggested_grade
    levels = {m.key: COGNITIVE_ORDER[m.cognitive_domain] for m in candidate.micros}
    monotone = all(
        levels[m.key] >= levels[p.id]
        for m in candidate.micros
        for p in m.prerequisites
        if p.id in levels
    )
    known = {p.chunk_id for p in case.evidence}
    try:
        errors = tuple(graph_errors(materialize(task, candidate), {case.skill.skill.id}))
    except ValueError as error:
        errors = (str(error),)
    return DesignQuality(
        single_observable_verb=mean(len(m.observable_verb.split()) == 1 for m in candidate.micros),
        distinct_cognitive_domains=len({m.cognitive_domain for m in candidate.micros}),
        has_reasoning_step=any(
            m.cognitive_domain == CognitiveDomain.REASONING for m in candidate.micros
        ),
        monotone_cognitive_chain=monotone,
        grade_band_contains_grade=all(
            m.grade_band.minimum <= grade <= m.grade_band.maximum for m in candidate.micros
        ),
        minutes_within_assignment=sum(m.estimated_minutes for m in candidate.micros)
        <= task.assignments[0].minutes,
        citations_resolve=all(r in known for m in candidate.micros for r in m.source_refs),
        structural_errors=errors,
    )


def revised_pass(agreement: ReferenceAgreement, design: DesignQuality) -> bool:
    """Declared rule: the candidate must name most reference steps, order the shared ones
    correctly, and satisfy every deterministic design check. It never requires an exact
    three-of-three match with one agent-authored decomposition."""
    if design.structural_errors or not design.citations_resolve:
        return False
    if not (design.has_reasoning_step and design.monotone_cognitive_chain):
        return False
    if not (design.grade_band_contains_grade and design.minutes_within_assignment):
        return False
    if agreement.node_match_rate < MINIMUM_NODE_MATCH:
        return False
    recall = agreement.edge_recall_matched
    return recall is None or recall >= MINIMUM_EDGE_RECALL


def rescore_cells(
    cells: list[FollowUpCell], cases: dict[str, AnnotatedCase], observations: Path
) -> list[tuple[str, RevisedMetric]]:
    """One revised metric per arm, case and replicate, taken at each case's final revision."""
    rows = []
    for cell in cells:
        for case_id in cell.case_ids:
            history = [m for m in cell.metrics if m.case_id == case_id]
            final = history[-1]
            case = cases[case_id]
            candidate, review = recorded_attempt(cell, observations, case_id, final.revisions)
            agreement = (
                reference_agreement(case, candidate, review)
                if candidate is not None and review is not None
                else None
            )
            design = design_quality(case, candidate) if candidate is not None else None
            rows.append(
                (
                    cell.arm,
                    RevisedMetric(
                        arm=cell.arm,
                        case_id=case_id,
                        grade=case.skill.suggested_grade,
                        domain=case.skill.domain.value,
                        revision=final.revisions,
                        schema_valid=candidate is not None,
                        agreement=agreement,
                        design=design,
                        frozen_passed=final.passed,
                        revised_passed=bool(
                            agreement and design and revised_pass(agreement, design)
                        ),
                    ),
                )
            )
    return rows


def _slice_rows(metrics: list[RevisedMetric]) -> tuple[RescoreSliceRow, ...]:
    rows = []
    for dimension in ("grade", "domain"):
        groups: dict[str, list[RevisedMetric]] = {}
        for metric in metrics:
            key = str(metric.grade) if dimension == "grade" else metric.domain
            groups.setdefault(key, []).append(metric)
        for key, items in sorted(groups.items()):
            matches = [m.agreement.node_match_rate for m in items if m.agreement]
            recalls = [
                m.agreement.edge_recall_matched
                for m in items
                if m.agreement and m.agreement.edge_recall_matched is not None
            ]
            rows.append(
                RescoreSliceRow(
                    dimension=dimension,
                    key=key,
                    cases=len({m.case_id for m in items}),
                    frozen_pass_rate=mean(m.frozen_passed for m in items),
                    revised_pass_rate=mean(m.revised_passed for m in items),
                    node_match_rate=mean(matches) if matches else None,
                    edge_recall_matched=mean(recalls) if recalls else None,
                    insufficient=len({m.case_id for m in items}) < MINIMUM_SLICE_CASES,
                )
            )
    return tuple(rows)


def make_rescore_report(
    cells_path: Path,
    cases_path: Path,
    observations: Path,
    source_report: Path,
) -> RescoreReport:
    cells = [
        FollowUpCell.model_validate_json(p.read_bytes()) for p in sorted(cells_path.glob("*.json"))
    ]
    cases = {c.id: c for c in load_cases(cases_path)}
    rows = rescore_cells(cells, cases, observations)
    specs = {c.arm: c.spec for c in cells}
    summaries = []
    for arm in sorted(specs):
        metrics = [m for key, m in rows if key == arm]
        scored = [m for m in metrics if m.agreement and m.design]
        matches = [m.agreement.node_match_rate for m in scored if m.agreement]
        precisions = [
            m.agreement.edge_precision_matched
            for m in scored
            if m.agreement and m.agreement.edge_precision_matched is not None
        ]
        recalls = [
            m.agreement.edge_recall_matched
            for m in scored
            if m.agreement and m.agreement.edge_recall_matched is not None
        ]
        verbs = [m.design.single_observable_verb for m in scored if m.design]
        slices = _slice_rows(metrics)
        graded = [r for r in slices if r.dimension == "grade"]
        summaries.append(
            RescoreArmSummary(
                arm=arm,
                generator_model=specs[arm].generator_model,
                prompt_id=specs[arm].prompt.prompt_id,
                case_replicates=len(metrics),
                frozen_pass_rate=mean(m.frozen_passed for m in metrics),
                revised_pass_rate=mean(m.revised_passed for m in metrics),
                node_match_rate=mean(matches) if matches else None,
                edge_precision_matched=mean(precisions) if precisions else None,
                edge_recall_matched=mean(recalls) if recalls else None,
                single_observable_verb=mean(verbs) if verbs else None,
                monotone_cognitive_chain_rate=mean(
                    m.design.monotone_cognitive_chain for m in scored if m.design
                )
                if scored
                else 0.0,
                reasoning_step_rate=mean(m.design.has_reasoning_step for m in scored if m.design)
                if scored
                else 0.0,
                citations_resolve_rate=mean(m.design.citations_resolve for m in scored if m.design)
                if scored
                else 0.0,
                structural_error_rate=mean(
                    bool(m.design.structural_errors) for m in scored if m.design
                )
                if scored
                else 0.0,
                slices=slices,
                worst_grade_slice=min(graded, key=lambda r: (r.revised_pass_rate, r.key))
                if graded
                else None,
            )
        )
    ordered = sorted(rows, key=lambda row: (row[0], row[1].case_id, row[1].revision))
    return RescoreReport(
        source_report_sha256=content_hash(source_report.read_text()),
        dataset_sha256=content_hash([c.model_dump(mode="json") for c in cases.values()]),
        rescored_case_replicates=len(ordered),
        arms=tuple(summaries),
        metrics=tuple(m for _, m in ordered),
        limitations=(
            "The revised rule is declared policy, not a validated threshold; it was written after reading first-experiment failures and before computing these numbers.",
            "Reference agreement still depends on an agent-authored decomposition and on a model reviewer's equivalence judgments.",
            "Design-quality checks are contract-level facts about the output, not evidence that a task teaches anything.",
            "Grade and domain slices hold five to eight distinct cases and are flagged insufficient.",
            "No provider request was made: every number re-reads recorded responses from the follow-up experiment.",
        ),
    )
