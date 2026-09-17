# src/goes_natural_science_kg/eval/reviewer_agreement.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure whether independent reviewer models agree on the equivalence judgments that gate scores.
"""Re-judges recorded revision-0 candidates; generation is never repeated."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from google import genai

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.experiment import evaluate_request, read_observation
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.orchestration import Decomposition
from goes_natural_science_kg.schemas.prompt_evaluation import (
    AnnotatedCase,
    BatchOutput,
    FollowUpCell,
    PromptRegistry,
    ReferenceReview,
    SemanticMatch,
)
from goes_natural_science_kg.schemas.reviewer_agreement import (
    PairLabel,
    PairwiseAgreement,
    ReviewerAgreementReport,
    ReviewerArm,
    ReviewerNodeMatch,
    ReviewerProbePlan,
)


def review_payload(
    cell: FollowUpCell, observations: Path, cases: dict[str, AnnotatedCase]
) -> tuple[dict[str, Any], dict[str, Decomposition]] | None:
    """Rebuild the exact revision-0 review request that the follow-up experiment sent."""
    generation = next(
        (
            record
            for key in cell.observations
            if (record := read_observation(observations, key)).request["revision"] == 0
            and record.request["prompt_id"] != "reference-review"
            and record.response is not None
        ),
        None,
    )
    if generation is None:
        return None
    try:
        batch = BatchOutput[Decomposition].model_validate_json(generation.response or "")
    except ValueError:
        return None
    outputs = {r.case_id: r.output for r in batch.results}
    if set(outputs) != set(cell.case_ids):
        return None
    return {
        "cases": [
            {
                "case_id": key,
                "expected": cases[key].expected.model_dump(mode="json"),
                "candidate": value.model_dump(mode="json"),
                "evidence": [p.model_dump(mode="json") for p in cases[key].evidence],
            }
            for key, value in outputs.items()
        ]
    }, outputs


def pair_labels(
    arm: str,
    replicate: int,
    outputs: dict[str, Decomposition],
    cases: dict[str, AnnotatedCase],
    reviews: dict[str, ReferenceReview],
) -> list[PairLabel]:
    """Label the full candidate-by-expected grid; a pair the reviewer did not assert is False."""
    labels = []
    for case_id, candidate in outputs.items():
        asserted = {
            (m.candidate_key, m.expected_key)
            for m in reviews.get(case_id, ReferenceReview(matches=(), support=())).matches
            if m.equivalent
        }
        for left in sorted({m.key for m in candidate.micros}):
            for right in sorted({m.key for m in cases[case_id].expected.micros}):
                labels.append(
                    PairLabel(
                        generator_arm=arm,
                        replicate=replicate,
                        case_id=case_id,
                        candidate_key=left,
                        expected_key=right,
                        equivalent=(left, right) in asserted,
                    )
                )
    return labels


def cohen_kappa(a: list[bool], b: list[bool]) -> tuple[float, float, Counter[tuple[bool, bool]]]:
    """Binary Cohen's kappa; returns observed agreement, kappa and the confusion counts."""
    counts = Counter(zip(a, b, strict=True))
    total = len(a)
    observed = (counts[True, True] + counts[False, False]) / total
    pa, pb = sum(a) / total, sum(b) / total
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = 0.0 if expected == 1 else (observed - expected) / (1 - expected)
    return observed, kappa, counts


def krippendorff_alpha_nominal(matrix: list[list[bool | None]]) -> float | None:
    """Nominal alpha over units by raters; units with fewer than two labels are dropped."""
    units = [[v for v in row if v is not None] for row in matrix]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    observed = 0.0
    pairs = 0
    counts: Counter[bool] = Counter()
    for unit in units:
        n = len(unit)
        for value in unit:
            counts[value] += 1
        for i in range(n):
            for j in range(n):
                if i != j:
                    observed += unit[i] != unit[j]
                    pairs += 1
    if not pairs:
        return None
    total = sum(counts.values())
    disagreement_observed = observed / pairs
    expected = 1 - sum(c * (c - 1) for c in counts.values()) / (total * (total - 1))
    return None if expected == 0 else float(1 - disagreement_observed / expected)


def recorded_reviews(cell: FollowUpCell, observations: Path) -> dict[str, ReferenceReview]:
    """The equivalence judgments the follow-up experiment actually scored at revision 0."""
    record = next(
        (
            found
            for key in cell.observations
            if (found := read_observation(observations, key)).request["revision"] == 0
            and found.request["prompt_id"] == "reference-review"
            and found.response is not None
        ),
        None,
    )
    if record is None:
        return {}
    try:
        batch = BatchOutput[ReferenceReview].model_validate_json(record.response or "")
    except ValueError:
        return {}
    return {r.case_id: r.output for r in batch.results}


async def run_reviewer_probe(
    plan: ReviewerProbePlan,
    cases_path: Path,
    prompts: Path,
    cells_path: Path,
    observations: Path,
    cache: Path,
    *,
    online: bool = False,
    credentials: Any = None,
) -> tuple[
    list[PairLabel],
    dict[str, list[PairLabel]],
    dict[str, int],
    list[Any],
    dict[str, Decomposition],
]:
    """Returns recorded labels, per-reviewer labels, failures, provider records and candidates."""
    cases = {c.id: c for c in load_cases(cases_path)}
    cells = [
        FollowUpCell.model_validate_json(p.read_bytes()) for p in sorted(cells_path.glob("*.json"))
    ]
    selected = [c for c in cells if c.arm in plan.generator_arms and c.replicate < plan.replicates]
    clients = (
        {
            location: genai.Client(
                vertexai=True,
                project=plan.project,
                location=location,
                credentials=credentials,
                http_options={"timeout": 180000, "retry_options": {"attempts": 3}},
            )
            for location in sorted({r.location for r in plan.reviewers if not r.recorded})
        }
        if online
        else None
    )
    limiter = asyncio.Semaphore(4)
    locks: dict[str, asyncio.Lock] = {}
    from goes_natural_science_kg.eval.registry import load_registry

    registry: PromptRegistry = load_registry(prompts)
    recorded_labels: list[PairLabel] = []
    fresh: dict[str, list[PairLabel]] = {r.key: [] for r in plan.reviewers if not r.recorded}
    failures: dict[str, int] = dict.fromkeys(fresh, 0)
    records: list[Any] = []

    async def judge(reviewer: ReviewerArm, payload: dict[str, Any], cell: FollowUpCell) -> None:
        record = await evaluate_request(
            clients.get(reviewer.location) if clients else None,
            limiter,
            cache,
            plan,
            registry,
            plan.prompt_id,
            payload,
            reviewer.model,
            BatchOutput[ReferenceReview],
            cell.replicate,
            0,
            locks,
            prompt_version=plan.prompt_version,
            location=reviewer.location,
        )
        records.append(record)
        try:
            batch = BatchOutput[ReferenceReview].model_validate_json(record.response or "")
        except ValueError:
            # A failed request is missing data, never a judgment that nothing is equivalent.
            failures[reviewer.key] += 1
            return
        fresh[reviewer.key].extend(
            pair_labels(
                cell.arm,
                cell.replicate,
                outputs_by_cell[cell.id],
                cases,
                {r.case_id: r.output for r in batch.results},
            )
        )

    outputs_by_cell: dict[str, dict[str, Decomposition]] = {}
    candidates: dict[str, Decomposition] = {}
    jobs = []
    for cell in selected:
        rebuilt = review_payload(cell, observations, cases)
        if rebuilt is None:
            continue
        payload, outputs = rebuilt
        outputs_by_cell[cell.id] = outputs
        reviews = recorded_reviews(cell, observations)
        recorded_labels.extend(pair_labels(cell.arm, cell.replicate, outputs, cases, reviews))
        for case_id, output in outputs.items():
            candidates[f"{cell.arm}|{cell.replicate}|{case_id}"] = output
        for reviewer in plan.reviewers:
            if not reviewer.recorded:
                jobs.append(judge(reviewer, payload, cell))
    try:
        await asyncio.gather(*jobs)
    finally:
        for client in (clients or {}).values():
            await client.aio.aclose()
    return recorded_labels, fresh, failures, records, candidates


def reviewer_effect(
    labels: list[PairLabel], cases: dict[str, AnnotatedCase], candidates: dict[str, Decomposition]
) -> tuple[float, float]:
    """Node match rate and the revised pass rate this reviewer's labels would produce.

    Design quality does not depend on the reviewer, so any difference in the pass rate comes
    only from the equivalence judgments. Units the reviewer did not label are skipped.
    """
    from goes_natural_science_kg.eval.rescore import (
        design_quality,
        reference_agreement,
        revised_pass,
    )

    grouped: dict[tuple[str, int, str], list[PairLabel]] = {}
    for label in labels:
        grouped.setdefault((label.generator_arm, label.replicate, label.case_id), []).append(label)
    rates: list[float] = []
    passes: list[bool] = []
    for (arm, replicate, case_id), items in grouped.items():
        candidate = candidates.get(f"{arm}|{replicate}|{case_id}")
        if candidate is None:
            continue
        review = ReferenceReview(
            matches=tuple(
                SemanticMatch(
                    candidate_key=i.candidate_key,
                    expected_key=i.expected_key,
                    equivalent=i.equivalent,
                    rationale="replayed label",
                )
                for i in items
            ),
            support=(),
        )
        agreement = reference_agreement(cases[case_id], candidate, review)
        rates.append(agreement.node_match_rate)
        passes.append(revised_pass(agreement, design_quality(cases[case_id], candidate)))
    if not rates:
        return 0.0, 0.0
    return float(np.mean(rates)), float(np.mean(passes))


def make_reviewer_report(
    plan: ReviewerProbePlan,
    cases_path: Path,
    recorded: list[PairLabel],
    fresh: dict[str, list[PairLabel]],
    failures: dict[str, int],
    records: list[Any],
    candidates: dict[str, Decomposition],
) -> ReviewerAgreementReport:
    cases = {c.id: c for c in load_cases(cases_path)}
    reference = next(r for r in plan.reviewers if r.recorded)
    all_labels = {reference.key: recorded, **fresh}

    def keyed(items: list[PairLabel]) -> dict[tuple[str, int, str, str, str], bool]:
        return {
            (i.generator_arm, i.replicate, i.case_id, i.candidate_key, i.expected_key): i.equivalent
            for i in items
        }

    indexed = {name: keyed(items) for name, items in all_labels.items()}
    shared = sorted(set.intersection(*(set(v) for v in indexed.values())))
    agreements = []
    names = sorted(indexed)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = [indexed[left][k] for k in shared]
            b = [indexed[right][k] for k in shared]
            observed, kappa, counts = cohen_kappa(a, b)
            agreements.append(
                PairwiseAgreement(
                    reviewer_a=left,
                    reviewer_b=right,
                    pairs=len(shared),
                    observed_agreement=observed,
                    cohen_kappa=kappa,
                    both_equivalent=counts[True, True],
                    only_a_equivalent=counts[True, False],
                    only_b_equivalent=counts[False, True],
                    neither_equivalent=counts[False, False],
                )
            )
    matrix: list[list[bool | None]] = [[indexed[n][k] for n in names] for k in shared]
    by_model = {r.key: r.model for r in plan.reviewers}
    effects = {name: reviewer_effect(all_labels[name], cases, candidates) for name in names}
    summaries = tuple(
        ReviewerNodeMatch(
            reviewer=name,
            model=by_model[name],
            node_match_rate=effects[name][0],
            implied_revised_pass_rate=effects[name][1],
            cases=len({(i.generator_arm, i.replicate, i.case_id) for i in all_labels[name]}),
            equivalent_pairs=sum(i.equivalent for i in all_labels[name]),
            judged_pairs=len(all_labels[name]),
            failed_requests=failures.get(name, 0),
        )
        for name in names
    )
    rates = [s.node_match_rate for s in summaries]
    implied = [s.implied_revised_pass_rate for s in summaries]
    return ReviewerAgreementReport(
        plan_sha256=content_hash(plan),
        dataset_sha256=content_hash([c.model_dump(mode="json") for c in cases.values()]),
        complete=all(f == 0 for f in failures.values()) and bool(shared),
        estimated_total_usd=sum(r.estimated_usd for r in records if r.estimated_usd is not None),
        provider_requests=len(records),
        missing_usage_requests=sum(r.estimated_usd is None for r in records),
        reviewers=summaries,
        agreements=tuple(agreements),
        krippendorff_alpha=krippendorff_alpha_nominal(matrix),
        node_match_rate_spread=max(rates) - min(rates),
        implied_pass_rate_spread=max(implied) - min(implied),
        limitations=(
            "Every rater is a model. Agreement between models bounds how much the score can be trusted; it is not human validation and cannot establish which reviewer is right.",
            "Raters share a prompt, a rubric and largely overlapping training data, so agreement is an optimistic upper bound on independence.",
            "Only revision-0 candidates are re-judged; later revisions react to the recorded reviewer's feedback and are not comparable across raters.",
            "Kappa is prevalence-sensitive: the candidate-by-expected grid is mostly non-equivalent by construction.",
            "Node match rate and the implied pass rate use the same matching rule and the same reviewer-independent design checks, so differences between reviewers come only from the equivalence labels.",
            "The implied pass rate applies the revised rule of decision 0014 and inherits its declared, unvalidated thresholds.",
        ),
    )
