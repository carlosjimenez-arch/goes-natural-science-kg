# src/goes_natural_science_kg/eval/reporting.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Aggregate paired prompt trials without hiding failures or missing usage.
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from goes_natural_science_kg.eval.cases import ROLES, load_cases
from goes_natural_science_kg.eval.experiment import read_observation
from goes_natural_science_kg.eval.harness import cell_identity, validate_cell_prompts
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.prompt_evaluation import (
    EvaluationCell,
    EvaluationReport,
    ExperimentSettings,
    PromptRegistry,
    ReplicateSummary,
    VariantSummary,
)


def average(values: list[Any]) -> float | None:
    available = [float(v) for v in values if v is not None]
    return mean(available) if available else None


def summarize_variant(cells: list[EvaluationCell], observations: Path) -> VariantSummary:
    first = cells[0]
    final = [
        next(m for m in reversed(c.metrics) if m.case_id == key)
        for c in cells
        for key in c.case_ids
    ]
    initial = [m for c in cells for m in c.metrics if m.revisions == 0]
    replicas = sorted({c.replicate for c in cells})
    rates = [mean(m.passed for m in final if m.replicate == r) for r in replicas]
    hashes = {h for c in cells for h in c.observations}
    records = [read_observation(observations, h) for h in sorted(hashes)]
    schema_rate = mean(m.schema_valid for m in initial)
    deviation = stdev(rates) if len(rates) == 3 else 1.0
    accuracy = average([m.judge_correct for m in initial])
    reasons = []
    if len(final) != 120 or len(replicas) != 3:
        reasons.append("incomplete three-replicate comparison")
    if deviation > 0.1:
        reasons.append("final pass-rate standard deviation exceeds 0.10")
    if schema_rate < 0.95:
        reasons.append("initial schema adherence below 0.95")
    if mean(m.passed for m in final) < 0.8:
        reasons.append("final pass rate below 0.80")
    if accuracy is not None and accuracy < 0.95:
        reasons.append("initial judge accuracy below 0.95")
    if any(m.invalid_reference_rate not in (0, None) for m in final):
        reasons.append("invalid corpus references remain")
    if any(r.estimated_usd is None for r in records):
        reasons.append("incomplete provider usage")
    replica_rows = tuple(summarize_replicate(cells, r, observations) for r in replicas)
    deviations = {
        name: stdev([value for r in replica_rows if (value := r.metrics[name]) is not None])
        if len(replica_rows) == 3 and all(r.metrics[name] is not None for r in replica_rows)
        else None
        for name in replica_rows[0].metrics
    }
    if any(
        value is not None and value > 0.1
        for key, value in deviations.items()
        if key != "mean_revisions"
    ):
        reasons.append("a measured quality-rate standard deviation exceeds 0.10")
    for name in ("estimated_usd", "mean_request_seconds"):
        deviations[name] = (
            stdev(getattr(r, name) for r in replica_rows) if len(replica_rows) == 3 else None
        )
    return VariantSummary(
        role=first.role,
        technique=first.technique,
        cases=len(final),
        replicates=len(replicas),
        initial_pass_rate=mean(m.passed for m in initial),
        final_pass_rate=mean(rates),
        final_pass_sd=deviation,
        schema_rate=schema_rate,
        coverage=average([m.coverage for m in final]),
        prerequisite_precision=average([m.prerequisite_precision for m in final]),
        prerequisite_recall=average([m.prerequisite_recall for m in final]),
        invalid_reference_rate=average([m.invalid_reference_rate for m in final]),
        unsupported_claim_rate=average([m.unsupported_claim_rate for m in final]),
        judge_accuracy=accuracy,
        mean_revisions=mean(m.revisions for m in final),
        censored_rate=mean(m.censored for m in final),
        estimated_usd=sum(r.estimated_usd for r in records if r.estimated_usd is not None),
        missing_usage_requests=sum(r.estimated_usd is None for r in records),
        mean_request_seconds=mean(r.latency_seconds for r in records),
        provider_requests=len(records),
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
        replicate_metrics=replica_rows,
        metric_sd=deviations,
    )


def make_evaluation_report(
    cases_path: Path,
    cells_path: Path,
    observations: Path,
    *,
    settings: ExperimentSettings,
    registry: PromptRegistry,
) -> EvaluationReport:
    cases = load_cases(cases_path)
    cells = [
        EvaluationCell.model_validate_json(p.read_bytes())
        for p in sorted(cells_path.glob("*.json"))
    ]
    expected_ids = {c.id for c in cases if c.split == "evaluation"}
    seen = set()
    by_id = {c.id: c for c in cases}
    for cell in cells:
        prompt_id = cell.role.replace("_", "-") + "-" + cell.technique.replace("_", "-")
        expected_identity = cell_identity(
            tuple(by_id[key] for key in cell.case_ids),
            registry,
            prompt_id,
            cell.replicate,
            settings,
        )
        if expected_identity != cell.id:
            raise ValueError("dataset, settings or prompt differs from measured cell")
        validate_cell_prompts(cell, registry, observations)
        for key in cell.case_ids:
            identity = (cell.role, cell.technique, cell.replicate, key)
            if key not in expected_ids or identity in seen:
                raise ValueError("unknown or duplicate evaluation case")
            seen.add(identity)
    groups = {(c.role, c.technique) for c in cells}
    summaries = tuple(
        summarize_variant([c for c in cells if (c.role, c.technique) == group], observations)
        for group in sorted(groups)
    )
    complete = (
        len(cells) == 630
        and len(summaries) == 21
        and all(s.cases == 120 and s.replicates == 3 for s in summaries)
    )
    human = all(c.annotation_status == "human_reviewed" for c in cases)
    choices = {}
    for role in ROLES:
        eligible = [s for s in summaries if s.role == role and s.eligible and complete]
        ordered = sorted(
            eligible,
            key=lambda s: (
                -s.final_pass_rate,
                -s.initial_pass_rate,
                s.estimated_usd,
                s.mean_request_seconds,
                s.technique,
            ),
        )
        choices[role] = ordered[0].technique if ordered else None
    all_records = [
        read_observation(observations, h)
        for h in sorted({h for c in cells for h in c.observations})
    ]
    return EvaluationReport(
        dataset_sha256=content_hash([c.model_dump(mode="json") for c in cases]),
        estimated_total_usd=sum(
            r.estimated_usd for r in all_records if r.estimated_usd is not None
        ),
        missing_usage_requests=sum(r.estimated_usd is None for r in all_records),
        provider_requests=len(all_records),
        human_reviewed=human,
        complete=complete,
        expected_cells=630,
        completed_cells=len(cells),
        variants=summaries,
        provisional_choices=choices,
        production_choices=choices if human else dict.fromkeys(ROLES),
        limitations=(
            "Reference annotations and examples are agent-authored, pending independent teacher review.",
            "Semantic entailment and equivalence are model judgments plus exact quote checks, not automatic proof.",
            "Three seeded replicates estimate instability weakly; cases within a provider batch are correlated.",
            "Costs use recorded token usage and published USD list rates, not a billing invoice; missing usage is unknown.",
            "Curriculum task support is model-assessed against exact cited paragraphs; hypothetical activities are distinguished from factual assertions.",
            "Null metrics are not applicable or unavailable, never evidence of zero hallucination.",
            "Revisions include right-censored failures at three; mean revisions alone is not a success measure.",
        ),
    )


def summarize_replicate(
    cells: list[EvaluationCell], replica: int, observations: Path
) -> ReplicateSummary:
    selected = [c for c in cells if c.replicate == replica]
    final = [
        next(m for m in reversed(c.metrics) if m.case_id == key)
        for c in selected
        for key in c.case_ids
    ]
    initial = [m for c in selected for m in c.metrics if m.revisions == 0]
    hashes = {h for c in selected for h in c.observations}
    records = [read_observation(observations, h) for h in sorted(hashes)]
    values: dict[str, list[Any]] = {
        "initial_pass_rate": [m.passed for m in initial],
        "final_pass_rate": [m.passed for m in final],
        "initial_schema_rate": [m.schema_valid for m in initial],
        "initial_judge_accuracy": [m.judge_correct for m in initial],
        "mean_revisions": [m.revisions for m in final],
        "censored_rate": [m.censored for m in final],
    }
    for name in (
        "coverage",
        "prerequisite_precision",
        "prerequisite_recall",
        "invalid_reference_rate",
        "unsupported_claim_rate",
    ):
        values[name] = [getattr(m, name) for m in final]
    return ReplicateSummary(
        replicate=replica,
        cases=len(final),
        metrics={name: average(v) for name, v in values.items()},
        metric_sample_counts={name: sum(x is not None for x in v) for name, v in values.items()},
        estimated_usd=sum(r.estimated_usd for r in records if r.estimated_usd is not None),
        missing_usage_requests=sum(r.estimated_usd is None for r in records),
        mean_request_seconds=mean(r.latency_seconds for r in records),
        provider_requests=len(records),
    )
