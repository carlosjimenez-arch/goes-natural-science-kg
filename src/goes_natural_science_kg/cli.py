# src/goes_natural_science_kg/cli.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Expose contract validation, schema export, semantic diffs and license-gate execution.
"""No curriculum generation commands are advertised before their implementation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import TypeAdapter

from goes_natural_science_kg.agents.license_workflow import build_license_workflow
from goes_natural_science_kg.corpus.license_gate import load_allowlist
from goes_natural_science_kg.graph.diff import diff_graphs, summarize_diff
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.decisions import ArchitectureDecision
from goes_natural_science_kg.schemas.graph import GraphSnapshot
from goes_natural_science_kg.schemas.registry import export_schemas
from goes_natural_science_kg.schemas.workflow import LicenseGateState

app = typer.Typer(no_args_is_help=True, help="Versioned science contracts and provenance tools.")
InputPath = Annotated[Path, typer.Argument(exists=True, dir_okay=False)]


@app.command("graph-diff")
def graph_diff(
    before: InputPath,
    after: InputPath,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print a semantic diff; use --json for the complete machine-readable change set."""
    try:
        result = diff_graphs(
            GraphSnapshot.model_validate_json(before.read_bytes()),
            GraphSnapshot.model_validate_json(after.read_bytes()),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(canonical_json(result) if as_json else summarize_diff(result), nl=as_json)


@app.command("validate-decision")
def validate_decision(path: InputPath) -> None:
    """Validate a decision against its Pydantic contract."""
    try:
        result = ArchitectureDecision.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"{result.id}: {result.status}")


@app.command("schemas-export")
def schemas_export(
    output: Annotated[Path, typer.Argument()],
    generated_at: Annotated[
        datetime, typer.Option("--generated-at", formats=["%Y-%m-%dT%H:%M:%S%z"])
    ],
    seed: Annotated[int, typer.Option(min=0)],
) -> None:
    """Export the versioned schema snapshots with an explicit timestamp and seed."""
    export_schemas(output, generated_at, seed)
    typer.echo(str(output))


@app.command("corpus-gate")
def corpus_gate(
    candidates: InputPath,
    manifest: Annotated[Path, typer.Option("--manifest")],
    checked_at: Annotated[datetime, typer.Option("--checked-at", formats=["%Y-%m-%dT%H:%M:%S%z"])],
    allowlist: Annotated[Path | None, typer.Option("--allowlist", exists=True)] = None,
) -> None:
    """Evaluate a JSON array of SourceDocuments; persist rejections and continue."""
    try:
        documents = TypeAdapter(tuple[SourceDocument, ...]).validate_json(candidates.read_bytes())
        state = LicenseGateState(
            documents=documents,
            allowlist=load_allowlist(allowlist),
            checked_at=checked_at,
            manifest_path=manifest,
        )
        result = LicenseGateState.model_validate(build_license_workflow().invoke(state))
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"accepted={len(result.ingestion_ready_ids)} rejected={len(result.results) - len(result.ingestion_ready_ids)}"
    )


@app.command("corpus-ingest")
def corpus_ingest(
    discovery: Annotated[Path, typer.Argument(exists=True)],
    project: Annotated[str, typer.Option(help="Existing GCP project; ADC supplies credentials.")],
    generated_at: Annotated[datetime, typer.Option(formats=["%Y-%m-%dT%H:%M:%S%z"])],
    seed: Annotated[int, typer.Option()],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    online: bool = typer.Option(False, "--online/--offline"),
) -> None:
    """Rebuild the corpus and report unmet grade or framework coverage explicitly."""
    import asyncio

    from goes_natural_science_kg.corpus.commands import run_ingestion
    from goes_natural_science_kg.schemas.ingestion import IngestionSettings

    report = asyncio.run(
        run_ingestion(
            discovery, data_dir, IngestionSettings(project_id=project), generated_at, seed, online
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if not report.complete:
        raise typer.Exit(2)


@app.command("corpus-search")
def corpus_search(
    query: str,
    project: Annotated[str, typer.Option()],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    limit: int = typer.Option(3, min=1, max=20),
    online: bool = typer.Option(False, "--online/--offline"),
) -> None:
    """Retrieve intact semantic units and their verified page/paragraph anchors."""
    import asyncio

    from goes_natural_science_kg.corpus.commands import run_search
    from goes_natural_science_kg.schemas.ingestion import IngestionSettings

    result = asyncio.run(
        run_search(query, data_dir, IngestionSettings(project_id=project), online, limit)
    )
    typer.echo(canonical_json(result))


@app.command("corpus-trace")
def corpus_trace(chunk_id: str, data_dir: Annotated[Path, typer.Option()] = Path("data")) -> None:
    """Resolve a source_refs chunk ID to original PDF provenance."""
    from goes_natural_science_kg.corpus.trace import trace_references

    typer.echo(canonical_json(trace_references(data_dir, (chunk_id,))))


@app.command("corpus-benchmark")
def corpus_benchmark(
    selection: Annotated[Path, typer.Argument(exists=True)],
    generated_at: Annotated[datetime, typer.Option(formats=["%Y-%m-%dT%H:%M:%S%z"])],
    seed: Annotated[int, typer.Option()],
    data_dir: Annotated[Path, typer.Option()] = Path("data"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/benchmarks"),
    rounds: Annotated[int, typer.Option(min=1, max=10)] = 3,
) -> None:
    """Measure original PDF parsing and real vector indexing, with no cloud calls."""
    from goes_natural_science_kg.corpus.benchmark import benchmark_corpus

    report = benchmark_corpus(selection, data_dir, output, generated_at, seed, rounds)
    typer.echo(f"Recorded {len(report.operations)} measured operations in {output}")


@app.command("orchestrate")
def orchestrate(
    request_path: InputPath,
    settings_path: InputPath,
    generated_at: Annotated[datetime, typer.Option(formats=["%Y-%m-%dT%H:%M:%SZ"])],
    data_dir: Path = Path("data"),
    prompts_dir: Path = Path("prompts"),
    cache_dir: Path = Path(".cache/llm"),
    checkpoints: Path = Path("checkpoints/orchestration.sqlite"),
    output: Path = Path("data/processed/orchestration"),
    online: Annotated[bool, typer.Option("--online/--offline")] = False,
    resume: bool = False,
) -> None:
    """Run four locally evaluated levels; exit 2 if human review prevents full publication."""
    import asyncio

    from goes_natural_science_kg.agents.runner import run_orchestration
    from goes_natural_science_kg.schemas.orchestration import (
        OrchestrationInput,
        OrchestrationSettings,
    )

    request = OrchestrationInput.model_validate_json(request_path.read_bytes())
    settings = OrchestrationSettings.model_validate_json(settings_path.read_bytes())
    result = asyncio.run(
        run_orchestration(
            request,
            settings,
            data_dir,
            prompts_dir,
            cache_dir,
            checkpoints,
            output,
            generated_at.replace(tzinfo=UTC),
            online=online,
            resume=resume,
        )
    )
    typer.echo(f"{len(result.items)} evaluated items; complete={result.complete}")
    if not result.complete:
        raise typer.Exit(2)


@app.command("confidence-fit")
def confidence_fit(
    samples_path: InputPath,
    output: Annotated[Path, typer.Option()],
    fitted_at: Annotated[datetime, typer.Option(formats=["%Y-%m-%dT%H:%M:%SZ"])],
) -> None:
    """Fit isotonic calibration from human-reviewed samples (JSONL); exit 2 when insufficient."""
    from goes_natural_science_kg.corpus.fetch import atomic_bytes
    from goes_natural_science_kg.eval.calibration import fit_isotonic
    from goes_natural_science_kg.schemas.calibration import CalibrationSample

    samples = [
        CalibrationSample.model_validate_json(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        record = fit_isotonic(samples, fitted_at.replace(tzinfo=UTC))
    except ValueError as error:
        typer.echo(f"calibration not fitted: {error}", err=True)
        raise typer.Exit(2) from error
    atomic_bytes(output, (canonical_json(record) + "\n").encode())
    typer.echo(
        f"{record.sample_size} samples; {len(record.blocks)} blocks; "
        f"ECE={record.expected_calibration_error:.3f}; Brier={record.brier_score:.3f}"
    )


@app.command("confidence-estimate")
def confidence_estimate(
    report_path: InputPath,
    output: Annotated[Path, typer.Option()],
    generated_at: Annotated[datetime, typer.Option(formats=["%Y-%m-%dT%H:%M:%SZ"])],
    seed: Annotated[int, typer.Option(min=0)] = 0,
    calibration_path: Annotated[Path | None, typer.Option("--calibration")] = None,
) -> None:
    """Recompute panel-derived confidence for every micro-skill in an orchestration report."""
    from goes_natural_science_kg.agents.runner import write_report
    from goes_natural_science_kg.eval.calibration import confidence_report
    from goes_natural_science_kg.schemas.calibration import CalibrationRecord
    from goes_natural_science_kg.schemas.orchestration import OrchestrationReport

    report = OrchestrationReport.model_validate_json(report_path.read_bytes())
    calibration = (
        None
        if calibration_path is None
        else CalibrationRecord.model_validate_json(calibration_path.read_bytes())
    )
    write_report(output, report, generated_at.replace(tzinfo=UTC), seed, calibration)
    summary = confidence_report(report, calibration).summary
    typer.echo(
        f"{summary.estimates} estimates; calibrated={summary.calibrated}; "
        f"mean shift from self-report={summary.mean_absolute_shift}"
    )


@app.command("prompts-evaluate")
def prompts_evaluate(
    settings_path: Path,
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    prompts_dir: Path = Path("prompts"),
    cache_dir: Path = Path("data/interim/prompt-evaluation/responses"),
    output: Path = Path("data/interim/prompt-evaluation/cells"),
    online: bool = False,
) -> None:
    """Run or resume three paired replicas; network requires --online."""
    import asyncio

    from goes_natural_science_kg.eval.harness import run_experiment
    from goes_natural_science_kg.schemas.prompt_evaluation import ExperimentSettings

    settings = ExperimentSettings.model_validate_json(settings_path.read_bytes())
    asyncio.run(run_experiment(cases_path, prompts_dir, settings, cache_dir, output, online=online))


@app.command("prompts-report")
def prompts_report(
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    cells_path: Path = Path("data/interim/prompt-evaluation/cells"),
    observations: Path = Path("data/interim/prompt-evaluation/responses"),
    settings_path: Path = Path("data/processed/prompt-evaluation/settings.json"),
    prompts_dir: Path = Path("prompts"),
    output: Path = Path("data/processed/prompt-evaluation/report.json"),
) -> None:
    """Aggregate recorded observations and apply the teacher-review promotion gate."""
    from goes_natural_science_kg.corpus.fetch import atomic_bytes
    from goes_natural_science_kg.eval.registry import load_registry
    from goes_natural_science_kg.eval.reporting import make_evaluation_report
    from goes_natural_science_kg.schemas.prompt_evaluation import ExperimentSettings

    report = make_evaluation_report(
        cases_path,
        cells_path,
        observations,
        settings=ExperimentSettings.model_validate_json(settings_path.read_bytes()),
        registry=load_registry(prompts_dir),
    )
    atomic_bytes(output, (canonical_json(report) + "\n").encode())
    typer.echo(
        f"{report.completed_cells}/{report.expected_cells} cells; human_reviewed={report.human_reviewed}"
    )


@app.command("prompts-followup")
def prompts_followup(
    plan_path: InputPath,
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    prompts_dir: Path = Path("prompts"),
    cache_dir: Path = Path("data/interim/prompt-evaluation/responses"),
    output: Path = Path("data/interim/prompt-evaluation/followup-cells"),
    online: bool = False,
) -> None:
    """Run or resume a registered follow-up comparison of prompt/model arms; network requires --online."""
    import asyncio

    from goes_natural_science_kg.eval.harness import run_followup
    from goes_natural_science_kg.schemas.prompt_evaluation import FollowUpPlan

    plan = FollowUpPlan.model_validate_json(plan_path.read_bytes())
    asyncio.run(run_followup(plan, cases_path, prompts_dir, cache_dir, output, online=online))


@app.command("prompts-followup-report")
def prompts_followup_report(
    plan_path: InputPath,
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    cells_path: Path = Path("data/interim/prompt-evaluation/followup-cells"),
    observations: Path = Path("data/interim/prompt-evaluation/responses"),
    prompts_dir: Path = Path("prompts"),
    output: Path = Path("data/processed/prompt-evaluation/followup-report.json"),
) -> None:
    """Aggregate recorded follow-up cells per arm with slices, worst slice and paired bootstrap."""
    from goes_natural_science_kg.corpus.fetch import atomic_bytes
    from goes_natural_science_kg.eval.registry import load_registry
    from goes_natural_science_kg.eval.reporting import make_followup_report
    from goes_natural_science_kg.schemas.prompt_evaluation import FollowUpPlan

    plan = FollowUpPlan.model_validate_json(plan_path.read_bytes())
    report = make_followup_report(
        plan, cases_path, cells_path, observations, registry=load_registry(prompts_dir)
    )
    atomic_bytes(output, (canonical_json(report) + "\n").encode())
    for arm in report.arms:
        worst = arm.worst_slice
        typer.echo(
            f"{arm.arm.key}: final={arm.summary.final_pass_rate:.3f} eligible={arm.summary.eligible}"
            + (f" worst={worst.dimension}={worst.key}:{worst.final_pass_rate:.3f}" if worst else "")
        )
    typer.echo(f"complete={report.complete}; provisional={report.provisional_choice}")


@app.command("prompts-rescore")
def prompts_rescore(
    cells_path: Path = Path("data/processed/prompt-evaluation/followup-cells"),
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    observations: Path = Path("data/processed/prompt-evaluation/observations"),
    source_report: Path = Path("data/processed/prompt-evaluation/followup-report.json"),
    output: Path = Path("data/processed/prompt-evaluation/rescore-report.json"),
) -> None:
    """Re-read recorded decomposition responses under the revised rule; no provider request."""
    from goes_natural_science_kg.corpus.fetch import atomic_bytes
    from goes_natural_science_kg.eval.rescore import make_rescore_report

    report = make_rescore_report(cells_path, cases_path, observations, source_report)
    atomic_bytes(output, (canonical_json(report) + "\n").encode())
    for arm in report.arms:
        worst = arm.worst_grade_slice
        typer.echo(
            f"{arm.arm:22s} frozen={arm.frozen_pass_rate:.3f} revised={arm.revised_pass_rate:.3f} "
            f"nodes={arm.node_match_rate or 0:.3f} edges={arm.edge_recall_matched or 0:.3f}"
            + (f" worst=grade {worst.key}:{worst.revised_pass_rate:.3f}" if worst else "")
        )


@app.command("prompts-reviewer-probe")
def prompts_reviewer_probe(
    plan_path: InputPath,
    cases_path: Path = Path("tests/golden/prompt-evaluation/cases.jsonl"),
    prompts_dir: Path = Path("prompts"),
    cells_path: Path = Path("data/processed/prompt-evaluation/followup-cells"),
    observations: Path = Path("data/processed/prompt-evaluation/observations"),
    cache_dir: Path = Path("data/interim/prompt-evaluation/responses"),
    output: Path = Path("data/processed/prompt-evaluation/reviewer-agreement.json"),
    online: bool = False,
) -> None:
    """Re-judge recorded candidates with independent reviewer models; network requires --online."""
    import asyncio

    from goes_natural_science_kg.corpus.fetch import atomic_bytes
    from goes_natural_science_kg.eval.reviewer_agreement import (
        make_reviewer_report,
        run_reviewer_probe,
    )
    from goes_natural_science_kg.schemas.reviewer_agreement import ReviewerProbePlan

    plan = ReviewerProbePlan.model_validate_json(plan_path.read_bytes())
    recorded, fresh, failures, records, candidates = asyncio.run(
        run_reviewer_probe(
            plan, cases_path, prompts_dir, cells_path, observations, cache_dir, online=online
        )
    )
    report = make_reviewer_report(plan, cases_path, recorded, fresh, failures, records, candidates)
    atomic_bytes(output, (canonical_json(report) + "\n").encode())
    for reviewer in report.reviewers:
        typer.echo(
            f"{reviewer.reviewer:16s} {reviewer.model:24s} node_match={reviewer.node_match_rate:.3f} "
            f"implied_pass={reviewer.implied_revised_pass_rate:.3f} "
            f"equivalent={reviewer.equivalent_pairs}/{reviewer.judged_pairs} failed={reviewer.failed_requests}"
        )
    for pair in report.agreements:
        typer.echo(
            f"{pair.reviewer_a} vs {pair.reviewer_b}: agreement={pair.observed_agreement:.3f} "
            f"kappa={pair.cohen_kappa:.3f} (only_a={pair.only_a_equivalent} only_b={pair.only_b_equivalent})"
        )
    typer.echo(
        f"krippendorff_alpha={report.krippendorff_alpha}; spread={report.node_match_rate_spread:.3f}; "
        f"usd={report.estimated_total_usd:.2f}"
    )


@app.command("build")
def build_curriculum(
    skills_map: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    budget_hours: int = 160,
    seed: int = 42,
    config: Path | None = None,
    output: Path = Path("data/processed/curriculum"),
) -> None:
    """Solve exact annual calendars from a graph and finite authored activity bank."""
    from goes_natural_science_kg.curriculum.artifacts import read_input, write_build
    from goes_natural_science_kg.curriculum.solver import solve
    from goes_natural_science_kg.schemas.sequencing import SequencingSettings

    data = read_input(skills_map)
    from goes_natural_science_kg.config import Settings

    values = (
        yaml.safe_load(config.read_text())
        if config
        else Settings().sequencing.model_dump(mode="json")
    )
    settings = SequencingSettings.model_validate_json(
        canonical_json({**values, "budget_hours": budget_hours, "seed": seed})
    )
    report = solve(data, settings)
    write_build(output, data, settings, report)
    typer.echo(
        f"{report.status}: {len(report.grades)} certified grade schedules; report: {output / 'report.html'}"
    )
    if report.hard_errors or report.status not in ("optimal", "feasible"):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
