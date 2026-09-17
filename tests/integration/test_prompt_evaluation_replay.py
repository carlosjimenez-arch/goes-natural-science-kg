# tests/integration/test_prompt_evaluation_replay.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Replay actual Vertex observations through scoring with networking disabled.
import asyncio
from pathlib import Path

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.harness import run_cell
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.prompt_evaluation import EvaluationCell, ExperimentSettings

ROOT = Path(__file__).resolve().parents[2]


def test_real_recorded_evaluation_replays_byte_identically(tmp_path):
    folder = ROOT / "tests/golden/prompt-evaluation"
    expected = EvaluationCell.model_validate_json((folder / "replay-cell.json").read_bytes())
    cases = tuple(c for c in load_cases(folder / "cases.jsonl") if c.split == "evaluation")
    by_id = {c.id: c for c in cases}
    selected = tuple(by_id[key] for key in expected.case_ids)
    settings = ExperimentSettings.model_validate_json(
        (ROOT / "data/processed/prompt-evaluation/settings.json").read_bytes()
    )
    asyncio.run(
        run_cell(
            None,
            asyncio.Semaphore(1),
            settings,
            load_registry(ROOT / "prompts"),
            selected,
            expected.role,
            expected.technique,
            expected.replicate,
            folder / "observations",
            tmp_path,
            {c.id for i, c in enumerate(cases) if i % 2},
        )
    )
    assert (tmp_path / (expected.id + ".json")).read_bytes() == (
        folder / "replay-cell.json"
    ).read_bytes()


def test_complete_recorded_experiment_and_report_replay(tmp_path):
    from goes_natural_science_kg.eval.harness import run_experiment
    from goes_natural_science_kg.eval.reporting import make_evaluation_report
    from goes_natural_science_kg.schemas.base import canonical_json

    folder = ROOT / "data/processed/prompt-evaluation"
    cases = ROOT / "tests/golden/prompt-evaluation/cases.jsonl"
    settings = ExperimentSettings.model_validate_json((folder / "settings.json").read_bytes())
    asyncio.run(
        run_experiment(
            cases, ROOT / "prompts", settings, folder / "observations", tmp_path, online=False
        )
    )
    expected = sorted((folder / "cells").glob("*.json"))
    assert len(expected) == 630
    for path in expected:
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    report = make_evaluation_report(
        cases,
        tmp_path,
        folder / "observations",
        settings=settings,
        registry=load_registry(ROOT / "prompts"),
    )
    assert report.complete and len(report.variants) == 21
    assert canonical_json(report) + "\n" == (folder / "report.json").read_text()
    assert all(value is None for value in report.production_choices.values())


def test_identical_concurrent_requests_share_one_recorded_provider_call(tmp_path):
    from types import SimpleNamespace

    from goes_natural_science_kg.eval.experiment import evaluate_request, read_observation
    from goes_natural_science_kg.schemas.orchestration import Decomposition
    from goes_natural_science_kg.schemas.prompt_evaluation import BatchOutput

    folder = ROOT / "tests/golden/prompt-evaluation"
    cell = EvaluationCell.model_validate_json((folder / "replay-cell.json").read_bytes())
    record = read_observation(folder / "observations", cell.observations[0])
    calls = []

    async def recorded_provider(**kwargs):
        # Only the LLM transport is replaced, using actual retained response and usage fields.
        calls.append(kwargs)
        await asyncio.sleep(0)
        return SimpleNamespace(
            text=record.response,
            model_version=record.model_version,
            usage_metadata=SimpleNamespace(
                prompt_token_count=record.input_tokens,
                candidates_token_count=record.output_tokens,
                thoughts_token_count=record.reasoning_tokens,
                cached_content_token_count=record.cached_input_tokens,
            ),
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorded_provider))
    )
    settings = ExperimentSettings.model_validate_json(
        (ROOT / "data/processed/prompt-evaluation/settings.json").read_bytes()
    )
    registry = load_registry(ROOT / "prompts")

    async def execute_concurrently():
        locks = {}
        return await asyncio.gather(
            *(
                evaluate_request(
                    client,
                    asyncio.Semaphore(4),
                    tmp_path,
                    settings,
                    registry,
                    record.request["prompt_id"],
                    record.request["input"],
                    record.request["model"],
                    BatchOutput[Decomposition],
                    record.request["replicate"],
                    record.request["revision"],
                    locks,
                )
                for _ in range(4)
            )
        )

    results = asyncio.run(execute_concurrently())
    assert len(calls) == 1
    assert {r.response_sha256 for r in results} == {record.response_sha256}
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_recorded_followup_arms_replay_and_report_byte_identically(tmp_path):
    from goes_natural_science_kg.eval.harness import run_followup
    from goes_natural_science_kg.eval.reporting import make_followup_report
    from goes_natural_science_kg.schemas.base import canonical_json
    from goes_natural_science_kg.schemas.prompt_evaluation import FollowUpPlan, FollowUpReport

    folder = ROOT / "data/processed/prompt-evaluation"
    cases = ROOT / "tests/golden/prompt-evaluation/cases.jsonl"
    plan = FollowUpPlan.model_validate_json((folder / "followup-plan.json").read_bytes())
    asyncio.run(
        run_followup(plan, cases, ROOT / "prompts", folder / "observations", tmp_path, online=False)
    )
    expected = sorted((folder / "followup-cells").glob("*.json"))
    assert len(expected) == len(plan.arms) * plan.replicates * 10 == 210
    for path in expected:
        assert (tmp_path / path.name).read_bytes() == path.read_bytes()
    report = make_followup_report(
        plan, cases, tmp_path, folder / "observations", registry=load_registry(ROOT / "prompts")
    )
    assert canonical_json(report) + "\n" == (folder / "followup-report.json").read_text()
    stored = FollowUpReport.model_validate_json((folder / "followup-report.json").read_bytes())
    # Declared rule: no arm reached the 0.80 pass threshold, so nothing is selected.
    assert stored.complete and stored.provisional_choice is None
    assert stored.production_choice is None
    assert all(not a.summary.eligible for a in stored.arms)
    baseline = next(a for a in stored.arms if a.arm.key == plan.baseline_arm)
    assert baseline.comparison is None
    # Every non-baseline arm carries a paired bootstrap interval; none excludes zero upward.
    for arm in stored.arms:
        if arm.arm.key != plan.baseline_arm:
            assert arm.comparison is not None and not arm.comparison.improves
    # Flash-lite failed on a provider constraint and is recorded, not hidden.
    lite = next(a for a in stored.arms if a.arm.generator_model == "gemini-2.5-flash-lite")
    assert (
        lite.summary.schema_rate == 0.0
        and "incomplete provider usage" in lite.summary.rejection_reasons
    )


def test_recorded_responses_rescore_to_the_committed_revised_report(tmp_path):
    from goes_natural_science_kg.eval.rescore import make_rescore_report
    from goes_natural_science_kg.schemas.base import canonical_json
    from goes_natural_science_kg.schemas.rescoring import RescoreReport

    folder = ROOT / "data/processed/prompt-evaluation"
    report = make_rescore_report(
        folder / "followup-cells",
        ROOT / "tests/golden/prompt-evaluation/cases.jsonl",
        folder / "observations",
        folder / "followup-report.json",
    )
    assert canonical_json(report) + "\n" == (folder / "rescore-report.json").read_text()
    stored = RescoreReport.model_validate_json((folder / "rescore-report.json").read_bytes())
    assert stored.rescored_case_replicates == 840
    # The frozen scorer conflated node naming with dependency order; the revised one does not.
    working = [a for a in stored.arms if a.node_match_rate]
    assert len(working) == 6
    assert all(a.revised_pass_rate > a.frozen_pass_rate for a in working)
    assert all(a.edge_recall_matched and a.edge_recall_matched >= 0.80 for a in working)
    # Node matching, not dependency order, is what the candidates actually fail.
    failures = [m for m in stored.metrics if not m.revised_passed and m.agreement]
    node_limited = [m for m in failures if m.agreement and m.agreement.node_match_rate < 0.6]
    assert len(node_limited) / len(failures) > 0.8
