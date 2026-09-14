# tests/unit/test_confidence.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify panel-derived confidence and human-labelled calibration on real recorded reports.
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from goes_natural_science_kg.cli import app
from goes_natural_science_kg.eval.calibration import (
    apply_calibration,
    confidence_report,
    estimate_confidence,
    fit_isotonic,
)
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.calibration import CalibrationRecord, CalibrationSample
from goes_natural_science_kg.schemas.orchestration import Ballot, OrchestrationReport

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/orchestration"
FITTED_AT = datetime(2026, 9, 14, tzinfo=UTC)


def golden_report() -> OrchestrationReport:
    return OrchestrationReport.model_validate_json((GOLDEN / "report.json").read_bytes())


def synthetic_samples(count: int, seed: int = 0) -> list[CalibrationSample]:
    """Acceptance probability grows with the raw score, but the raw score is over-confident."""
    generator = np.random.default_rng(seed)
    scores = generator.uniform(0.0, 1.0, count)
    accepted = generator.uniform(0.0, 1.0, count) < scores**2
    return [
        CalibrationSample(
            entity_id=f"micro-sample-{i}-{'0' * 16}",
            raw_score=float(round(s, 6)),
            human_accepted=bool(a),
            reviewer="curriculum specialist (role)",
            reviewed_at=FITTED_AT,
        )
        for i, (s, a) in enumerate(zip(scores, accepted, strict=True))
    ]


def test_panel_confidence_replaces_self_report_on_recorded_hierarchy():
    report = confidence_report(golden_report(), None)
    judged = {
        m.skill.id for item in golden_report().items if item.level == "L2" for m in item.micros
    }
    carried = {m.skill.id for item in golden_report().items for m in item.micros}
    assert judged == carried and report.summary.estimates == len(judged) > 0
    assert {e.entity_id for e in report.estimates} == judged
    judging_item = next(i.item_id for i in golden_report().items if i.level == "L2")
    assert all(e.item_id == judging_item for e in report.estimates)
    for estimate in report.estimates:
        assert estimate.source == "panel_derived" and not estimate.calibrated
        assert estimate.components.self_reported == pytest.approx(0.9)
        assert estimate.components.vote_fraction == 1.0
        assert estimate.components.mean_judge_score == 1.0
        assert estimate.components.hard_error_free is True
        assert estimate.value == estimate.raw_score == 1.0
    assert report.summary.mean_absolute_shift == pytest.approx(0.1)
    assert [e.entity_id for e in report.estimates] == sorted(e.entity_id for e in report.estimates)


def test_hard_errors_dominate_and_missing_panel_falls_back_to_self_report():
    item = next(i for i in golden_report().items if i.micros)
    micro = item.micros[0]
    last = item.ballots[-1]
    failed = item.model_copy(
        update={
            "ballots": (
                *item.ballots[:-1],
                Ballot(
                    revision=last.revision,
                    verdicts=last.verdicts,
                    hard_errors=("cycle detected",),
                    accepted=False,
                    votes=last.votes,
                ),
            )
        }
    )
    assert estimate_confidence(failed, micro, None).value == 0.0
    unjudged = item.model_copy(update={"ballots": ()})
    fallback = estimate_confidence(unjudged, micro, None)
    assert fallback.source == "self_reported" and fallback.value == micro.skill.confidence


def test_isotonic_fit_is_monotone_and_reports_calibration_quality():
    record = fit_isotonic(synthetic_samples(400), FITTED_AT)
    rates = [b.rate for b in record.blocks]
    assert rates == sorted(rates)
    assert 0.0 <= record.expected_calibration_error <= 0.25
    assert record.brier_score < 0.25
    values = [apply_calibration(x, record) for x in np.linspace(0.0, 1.0, 21)]
    assert values == sorted(values)
    # Over-confident raw scores are pulled down towards the empirical acceptance rate.
    assert apply_calibration(0.5, record) < 0.5


def test_calibration_rejects_small_or_single_class_samples():
    with pytest.raises(ValueError, match="insufficient"):
        fit_isotonic(synthetic_samples(10), FITTED_AT)
    positives = [s.model_copy(update={"human_accepted": True}) for s in synthetic_samples(40)]
    with pytest.raises(ValueError, match="both accepted and rejected"):
        fit_isotonic(positives, FITTED_AT)


def test_calibrated_estimates_record_their_calibration_id():
    record = fit_isotonic(synthetic_samples(200), FITTED_AT)
    report = confidence_report(golden_report(), record)
    assert report.calibration_id == record.id
    assert all(e.calibrated and e.calibration_id == record.id for e in report.estimates)
    assert all(e.value == apply_calibration(e.raw_score, record) for e in report.estimates)
    reloaded = CalibrationRecord.model_validate_json(canonical_json(record))
    assert reloaded == record


def test_cli_fit_and_estimate_write_one_manifest(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(canonical_json(s) + "\n" for s in synthetic_samples(120)))
    calibration = tmp_path / "calibration.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "confidence-fit",
            str(samples),
            "--output",
            str(calibration),
            "--fitted-at",
            "2026-09-14T00:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    output = tmp_path / "estimates"
    result = runner.invoke(
        app,
        [
            "confidence-estimate",
            str(GOLDEN / "report.json"),
            "--output",
            str(output),
            "--generated-at",
            "2026-09-14T00:00:00Z",
            "--calibration",
            str(calibration),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == {"report.json", "confidence.json"}
    assert manifest["inputs"]["confidence-calibration"] == json.loads(calibration.read_text())["id"]
    assert (output / "report.json").read_bytes() == (GOLDEN / "report.json").read_bytes()
    short = tmp_path / "short.jsonl"
    short.write_text("".join(canonical_json(s) + "\n" for s in synthetic_samples(5)))
    result = runner.invoke(
        app,
        [
            "confidence-fit",
            str(short),
            "--output",
            str(calibration),
            "--fitted-at",
            "2026-09-14T00:00:00Z",
        ],
    )
    assert result.exit_code == 2
