# tests/benchmarks/test_calibration_benchmarks.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure calibration fitting and application at graph scale.
from datetime import UTC, datetime

import numpy as np

from goes_natural_science_kg.eval.calibration import apply_blocks, fit_isotonic
from goes_natural_science_kg.schemas.calibration import CalibrationSample

FITTED_AT = datetime(2026, 9, 14, tzinfo=UTC)


def make_samples(count: int) -> list[CalibrationSample]:
    generator = np.random.default_rng(0)
    scores = generator.uniform(0.0, 1.0, count)
    accepted = generator.uniform(0.0, 1.0, count) < scores
    return [
        CalibrationSample(
            entity_id=f"micro-bench-{i}-{'0' * 16}",
            raw_score=float(round(s, 4)),
            human_accepted=bool(a),
            reviewer="specialist (role)",
            reviewed_at=FITTED_AT,
        )
        for i, (s, a) in enumerate(zip(scores, accepted, strict=True))
    ]


def test_fit_isotonic_5000_samples(benchmark):
    samples = make_samples(5_000)
    benchmark(fit_isotonic, samples, FITTED_AT)


def test_apply_calibration_50000_scores(benchmark):
    record = fit_isotonic(make_samples(2_000), FITTED_AT)
    scores = np.random.default_rng(1).uniform(0.0, 1.0, 50_000)
    benchmark(apply_blocks, scores, record.blocks)
