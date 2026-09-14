# tests/property/test_calibration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify panel-derived confidence and human-labelled calibration on real recorded reports.
from datetime import UTC, datetime

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from goes_natural_science_kg.eval.calibration import apply_blocks, fit_isotonic
from goes_natural_science_kg.schemas.calibration import CalibrationSample

FITTED_AT = datetime(2026, 9, 14, tzinfo=UTC)
pairs = st.lists(
    st.tuples(st.floats(0.0, 1.0, allow_nan=False, width=32), st.booleans()),
    min_size=30,
    max_size=120,
)


def samples(values: list[tuple[float, bool]]) -> list[CalibrationSample]:
    return [
        CalibrationSample(
            entity_id=f"micro-case-{i}-{'0' * 16}",
            raw_score=float(score),
            human_accepted=label,
            reviewer="specialist (role)",
            reviewed_at=FITTED_AT,
        )
        for i, (score, label) in enumerate(values)
    ]


@given(pairs, st.randoms(use_true_random=False))
def test_isotonic_calibration_is_monotone_bounded_and_permutation_invariant(values, rng):
    labels = {label for _, label in values}
    assume(len(labels) == 2)
    record = fit_isotonic(samples(values), FITTED_AT)
    rates = np.array([b.rate for b in record.blocks])
    assert np.all(np.diff(rates) >= 0)
    grid = np.linspace(0.0, 1.0, 41)
    mapped = apply_blocks(grid, record.blocks)
    assert np.all((mapped >= 0.0) & (mapped <= 1.0))
    assert np.all(np.diff(mapped) >= 0)
    assert record.positives == sum(label for _, label in values)
    shuffled = list(values)
    rng.shuffle(shuffled)
    assert fit_isotonic(samples(shuffled), FITTED_AT).blocks == record.blocks
