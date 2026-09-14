# tests/benchmarks/test_sequencing_benchmarks.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure real CP-SAT construction, audit and graph interchange.
from pathlib import Path

import pytest

from goes_natural_science_kg.curriculum.artifacts import read_input
from goes_natural_science_kg.curriculum.solver import solve
from goes_natural_science_kg.curriculum.validation import preflight
from goes_natural_science_kg.graph.linked_data import export_jsonld
from goes_natural_science_kg.schemas.sequencing import SequencingSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.benchmark(min_rounds=1, max_time=0.1, disable_gc=True)
def test_cp_sat_schedule_205_activities(benchmark):
    data = read_input(ROOT / "tests/golden/sequencing/engineering.json")
    result = benchmark(solve, data, SequencingSettings(max_deterministic_time=1))
    assert result.status in {"optimal", "feasible"}


def test_solver_preflight_and_source_graph(benchmark):
    data = read_input(ROOT / "tests/golden/sequencing/engineering.json")
    assert not benchmark(preflight, data, SequencingSettings())


def test_jsonld_120_micro_skills(benchmark):
    data = read_input(ROOT / "data/processed/curriculum/provisional-graph.json")
    assert len(benchmark(export_jsonld, data.graph).nodes) > 120
