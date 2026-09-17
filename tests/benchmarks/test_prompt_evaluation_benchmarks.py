# tests/benchmarks/test_prompt_evaluation_benchmarks.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Measure prompt loading and semantic correspondence assignment.
from pathlib import Path

import pytest

from goes_natural_science_kg.eval.cases import load_cases
from goes_natural_science_kg.eval.metrics import matched_keys
from goes_natural_science_kg.eval.registry import load_registry
from goes_natural_science_kg.schemas.prompt_evaluation import ReferenceReview, SemanticMatch

ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_prompt_registry(benchmark):
    registry = benchmark(load_registry, ROOT / "prompts")
    benchmark.extra_info["prompt_count"] = len(registry.artifacts)
    assert len(registry.artifacts) >= 21


@pytest.mark.benchmark(min_rounds=1000)
def test_benchmark_reference_matching(benchmark):
    case = load_cases(ROOT / "tests/golden/prompt-evaluation/cases.jsonl")[0]
    review = ReferenceReview(
        matches=tuple(
            SemanticMatch(
                candidate_key=m.key,
                expected_key=m.key,
                equivalent=True,
                rationale="Identity matching benchmark.",
            )
            for m in case.expected.micros
        ),
        support=(),
    )
    assert len(benchmark(matched_keys, case.expected, case.expected, review)) == 3
