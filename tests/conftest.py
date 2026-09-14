# tests/conftest.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Provide reviewed contract fixtures and deterministic offline test configuration.
from pathlib import Path

import pytest
from hypothesis import settings

from goes_natural_science_kg.schemas.graph import GraphSnapshot

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden"
settings.register_profile("offline", derandomize=True, deadline=None, max_examples=40)
settings.load_profile("offline")


@pytest.fixture
def before() -> GraphSnapshot:
    return GraphSnapshot.model_validate_json((GOLDEN / "graph-before.json").read_bytes())


@pytest.fixture
def after() -> GraphSnapshot:
    return GraphSnapshot.model_validate_json((GOLDEN / "graph-after.json").read_bytes())
