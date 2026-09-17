# tests/property/test_release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Verify proposal identity and dependency checks under generated graph changes.
from hypothesis import given
from hypothesis import strategies as st

from goes_natural_science_kg.curriculum.release import entity_identity
from goes_natural_science_kg.schemas.identity import stable_id


@given(
    st.integers(min_value=0, max_value=9999),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=80),
)
def test_long_local_keys_remain_deterministic_and_bounded(unit, local):
    identity = entity_identity(f"unit-{unit}", local)
    assert len(identity.key) <= 80
    assert stable_id("micro", identity) == stable_id(
        "micro", entity_identity(f"unit-{unit}", local)
    )
    assert stable_id("micro", identity) != stable_id(
        "micro", entity_identity(f"unit-{unit + 1}", local)
    )


def test_identity_component_boundaries_cannot_collide():
    assert entity_identity("a-b", "c") != entity_identity("a", "b-c")
