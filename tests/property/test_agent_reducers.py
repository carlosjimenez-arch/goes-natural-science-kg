# tests/property/test_agent_reducers.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
from hypothesis import given
from hypothesis import strategies as st

from goes_natural_science_kg.schemas.agent_state import merge_keyed


@given(
    st.dictionaries(st.text(min_size=1), st.integers()),
    st.dictionaries(st.text(min_size=1), st.integers()),
)
def test_reducer_permutation_and_partition_invariance(a, b):
    b = {k: v for k, v in b.items() if k not in a}
    assert merge_keyed(a, b) == merge_keyed(b, a)
    assert merge_keyed(merge_keyed(a, b), a) == merge_keyed(a, b)
    assert list(merge_keyed(a, b)) == sorted(a.keys() | b.keys())
