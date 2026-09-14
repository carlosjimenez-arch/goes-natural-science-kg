# tests/property/test_determinism.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Prove canonical graph behavior under permutations and Python hash seeds.
import os
import subprocess
import sys

from hypothesis import given
from hypothesis import strategies as st

from goes_natural_science_kg.graph.diff import diff_graphs
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.graph import GraphSnapshot
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.skills import FrameworkRef, Skill


@given(st.permutations(range(6)))
def test_node_permutation_is_semantically_empty(order):
    skills = []
    for i in range(6):
        identity = Identity(namespace="property", key=f"skill-{i}")
        skills.append(
            Skill(
                id=stable_id("skill", identity),
                identity=identity,
                label=f"Skill {i}",
                origin_framework=FrameworkRef(name="test", version="1"),
                domain="matter",
                hierarchy_level=0,
            )
        )
    a = GraphSnapshot(version="v1", skills=tuple(skills))
    b = GraphSnapshot(version="v1", skills=tuple(skills[i] for i in order))
    assert canonical_json(a) == canonical_json(b)
    assert not diff_graphs(a, b).nodes.modified


def test_hash_seed_independence():
    script = "from goes_natural_science_kg.schemas.identity import Identity,stable_id; print(stable_id('skill',Identity(namespace='goes',key='matter')))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        for seed in ["1", "98765"]
    ]
    assert outputs[0] == outputs[1]


@given(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20),
)
def test_identity_distinguishes_namespace(left, right):
    a = Identity(namespace=left, key="matter")
    b = Identity(namespace=right, key="matter")
    assert (stable_id("skill", a) == stable_id("skill", b)) == (left == right)
