# tests/unit/test_diff.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Compare semantic results against independently specified golden expectations.
import json
from pathlib import Path

from goes_natural_science_kg.graph.diff import diff_graphs, summarize_diff
from goes_natural_science_kg.schemas.graph import GraphSnapshot


def test_golden_diff(before, after):
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/diff-expected.json").read_text()
    )
    result = diff_graphs(before, after)
    for name in ["nodes", "edges"]:
        group = getattr(result, name)
        assert [x.identity.key for x in group.added] == expected[name]["added"]
        assert [x.identity.key for x in group.removed] == expected[name]["removed"]
        assert {x.after.identity.key: list(x.changed_fields) for x in group.modified} == expected[
            name
        ]["modified"]
    assert "nodes: +1 -1 ~2" in summarize_diff(result)
    assert "edges: +1 -1 ~1" in summarize_diff(result)


def test_self_and_empty_diff(before):
    result = diff_graphs(before, before)
    assert result.before_sha256 == result.after_sha256
    assert not any(
        [result.nodes.added, result.nodes.removed, result.nodes.modified, result.edges.modified]
    )
    empty = GraphSnapshot(version="empty")
    assert len(diff_graphs(empty, before).nodes.added) == 4


def test_reverse_diff(before, after):
    forward, reverse = diff_graphs(before, after), diff_graphs(after, before)
    assert forward.nodes.added == reverse.nodes.removed
    assert forward.edges.removed == reverse.edges.added
    assert forward.nodes.modified[0].before == reverse.nodes.modified[0].after
