# tests/unit/test_edge_and_manifest_boundaries.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Protect symmetric co-requisites, evidence sets and completed manifest boundaries.
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_natural_science_kg.corpus.manifest import read_manifest, write_manifest
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.corpus import SourceDocument
from goes_natural_science_kg.schemas.graph import GraphSnapshot
from goes_natural_science_kg.schemas.skills import Edge, MicroSkill


def test_co_requisite_symmetry(before):
    data = before.edges[0].model_dump()
    data["type"] = "CO_REQUISITE"
    left = Edge.model_validate(data)
    right = Edge.model_validate({**data, "source": data["target"], "target": data["source"]})
    assert canonical_json(left) == canonical_json(right)


def test_all_snapshot_permutations_are_canonical(before):
    data = before.model_dump(mode="json")
    for field in ["skills", "micro_skills", "edges", "source_ids"]:
        data[field].reverse()
    assert canonical_json(GraphSnapshot.model_validate(data)) == canonical_json(before)


def test_duplicate_evidence_and_self_edges_rejected(before):
    data = before.edges[0].model_dump()
    with pytest.raises(ValidationError, match="duplicate values"):
        Edge.model_validate({**data, "source_refs": data["source_refs"] * 2})
    with pytest.raises(ValidationError, match="self edges"):
        Edge.model_validate({**data, "target": data["source"]})
    with pytest.raises(ValidationError):
        MicroSkill.model_validate({**before.micro_skills[0].model_dump(), "confidence": True})


def test_pending_and_duplicate_manifest_records_rejected(tmp_path):
    data = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/source-candidates.json").read_text()
    )
    doc = SourceDocument.model_validate(data[0])
    with pytest.raises(ValueError, match="completed"):
        write_manifest(tmp_path / "corpus.jsonl", (doc,))
    with pytest.raises(ValueError, match="duplicate"):
        write_manifest(tmp_path / "corpus.jsonl", (doc, doc))
    path = tmp_path / "duplicate.jsonl"
    path.write_text((canonical_json(doc) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)
