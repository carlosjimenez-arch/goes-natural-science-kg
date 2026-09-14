# src/goes_natural_science_kg/corpus/manifest.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Persist accepted and rejected records as deterministic, rebuildable JSONL.
"""Manifest snapshots merge by stable ID and are replaced atomically."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.corpus import GateStatus, SourceDocument


def read_manifest(path: Path) -> tuple[SourceDocument, ...]:
    """Read every record; corruption is an error, not an empty-corpus fallback."""
    records = tuple(
        SourceDocument.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len({r.id for r in records}) != len(records):
        raise ValueError("duplicate manifest source IDs")
    return records


def write_manifest(path: Path, records: tuple[SourceDocument, ...]) -> None:
    """Store all completed verdicts, preserving unrelated existing records."""
    if len({r.id for r in records}) != len(records):
        raise ValueError("duplicate source IDs in manifest update")
    if any(r.license_gate_status == GateStatus.PENDING for r in records):
        raise ValueError("manifest requires completed license-gate results")
    existing = {r.id: r for r in read_manifest(path)} if path.exists() else {}
    for record in records:
        if record.id in existing and record.identity != existing[record.id].identity:
            raise ValueError("stable ID collision")
        existing[record.id] = record
    payload = "".join(canonical_json(existing[key]) + "\n" for key in sorted(existing))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
