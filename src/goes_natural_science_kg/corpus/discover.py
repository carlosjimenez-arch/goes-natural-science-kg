# src/goes_natural_science_kg/corpus/discover.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Load reviewed primary-source discovery manifests and measure honest coverage.
"""A reproducible curated discovery registry; automatic web search never approves rights."""

from pathlib import Path

from goes_natural_science_kg.schemas.ingestion import (
    CorpusCoverage,
    DiscoveryRecord,
    DocumentOutcome,
    GradeCoverage,
)


def discover(path: Path) -> tuple[DiscoveryRecord, ...]:
    records = tuple(
        DiscoveryRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if len({r.document.id for r in records}) != len(records):
        raise ValueError("duplicate discovery source IDs")
    return tuple(sorted(records, key=lambda r: r.document.id))


def coverage(
    records: tuple[DiscoveryRecord, ...], outcomes: tuple[DocumentOutcome, ...]
) -> CorpusCoverage:
    indexed = {o.source_id for o in outcomes if o.status == "indexed"}
    ready = [r for r in records if r.document.id in indexed]
    grades = []
    for grade in range(2, 7):
        matching = [
            r
            for r in ready
            if r.role == "national_curriculum" and any(grade in s.grades for s in r.sections)
        ]
        # A mirror of identical bytes is one source, not two towards the quota.
        unique = {
            r.document.sha256: r
            for r in sorted(matching, key=lambda r: r.document.id, reverse=True)
            if r.document.sha256
        }
        countries = tuple(sorted({r.document.country for r in unique.values()}))
        grades.append(
            GradeCoverage(
                grade=grade,
                source_ids=tuple(sorted(r.document.id for r in unique.values())),
                distinct_document_hashes=tuple(sorted(unique)),
                countries=countries,
                meets_target=8 <= len(unique) <= 10 and len(countries) >= 4,
            )
        )
    comparators = {
        name: tuple(sorted(r.document.id for r in ready if r.framework == name))
        for name in ("PISA", "ERCE", "TIMSS", "NGSS")
    }
    missing = tuple(name for name, ids in comparators.items() if not ids)
    return CorpusCoverage(
        grades=tuple(grades),
        comparators=comparators,
        missing_comparators=missing,
        complete=all(g.meets_target for g in grades) and not missing,
    )
