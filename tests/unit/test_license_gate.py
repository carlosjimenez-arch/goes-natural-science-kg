# tests/unit/test_license_gate.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Exercise genuine gate decisions, including restrictions and policy revocation.
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from goes_natural_science_kg.corpus.license_gate import evaluate_license, load_allowlist
from goes_natural_science_kg.schemas.corpus import SourceAllowlist, SourceDocument

CHECKED = datetime(2026, 9, 14, tzinfo=UTC)
GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def candidate(index=0, **changes):
    data = json.loads((GOLDEN / "source-candidates.json").read_text())[index]
    data.update(changes)
    return SourceDocument.model_validate(data)


def test_official_and_unknown_and_explicit_open():
    results = [evaluate_license(candidate(i), load_allowlist(), CHECKED) for i in range(3)]
    assert [r.license_gate_status.value for r in results] == ["accepted", "rejected", "accepted"]
    assert results[0].license.value == "official-publication"
    assert results[1].rejection_reason == "explicit_license_review_required"
    assert results[2].license.value == "CC-BY-4.0"
    assert all(r.sha256 is None for r in results)


@pytest.mark.parametrize(
    "url",
    [
        "https://mined.gob.sv.evil.example/book",
        "https://evilmined.gob.sv/book",
        "https://example.org/book?host=mined.gob.sv",
    ],
)
def test_domain_spoofing(url):
    assert (
        evaluate_license(candidate(url=url), load_allowlist(), CHECKED).license_gate_status.value
        == "rejected"
    )


def test_restrictions_override_official_domain():
    result = evaluate_license(candidate(license="restricted"), load_allowlist(), CHECKED)
    assert result.license_gate_status.value == "rejected"
    assert result.rejection_reason == "explicit_restriction"


def test_claiming_cc_without_evidence_fails():
    result = evaluate_license(candidate(license="CC-BY-4.0"), load_allowlist(), CHECKED)
    assert result.license_gate_status.value == "rejected"


def test_country_and_document_type_are_scoped():
    for changes in [dict(country="MX"), dict(type="research")]:
        assert (
            evaluate_license(
                candidate(**changes), load_allowlist(), CHECKED
            ).license_gate_status.value
            == "rejected"
        )


def test_review_timestamp_and_document_binding():
    doc = candidate(2).model_dump(mode="json")
    doc["license_evidence"]["reviewed_at"] = "2026-09-15T00:00:00Z"
    assert (
        evaluate_license(
            SourceDocument.model_validate(doc), load_allowlist(), CHECKED
        ).rejection_reason
        == "review_timestamp_after_gate"
    )
    doc["license_evidence"]["document_url"] = "https://example.org/another-book"
    with pytest.raises(ValueError, match="different document"):
        SourceDocument.model_validate(doc)


def test_revoked_policy_does_not_reuse_domain_attestation():
    accepted = evaluate_license(candidate(), load_allowlist(), CHECKED)
    revoked = SourceAllowlist(policy_version="revoked", domains=())
    assert evaluate_license(accepted, revoked, CHECKED).license_gate_status.value == "rejected"


def test_shared_cdn_and_international_publishers_need_review():
    for url, country in [
        ("https://assets-us-01.kc-usercontent.com/another-tenant/file", "CA"),
        ("https://www.unesco.org/book", "INT"),
        ("https://www.nextgenscience.org/book", "US"),
    ]:
        result = evaluate_license(candidate(url=url, country=country), load_allowlist(), CHECKED)
        assert result.license_gate_status.value == "rejected"


def test_naive_clock_rejected():
    with pytest.raises(ValueError, match="timezone"):
        evaluate_license(candidate(), load_allowlist(), datetime(2026, 9, 14))
