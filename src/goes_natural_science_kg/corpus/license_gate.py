# src/goes_natural_science_kg/corpus/license_gate.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Admit only scoped official publications or explicitly reviewed open/official sources.
"""Pure gate decisions; rejection is data and never aborts an otherwise valid batch."""

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import yaml

from goes_natural_science_kg.schemas.corpus import (
    EvidenceBasis,
    GateStatus,
    LicenseEvidence,
    LicenseLabel,
    SourceAllowlist,
    SourceDocument,
)


def load_allowlist(path: Path | None = None) -> SourceAllowlist:
    """Load and validate the single packaged source policy."""
    text = (
        path.read_text(encoding="utf-8")
        if path
        else files("goes_natural_science_kg.corpus")
        .joinpath("allowlist.yaml")
        .read_text(encoding="utf-8")
    )
    return SourceAllowlist.model_validate(yaml.safe_load(text))


def _evidence(
    document: SourceDocument,
    policy: SourceAllowlist,
    checked_at: datetime,
) -> tuple[LicenseEvidence | None, str | None]:
    explicit = document.license_evidence
    if document.license == LicenseLabel.RESTRICTED or (
        explicit and explicit.license == LicenseLabel.RESTRICTED
    ):
        return explicit, "explicit_restriction"
    if explicit and explicit.verification_method == "explicit_review":
        if explicit.reviewed_at > checked_at:
            return explicit, "review_timestamp_after_gate"
        if document.license not in {LicenseLabel.UNKNOWN, explicit.license}:
            return explicit, "declared_license_conflicts_with_review"
        return explicit, None
    matching = sorted((r for r in policy.domains if r.matches(document)), key=lambda r: r.id)
    # A specific per-document review requirement takes precedence over a broad official rule.
    if not matching or any(r.admission == "explicit_review" for r in matching):
        return None, "explicit_license_review_required"
    rule = matching[0]
    if document.license not in {LicenseLabel.UNKNOWN, LicenseLabel.OFFICIAL_PUBLICATION}:
        return None, "declared_open_license_has_no_document_evidence"
    return LicenseEvidence(
        document_url=document.url,
        evidence_url=rule.evidence_url,
        license=LicenseLabel.OFFICIAL_PUBLICATION,
        basis=EvidenceBasis.OFFICIAL_PUBLICATION,
        statement=f"Official-publication admission under rule {rule.id}; not an open copyright licence.",
        reviewed_by=f"allowlist:{policy.policy_version}:{rule.id}",
        reviewed_at=checked_at,
        verification_method="allowlist",
    ), None


def evaluate_license(
    document: SourceDocument,
    policy: SourceAllowlist,
    checked_at: datetime,
) -> SourceDocument:
    """Re-evaluate even previously accepted records; unknown/restricted sources are rejected."""
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("checked_at must include a timezone")
    evidence, reason = _evidence(document, policy, checked_at)
    payload = document.model_dump(mode="json")
    payload.update(
        license_gate_status=GateStatus.REJECTED if reason else GateStatus.ACCEPTED,
        license=evidence.license if evidence else document.license,
        license_evidence=evidence.model_dump(mode="json") if evidence else None,
        checked_at=checked_at.isoformat(),
        policy_version=policy.policy_version,
        rejection_reason=reason,
    )
    return SourceDocument.model_validate(payload)
