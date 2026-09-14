# src/goes_natural_science_kg/schemas/corpus.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Record document provenance, explicit licence review and gate outcomes.
"""Acceptance is permission to ingest; it does not mean bytes have been fetched."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Self
from urllib.parse import urlsplit

from pydantic import AfterValidator, AwareDatetime, Field, HttpUrl, model_validator

from goes_natural_science_kg.schemas.base import (
    Contract,
    Digest,
    Minutes,
    Slug,
    Text,
    unique_sorted,
)
from goes_natural_science_kg.schemas.identity import Entity


def public_url(value: HttpUrl) -> HttpUrl:
    """Disallow credentials and fragments in provenance URLs."""
    if value.username or value.password or value.fragment:
        raise ValueError("source URLs cannot contain credentials or fragments")
    return value


SourceUrl = Annotated[HttpUrl, AfterValidator(public_url)]


class LicenseLabel(StrEnum):
    UNKNOWN = "unknown"
    OFFICIAL_PUBLICATION = "official-publication"
    CC0 = "CC0-1.0"
    CC_BY = "CC-BY-4.0"
    CC_BY_SA = "CC-BY-SA-4.0"
    CC_BY_IGO = "CC-BY-3.0-IGO"
    CC_BY_SA_IGO = "CC-BY-SA-3.0-IGO"
    OGL_ON = "OGL-Ontario-1.0"
    RESTRICTED = "restricted"


class GateStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DocumentType(StrEnum):
    CURRICULUM = "curriculum"
    TEXTBOOK = "textbook"
    ASSESSMENT_FRAMEWORK = "assessment_framework"
    RESEARCH = "research"
    TEACHING_RESOURCE = "teaching_resource"


class EvidenceBasis(StrEnum):
    OPEN_LICENSE = "open_license"
    OFFICIAL_PUBLICATION = "official_publication"
    RESTRICTION = "restriction"


class LicenseEvidence(Contract):
    """Trusted explicit review, tied to one document; not a keyword/LLM guess."""

    document_url: SourceUrl
    evidence_url: SourceUrl
    license: LicenseLabel
    basis: EvidenceBasis
    statement: Text
    reviewed_by: Text
    reviewed_at: AwareDatetime
    verification_method: Literal["explicit_review", "allowlist"] = "explicit_review"

    @model_validator(mode="after")
    def consistent_basis(self) -> Self:
        required = {
            LicenseLabel.UNKNOWN: None,
            LicenseLabel.RESTRICTED: EvidenceBasis.RESTRICTION,
            LicenseLabel.OFFICIAL_PUBLICATION: EvidenceBasis.OFFICIAL_PUBLICATION,
        }.get(self.license, EvidenceBasis.OPEN_LICENSE)
        if self.basis != required:
            raise ValueError("licence label is inconsistent with review basis")
        return self


class SourceDocument(Entity):
    schema_version: Literal["source-document/1.0"] = "source-document/1.0"
    identity_kind: ClassVar[str] = "source"
    url: SourceUrl
    title: Text
    license: LicenseLabel = LicenseLabel.UNKNOWN
    license_evidence: LicenseEvidence | None = None
    sha256: Digest | None = None
    bytes: Minutes | None = None
    downloaded_at: AwareDatetime | None = None
    country: Annotated[str, Field(pattern=r"^(?:[A-Z]{2}|INT)$")]
    grades: tuple[Annotated[int, Field(strict=True, ge=1, le=12)], ...] = ()
    type: DocumentType
    license_gate_status: GateStatus = GateStatus.PENDING
    rejection_reason: Text | None = None
    checked_at: AwareDatetime | None = None
    policy_version: Text | None = None

    @model_validator(mode="after")
    def lifecycle(self) -> Self:
        if len(set(self.grades)) != len(self.grades):
            raise ValueError("duplicate source grades")
        object.__setattr__(self, "grades", tuple(sorted(self.grades)))
        fetched = (self.sha256 is not None, self.bytes is not None, self.downloaded_at is not None)
        if any(fetched) and not all(fetched):
            raise ValueError("sha256, bytes and downloaded_at must be supplied together")
        if self.license_evidence and self.license_evidence.document_url != self.url:
            raise ValueError("licence evidence belongs to a different document")
        self._validate_gate()
        return self

    def _validate_gate(self) -> None:
        if self.license_gate_status == GateStatus.PENDING:
            if self.checked_at or self.policy_version or self.rejection_reason or self.sha256:
                raise ValueError("pending records cannot carry gate results or downloaded bytes")
            return
        if self.checked_at is None or self.policy_version is None:
            raise ValueError("completed gate requires checked_at and policy_version")
        if self.license_gate_status == GateStatus.REJECTED:
            if not self.rejection_reason:
                raise ValueError("rejected records require a reason")
            return
        if self.rejection_reason or self.license in {LicenseLabel.UNKNOWN, LicenseLabel.RESTRICTED}:
            raise ValueError("accepted record has a rejected or unknown licence")
        if self.license_evidence is None or self.license_evidence.license != self.license:
            raise ValueError("accepted records require matching licence evidence")


class DomainPolicy(Contract):
    id: Slug
    publisher: Text
    country: Annotated[str, Field(pattern=r"^(?:[A-Z]{2}|INT)$")]
    domain: Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[.-][a-z0-9]+)*$")]
    include_subdomains: bool = False
    path_prefixes: tuple[Text, ...] = ("/",)
    document_types: tuple[DocumentType, ...]
    license: LicenseLabel
    admission: Literal["official_publication", "explicit_review"]
    evidence_url: SourceUrl
    retrieved_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    notes: Text

    @model_validator(mode="after")
    def coherent_policy(self) -> Self:
        if not self.path_prefixes or any(not p.startswith("/") for p in self.path_prefixes):
            raise ValueError("policy paths must be nonempty absolute URL prefixes")
        if not self.document_types:
            raise ValueError("policy must declare document types")
        if (
            self.admission == "official_publication"
            and self.license != LicenseLabel.OFFICIAL_PUBLICATION
        ):
            raise ValueError("official admission must not invent an open licence")
        if self.admission == "explicit_review" and self.license != LicenseLabel.UNKNOWN:
            raise ValueError("per-resource review must not declare a blanket domain licence")
        return self

    def matches(self, document: SourceDocument) -> bool:
        parts = urlsplit(str(document.url))
        host = parts.hostname or ""
        matched_host = host == self.domain or (
            self.include_subdomains and host.endswith("." + self.domain)
        )
        matched_path = any(
            prefix == "/" or parts.path == prefix or parts.path.startswith(prefix.rstrip("/") + "/")
            for prefix in self.path_prefixes
        )
        return (
            matched_host
            and matched_path
            and document.type in self.document_types
            and document.country == self.country
        )


class SourceAllowlist(Contract):
    schema_version: Literal["source-allowlist/1.0"] = "source-allowlist/1.0"
    policy_version: Text
    domains: tuple[DomainPolicy, ...]

    @model_validator(mode="after")
    def unique_rules(self) -> Self:
        unique_sorted(tuple(rule.id for rule in self.domains))
        return self
