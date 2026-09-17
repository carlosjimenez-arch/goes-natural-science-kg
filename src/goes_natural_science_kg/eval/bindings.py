# src/goes_natural_science_kg/eval/bindings.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Independently verify published quote hashes against parsed source offsets.
import hashlib
from collections.abc import Mapping

from goes_natural_science_kg.schemas.ingestion import EvidenceChunk, ParsedDocument
from goes_natural_science_kg.schemas.release import BindingAudit, PublishedBinding


def audit_bindings(
    bindings: tuple[PublishedBinding, ...],
    documents: Mapping[str, ParsedDocument],
    chunks: Mapping[str, EvidenceChunk],
    source_hashes: Mapping[str, str],
) -> BindingAudit:
    errors = []
    for binding in bindings:
        document = documents.get(binding.source_id)
        chunk = chunks.get(binding.chunk_id)
        if document is None or chunk is None:
            errors.append(binding.entity_id + ": missing document or chunk")
            continue
        paragraph = next((p for p in document.paragraphs if p.id == binding.paragraph_id), None)
        if (
            document.document_sha256 != source_hashes.get(binding.source_id)
            or chunk.source_id != binding.source_id
            or binding.paragraph_id not in chunk.paragraph_ids
            or paragraph is None
            or paragraph.page != binding.page
            or not paragraph.start <= binding.quote_start < binding.quote_end <= paragraph.end
        ):
            errors.append(binding.entity_id + ": inconsistent source location")
            continue
        quote = document.text[binding.quote_start : binding.quote_end]
        if hashlib.sha256(quote.encode()).hexdigest() != binding.quote_sha256:
            errors.append(binding.entity_id + ": quotation hash mismatch")
    return BindingAudit(
        checked=len(bindings), source_sha256=dict(source_hashes), errors=tuple(errors)
    )
