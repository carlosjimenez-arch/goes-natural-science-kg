# src/goes_natural_science_kg/curriculum/editorial.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Record transparent editorial normalization without silently approving model content.
from goes_natural_science_kg.corpus.quotation import locate_quote
from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.release import EditorialCorrection, UnitAssignment, UnitDesign


def normalize_candidate(
    design: UnitDesign, assignment: UnitAssignment
) -> tuple[UnitDesign, tuple[EditorialCorrection, ...]]:
    """Recover literal source spans and authoritative routing metadata, never changed wording."""
    changes = []
    anchors = {a.key: a for a in assignment.anchors}
    skills = []
    for skill in design.skills:
        micros = []
        for micro in skill.micros:
            support = []
            for claim in micro.support:
                anchor = anchors.get(claim.anchor_key)
                location = locate_quote(anchor.text, claim.quote) if anchor else None
                if anchor and location and claim.quote != anchor.text[slice(*location)]:
                    revised = claim.model_copy(update={"quote": anchor.text[slice(*location)]})
                    changes.append(
                        EditorialCorrection(
                            unit_key=assignment.key,
                            entity_key=micro.key,
                            category="transport_normalization",
                            reason="Recovered the exact source span after whitespace, explicit PDF word-wrap or double-escaped line-break normalization; no words were fuzzy-matched.",
                            before_sha256=content_hash(claim),
                            after_sha256=content_hash(revised),
                        )
                    )
                    support.append(revised)
                else:
                    support.append(claim)
            micros.append(micro.model_copy(update={"support": tuple(support)}))
        skills.append(skill.model_copy(update={"micros": tuple(micros)}))
    revised_design = UnitDesign(
        **{**design.model_dump(), "unit_key": assignment.key, "skills": tuple(skills)}
    )
    if design.unit_key != assignment.key:
        changes.append(
            EditorialCorrection(
                unit_key=assignment.key,
                entity_key=design.unit_key,
                category="transport_normalization",
                reason="Restored the immutable assignment key; generators do not own routing identity.",
                before_sha256=content_hash(design.unit_key),
                after_sha256=content_hash(assignment.key),
            )
        )
    return revised_design, tuple(changes)
