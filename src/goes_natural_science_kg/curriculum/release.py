# src/goes_natural_science_kg/curriculum/release.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Compile contextualized activities separately from auditable skill archetypes.
from __future__ import annotations

import hashlib
from functools import lru_cache

from goes_natural_science_kg.schemas.base import content_hash
from goes_natural_science_kg.schemas.graph import EvidenceGraphSnapshot
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.ingestion import EvidenceReadyMicroSkill
from goes_natural_science_kg.schemas.release import (
    PedagogyRecord,
    ProposalChange,
    ProposalDiff,
    PublishedBinding,
    ReleaseUnitResult,
)
from goes_natural_science_kg.schemas.sequencing import (
    LearningActivity,
    LocalContext,
    SequencingInput,
)
from goes_natural_science_kg.schemas.skills import (
    Edge,
    EdgeType,
    FrameworkRef,
    GradeBand,
    PrerequisiteRef,
    Skill,
)


@lru_cache(maxsize=8192)
def entity_identity(unit: str, local: str) -> Identity:
    payload = {"unit": unit, "local": local}
    key = (unit + "-" + local)[:60].rstrip("-") + "-" + content_hash(payload)[:12]
    return Identity(namespace="mined-proposal", key=key)


def compile_records(  # noqa: C901 - preserve all cross-contract references in one transaction
    results: tuple[ReleaseUnitResult, ...],
    contexts: tuple[LocalContext, ...],
    version: str,
    *,
    initial: bool = False,
) -> tuple[
    EvidenceGraphSnapshot,
    tuple[LearningActivity, ...],
    tuple[str, ...],
    tuple[PedagogyRecord, ...],
    tuple[PublishedBinding, ...],
]:
    skills = []
    micros = []
    edges = []
    activities = []
    pedagogy = []
    bindings = []
    chunk_sources = {}
    key_micro_ids = []
    for result in sorted(results, key=lambda r: r.assignment.key):
        design = result.initial if initial else result.final
        if design is None:
            raise ValueError("missing unit design: " + result.assignment.key)
        assignment = result.assignment
        anchors = {a.key: a for a in assignment.anchors}
        micro_ids = {
            m.key: stable_id("micro", entity_identity(assignment.key, m.key))
            for s in design.skills
            for m in s.micros
        }
        panel = (
            result.panels[0]
            if initial and result.panels
            else result.panels[-1]
            if result.panels
            else None
        )
        confidence = (
            sum(
                v.passed
                and v.score >= 0.8
                and not any(i.severity in {"major", "critical"} for i in v.issues)
                for v in panel.verdicts
            )
            / 5
            if panel
            else 0.0
        )
        for skill in design.skills:
            identity = entity_identity(assignment.key, skill.key)
            parent = Skill(
                id=stable_id("skill", identity),
                identity=identity,
                label=skill.label,
                origin_framework=FrameworkRef(
                    name="MINED official textbook-derived proposal",
                    version="2026",
                    source_code=assignment.key,
                ),
                domain=skill.domain,
                hierarchy_level=0,
            )
            skills.append(parent)
            for micro in skill.micros:
                refs = []
                for support in micro.support:
                    if support.anchor_key not in anchors:
                        raise ValueError("unresolved evidence anchor: " + support.anchor_key)
                    anchor = anchors[support.anchor_key]
                    offset = anchor.text.find(support.quote)
                    if offset < 0:
                        raise ValueError("quotation is absent from source: " + micro.key)
                    refs.append(anchor.chunk_id)
                    chunk_sources[anchor.chunk_id] = anchor.source_id
                    bindings.append(
                        PublishedBinding(
                            entity_id=micro_ids[micro.key],
                            source_id=anchor.source_id,
                            source_url=anchor.source_url,
                            chunk_id=anchor.chunk_id,
                            page=anchor.page,
                            paragraph_id=anchor.paragraph_id,
                            quote_start=anchor.start + offset,
                            quote_end=anchor.start + offset + len(support.quote),
                            quote_sha256=hashlib.sha256(support.quote.encode()).hexdigest(),
                            claim=support.claim,
                            location_verified=True,
                        )
                    )
                identity = entity_identity(assignment.key, micro.key)
                micros.append(
                    EvidenceReadyMicroSkill(
                        id=micro_ids[micro.key],
                        identity=identity,
                        parent_skill_id=parent.id,
                        observable_verb=micro.verb,
                        knowledge_object=micro.knowledge_object,
                        cognitive_domain=micro.cognitive_domain,
                        grade_band=GradeBand(minimum=assignment.grade, maximum=assignment.grade),
                        estimated_minutes=micro.estimated_minutes,
                        prerequisites=tuple(
                            PrerequisiteRef(id=micro_ids[k], type=EdgeType.PREREQUISITE)
                            for k in micro.prerequisites
                        ),
                        misconceptions=micro.misconceptions,
                        evidence_of_mastery=micro.mastery,
                        source_refs=tuple(sorted(set(refs))),
                        confidence=confidence,
                    )
                )
                if micro.key_for_retrieval:
                    key_micro_ids.append(micro_ids[micro.key])
                pedagogy.append(
                    PedagogyRecord(
                        entity_id=micro_ids[micro.key],
                        unit_key=assignment.key,
                        grade=assignment.grade,
                        assessment_task=micro.assessment_task,
                        expected_response=micro.expected_response,
                        scoring_rule=micro.scoring_rule,
                        confidence_basis="uncalibrated_unit_panel_vote_fraction",
                        limitations=design.limitations,
                    )
                )
                for prerequisite in micro.prerequisites:
                    identity = entity_identity(assignment.key, prerequisite + "-to-" + micro.key)
                    edges.append(
                        Edge(
                            id=stable_id("edge", identity),
                            identity=identity,
                            source=micro_ids[prerequisite],
                            target=micro_ids[micro.key],
                            type=EdgeType.PREREQUISITE,
                            strength=confidence,
                            justification="Proposed pedagogical dependency. Sources support the concepts, not an empirically measured necessity. "
                            + design.progression,
                            source_refs=tuple(
                                sorted({anchors[s.anchor_key].source_id for s in micro.support})
                            ),
                        )
                    )
        for activity in design.activities:
            activities.append(
                LearningActivity(
                    id=assignment.key + "-" + activity.key,
                    micro_skill_ids=tuple(micro_ids[k] for k in activity.micro_keys),
                    kind=activity.kind,
                    minutes=activity.minutes,
                    cognitive_domain=activity.cognitive_domain,
                    inquiry=activity.inquiry,
                    theme=activity.theme,
                    task=activity.task,
                    mastery_criterion=activity.mastery_criterion,
                    allowed_grades=(assignment.grade,),
                    context_tasks=activity.context_tasks,
                )
            )
    graph = EvidenceGraphSnapshot(
        version=version,
        skills=tuple(skills),
        micro_skills=tuple(micros),
        edges=tuple(edges),
        source_ids=tuple(sorted(set(chunk_sources.values()))),
        chunk_sources=chunk_sources,
    )
    return graph, tuple(activities), tuple(sorted(key_micro_ids)), tuple(pedagogy), tuple(bindings)


def compile_proposal(
    results: tuple[ReleaseUnitResult, ...],
    contexts: tuple[LocalContext, ...],
    version: str,
    *,
    initial: bool = False,
) -> tuple[SequencingInput, tuple[PedagogyRecord, ...], tuple[PublishedBinding, ...]]:
    graph, activities, key_micro_ids, pedagogy, bindings = compile_records(
        results, contexts, version, initial=initial
    )
    data = SequencingInput(
        graph=graph,
        approval="provisional",
        activities=activities,
        key_micro_ids=key_micro_ids,
        contexts=contexts,
    )
    return data, pedagogy, bindings


def proposal_diff(before: EvidenceGraphSnapshot, after: EvidenceGraphSnapshot) -> ProposalDiff:
    old = {e.id: e for e in (*before.skills, *before.micro_skills, *before.edges)}
    new = {e.id: e for e in (*after.skills, *after.micro_skills, *after.edges)}
    changes = []
    for key in sorted(old.keys() | new.keys()):
        if key not in old:
            changes.append(ProposalChange(entity_id=key, operation="added", changed_fields=()))
        elif key not in new:
            changes.append(ProposalChange(entity_id=key, operation="removed", changed_fields=()))
        else:
            left, right = old[key].model_dump(mode="json"), new[key].model_dump(mode="json")
            fields = tuple(
                k for k in sorted(left.keys() | right.keys()) if left.get(k) != right.get(k)
            )
            if fields:
                changes.append(
                    ProposalChange(entity_id=key, operation="modified", changed_fields=fields)
                )
    return ProposalDiff(
        before_sha256=content_hash(before), after_sha256=content_hash(after), changes=tuple(changes)
    )
