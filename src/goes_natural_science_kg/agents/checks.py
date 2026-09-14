# src/goes_natural_science_kg/agents/checks.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]

from goes_natural_science_kg.corpus.normalize import normalize
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.curriculum import CurriculumUnit, GradeCurriculum
from goes_natural_science_kg.schemas.identity import Identity, stable_id
from goes_natural_science_kg.schemas.ingestion import EvidenceReadyMicroSkill
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    Ballot,
    Candidate,
    Claim,
    CurriculumProposal,
    Decomposition,
    DomainPlan,
    EvaluableMicro,
    JudgeRole,
    LoopPolicy,
    RootPlan,
    Verdict,
    WorkItem,
)
from goes_natural_science_kg.schemas.skills import EdgeType


def materialize(task: WorkItem, candidate: Decomposition) -> tuple[EvaluableMicro, ...]:
    parent = task.assignments[0].skill_id
    identities = {
        d.key: Identity(namespace=parent.removeprefix("skill-")[:80].rstrip("-"), key=d.key)
        for d in candidate.micros
    }
    if len(identities) != len(candidate.micros):
        raise ValueError("duplicate editorial micro-skill keys")
    ids = {key: stable_id("micro", identity) for key, identity in identities.items()}
    results = []
    for draft in candidate.micros:
        values = draft.model_dump(exclude={"key", "task", "atomicity_reason", "prerequisites"})
        prerequisites = [{"id": ids.get(p.id, p.id), "type": p.type} for p in draft.prerequisites]
        skill = EvidenceReadyMicroSkill(
            id=ids[draft.key],
            identity=identities[draft.key],
            parent_skill_id=parent,
            prerequisites=prerequisites,
            **values,
        )
        results.append(
            EvaluableMicro(skill=skill, task=draft.task, atomicity_reason=draft.atomicity_reason)
        )
    return tuple(sorted(results, key=lambda m: m.skill.id))


def prerequisite_pairs(micros: tuple[EvaluableMicro, ...]) -> tuple[tuple[str, str, EdgeType], ...]:
    groups: dict[str, list[str]] = {}
    for micro in micros:
        groups.setdefault(micro.skill.parent_skill_id, []).append(micro.skill.id)
    return tuple(
        (source, m.skill.id, p.type)
        for m in micros
        for p in m.skill.prerequisites
        for source in groups.get(p.id, [p.id])
    )


def graph_errors(
    micros: tuple[EvaluableMicro, ...], parents: set[str], *, allow_external: bool = False
) -> tuple[str, ...]:
    ids = [m.skill.id for m in micros]
    known = set(ids)
    errors = []
    if len(known) != len(ids):
        errors.append("duplicate micro-skill IDs")
    if any(m.skill.parent_skill_id not in parents for m in micros):
        errors.append("orphan parent")
    signatures = [
        (m.skill.observable_verb.casefold(), m.skill.knowledge_object.casefold(), m.task.casefold())
        for m in micros
    ]
    if len(signatures) != len(set(signatures)):
        errors.append("duplicate observable tasks")
    pairs = prerequisite_pairs(micros)
    if any(a not in known and not (allow_external and a in parents) for a, _, _ in pairs):
        errors.append("unresolved prerequisite")
    pos = {key: i for i, key in enumerate(sorted(known))}
    directed = [(pos[a], pos[b]) for a, b, t in pairs if a in pos and t == EdgeType.PREREQUISITE]
    if any(a == b for a, b in directed):
        errors.append("self dependency")
    if directed:
        rows, cols = zip(*directed, strict=True)
        matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pos), len(pos)))
        _, labels = connected_components(matrix, directed=True, connection="strong")
        if np.any(np.bincount(labels) > 1):
            errors.append("prerequisite cycle")
    return tuple(errors)


def planning_errors(runtime: AgentRuntime, task: WorkItem, candidate: Candidate) -> tuple[str, ...]:
    errors = []
    if isinstance(candidate, RootPlan):
        expected = {s.skill.id: s for s in runtime.request.skills}
        ids = [a.skill_id for a in candidate.assignments]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            errors.append("root plan must cover every input exactly once")
        for a in candidate.assignments:
            if a.skill_id in expected and a.domain != expected[a.skill_id].domain:
                errors.append("unreviewed domain reassignment")
            if a.grade not in runtime.request.budget_minutes:
                errors.append("unbudgeted grade")
        for grade, budget in runtime.request.budget_minutes.items():
            if sum(a.minutes for a in candidate.assignments if a.grade == grade) > budget:
                errors.append("root budget exceeded")
    elif isinstance(candidate, DomainPlan):
        ids = [w.skill_id for w in candidate.workers]
        if len(ids) != len(set(ids)) or set(ids) != {a.skill_id for a in task.assignments}:
            errors.append("domain plan coverage mismatch")
    return tuple(errors)


def compile_curriculum(task: WorkItem, proposal: CurriculumProposal) -> GradeCurriculum:
    if task.grade is None:
        raise ValueError("missing curriculum grade")
    units = []
    for draft in proposal.units:
        identity = Identity(namespace=f"curriculum-grade-{task.grade}", key=draft.key)
        values = draft.model_dump(exclude={"key", "contextualized_task"})
        total = (
            draft.instruction_minutes
            + draft.assessment_minutes
            + draft.review_minutes
            + draft.setup_minutes
        )
        units.append(
            CurriculumUnit(
                id=stable_id("unit", identity), identity=identity, total_minutes=total, **values
            )
        )
    identity = Identity(namespace="curriculum", key=f"grade-{task.grade}")
    return GradeCurriculum(
        id=stable_id("grade", identity),
        identity=identity,
        grade=task.grade,
        units=tuple(units),
        total_minutes=sum(u.total_minutes for u in units),
        budget_minutes=task.budget_minutes,
    )


def curriculum_errors(task: WorkItem, proposal: CurriculumProposal) -> tuple[str, ...]:
    try:
        compile_curriculum(task, proposal)
    except ValueError as error:
        return (str(error),)
    assigned = {a.skill_id: a.grade for a in task.assignments}
    expected = {
        m.skill.id: m for m in task.micros if assigned.get(m.skill.parent_skill_id) == task.grade
    }
    actual = [key for u in proposal.units for key in u.micro_skill_ids]
    errors = []
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        errors.append("curriculum coverage mismatch")
    positions = {key: i for i, u in enumerate(proposal.units) for key in u.micro_skill_ids}
    for u in proposal.units:
        required = sum(
            expected[k].skill.estimated_minutes for k in u.micro_skill_ids if k in expected
        )
        if u.instruction_minutes + u.assessment_minutes < required:
            errors.append("underallocated observable task time")
    errors.extend(sequence_errors(task, positions))
    return tuple(errors)


def sequence_errors(task: WorkItem, positions: dict[str, int]) -> tuple[str, ...]:
    errors = []
    assigned = {a.skill_id: a.grade for a in task.assignments}
    for a, b, kind in prerequisite_pairs(task.micros):
        if b not in positions:
            continue
        if a not in positions:
            prior = next((m for m in task.micros if m.skill.id == a), None)
            if (
                prior is None
                or task.grade is None
                or assigned.get(prior.skill.parent_skill_id, 7) >= task.grade
            ):
                errors.append("missing or backward-grade prerequisite")
        elif kind == EdgeType.PREREQUISITE and positions[a] >= positions[b]:
            errors.append("prerequisite must precede dependent unit")
        elif kind == EdgeType.CO_REQUISITE and positions[a] != positions[b]:
            errors.append("co-requisites require one shared unit")
    return tuple(errors)


def claims_for(
    candidate: Candidate | None, micros: tuple[EvaluableMicro, ...]
) -> tuple[Claim, ...]:
    claims = []
    if isinstance(candidate, (RootPlan, DomainPlan, CurriculumProposal)):
        claims.append(
            Claim(
                id="plan",
                text=candidate.rationale,
                source_refs=candidate.source_refs,
                kind="alignment",
            )
        )
    for m in micros:
        for name in ("knowledge_object", "evidence_of_mastery", "misconceptions"):
            value = getattr(m.skill, name)
            if value:
                claims.append(
                    Claim(
                        id=m.skill.id + "/" + name,
                        text=canonical_json(value),
                        source_refs=m.skill.source_refs,
                        kind="scientific" if name == "misconceptions" else "alignment",
                    )
                )
        claims.append(
            Claim(
                id=m.skill.id + "/task", text=m.task, source_refs=m.skill.source_refs, kind="design"
            )
        )
    if isinstance(candidate, CurriculumProposal):
        for u in candidate.units:
            claims.append(
                Claim(
                    id="unit/" + u.key,
                    text=u.contextualized_task,
                    source_refs=candidate.source_refs,
                    kind="design",
                )
            )
    return tuple(claims)


def evidence_errors(
    runtime: AgentRuntime, claims: tuple[Claim, ...], verdict: Verdict
) -> tuple[str, ...]:
    errors = []
    expected = {c.id: c for c in claims}
    if len(verdict.claims) != len(expected) or {c.claim_id for c in verdict.claims} != set(
        expected
    ):
        return ("evidence judge did not inspect every claim",)
    packets = {p.chunk_id: p for p in runtime.evidence}
    for check in verdict.claims:
        claim = expected[check.claim_id]
        packet = packets.get(check.source_ref or "")
        if not check.supported or packet is None or check.source_ref not in claim.source_refs:
            errors.append("unsupported claim: " + check.claim_id)
            continue
        anchor = next((a for a in packet.anchors if a.paragraph_id == check.paragraph_id), None)
        if anchor is None or verified_quote_span(anchor.text, check.quote) is None:
            errors.append("unverified quotation: " + check.claim_id)
        if claim.kind == "scientific" and packet.source_type == "curriculum":
            errors.append("curriculum alone cannot ground a scientific misconception claim")
    return tuple(errors)


def aggregate(
    verdicts: tuple[Verdict, ...], errors: tuple[str, ...], revision: int, policy: LoopPolicy
) -> Ballot:
    by_role = {v.role: v for v in verdicts}
    complete = len(verdicts) == 5 and set(by_role) == set(JudgeRole)
    passing = {r for r, v in by_role.items() if v.passed and v.score >= policy.score_threshold}
    accepted = (
        complete
        and not errors
        and len(passing) >= policy.minimum_votes
        and {JudgeRole.GRAPH, JudgeRole.EVIDENCE} <= passing
    )
    return Ballot(
        revision=revision,
        verdicts=tuple(sorted(verdicts, key=lambda v: v.role)),
        hard_errors=errors,
        accepted=accepted,
        votes=len(passing),
    )


def verified_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    """Permit whitespace-only PDF layout differences and recover the exact original span."""
    if not quote.strip():
        return None
    source = normalize(text)
    needle = normalize(quote).text.strip()
    start = source.text.find(needle)
    if start < 0 or source.text.find(needle, start + 1) >= 0:
        return None
    return source.original_spans[start][0], source.original_spans[start + len(needle) - 1][1]
