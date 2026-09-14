# src/goes_natural_science_kg/eval/cases.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Version and measure evidence-grounded prompt artifacts.
from pathlib import Path

from goes_natural_science_kg.agents.checks import claims_for, materialize
from goes_natural_science_kg.schemas.orchestration import (
    Assignment,
    ClaimVerdict,
    CurriculumProposal,
    Decomposition,
    JudgeRole,
    UnitProposal,
    Verdict,
    WorkItem,
)
from goes_natural_science_kg.schemas.prompt_evaluation import AnnotatedCase

ROLES = (
    "decomposition",
    "curricular_judge",
    "cognitive_judge",
    "graph_judge",
    "evidence_judge",
    "age_appropriateness_judge",
    "curricularization",
)


def load_cases(path: Path) -> tuple[AnnotatedCase, ...]:
    cases = tuple(
        AnnotatedCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    return cases


def work_item(case: AnnotatedCase) -> WorkItem:
    return WorkItem(
        id=case.id,
        level="L2",
        instructions="Decompose the supplied skill into atomic tasks.",
        assignments=(
            Assignment(
                skill_id=case.skill.skill.id,
                grade=case.skill.suggested_grade,
                domain=case.skill.domain,
                minutes=300,
                instructions="Use the cited source evidence.",
            ),
        ),
    )


def curriculum_task(case: AnnotatedCase) -> WorkItem:
    task = work_item(case)
    return task.model_copy(
        update={
            "level": "L3",
            "grade": case.skill.suggested_grade,
            "micros": materialize(task, case.expected),
            "budget_minutes": 9600,
        }
    )


def expected_curriculum(case: AnnotatedCase) -> CurriculumProposal:
    micros = materialize(work_item(case), case.expected)
    indexed = {m.skill.identity.key: m for m in micros}
    units = []
    for draft in case.expected.micros:
        micro = indexed[draft.key]
        units.append(
            UnitProposal(
                key=draft.key,
                label=draft.knowledge_object,
                micro_skill_ids=(micro.skill.id,),
                contexts=({"type": "everyday", "domain": "environmental"},),
                contextualized_task="In an El Salvador classroom, "
                + micro.task[0].lower()
                + micro.task[1:],
                instruction_minutes=draft.estimated_minutes - 5,
                assessment_minutes=5,
                review_minutes=5,
                setup_minutes=5,
            )
        )
    return CurriculumProposal(
        units=tuple(units),
        rationale="Preserve each atomic task and prerequisite; add classroom context only here.",
        source_refs=case.skill.source_refs,
    )


def judge_candidate(case: AnnotatedCase, role: str, negative: bool) -> Decomposition:
    if not negative:
        return case.expected
    data = case.expected.model_dump(mode="json")
    first = data["micros"][0]
    if role == "curricular_judge":
        first["knowledge_object"] = "campaign finance law unrelated to the science skill"
        first["task"] = "Compare political campaign finance systems."
    elif role == "cognitive_judge":
        first.update(
            observable_verb="knows",
            knowledge_object=case.skill.skill.label,
            cognitive_domain="reasoning",
            task="Understand the topic.",
            evidence_of_mastery="Understands the topic.",
        )
    elif role == "graph_judge":
        first["prerequisites"] = [{"id": data["micros"][-1]["key"], "type": "PREREQUISITE"}]
    elif role == "evidence_judge":
        first["knowledge_object"] = "the Moon producing its own sunlight"
        first["task"] = "Explain why the Moon produces its own sunlight."
    elif role == "age_appropriateness_judge":
        first["task"] = (
            "Independently derive and solve nonlinear differential equations with no scaffolding."
        )
    return Decomposition.model_validate(data)


def case_payload(case: AnnotatedCase, role: str, negative: bool = False) -> dict[str, object]:
    evidence = [p.model_dump(mode="json") for p in case.evidence]
    task = work_item(case)
    payload: dict[str, object] = {
        "case_id": case.id,
        "skill": case.skill.model_dump(mode="json"),
        "evidence": evidence,
        "task": task.model_dump(mode="json"),
    }
    if role == "curricularization":
        payload["task"] = curriculum_task(case).model_dump(mode="json")
    elif role != "decomposition":
        candidate = judge_candidate(case, role, negative)
        # Cyclic draft is deliberately retained for the graph judge.
        micros = materialize(task, candidate)
        payload["candidate"] = candidate.model_dump(mode="json")
        payload["micros"] = [m.model_dump(mode="json") for m in micros]
        payload["claims"] = [c.model_dump(mode="json") for c in claims_for(candidate, micros)]
    return payload


def expected_output(
    case: AnnotatedCase, role: str, negative: bool = False
) -> Decomposition | CurriculumProposal | Verdict:
    if role == "decomposition":
        return case.expected
    if role == "curricularization":
        return expected_curriculum(case)
    checks = []
    if role == "evidence_judge":
        candidate = judge_candidate(case, role, negative)
        for claim in claims_for(candidate, materialize(work_item(case), candidate)):
            packet = case.evidence[0]
            anchor = max(packet.anchors, key=lambda a: len(a.text))
            supported = not (negative and "Moon" in claim.text)
            checks.append(
                ClaimVerdict(
                    claim_id=claim.id,
                    supported=supported,
                    source_ref=packet.chunk_id if supported else None,
                    paragraph_id=anchor.paragraph_id if supported else None,
                    quote=anchor.text if supported else "",
                    rationale="Editorial alignment/design proposal supported by the cited indicator; not an empirical claim."
                    if supported
                    else "No cited paragraph supports the unrelated claim.",
                )
            )
    return Verdict(
        role=JudgeRole(role),
        passed=not negative,
        score=0.0 if negative else 0.9,
        rationale="Reject the deliberately injected criterion-specific defect."
        if negative
        else "The candidate satisfies this criterion against the supplied source-backed editorial reference.",
        claims=tuple(checks),
    )
