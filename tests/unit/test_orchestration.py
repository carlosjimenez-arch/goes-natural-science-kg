# tests/unit/test_orchestration.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_natural_science_kg.agents.checks import (
    aggregate,
    curriculum_errors,
    evidence_errors,
    graph_errors,
    materialize,
)
from goes_natural_science_kg.schemas.agent_state import merge_keyed
from goes_natural_science_kg.schemas.base import canonical_json
from goes_natural_science_kg.schemas.curriculum import ContextTag
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    Assignment,
    Claim,
    ClaimVerdict,
    CurriculumProposal,
    Decomposition,
    Domain,
    DomainPlan,
    EvidencePacket,
    JudgeRole,
    LoopPolicy,
    MicroProposal,
    OrchestrationInput,
    OrchestrationSettings,
    RootPlan,
    UnitProposal,
    Verdict,
    WorkItem,
)
from goes_natural_science_kg.schemas.skills import EdgeType, GradeBand, PrerequisiteRef

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/golden/orchestration"


def case():
    request = OrchestrationInput.model_validate_json((GOLDEN / "input.json").read_bytes())
    runtime = AgentRuntime(
        request=request,
        settings=OrchestrationSettings(project="test-project", seed=0),
        evidence=tuple(
            EvidencePacket.model_validate(p)
            for p in json.loads((GOLDEN / "evidence.json").read_text())
        ),
        cache=GOLDEN / "cassettes",
        prompts=ROOT / "prompts",
    )
    task = WorkItem(
        id="test",
        level="L2",
        instructions="Contract test",
        assignments=(
            Assignment(
                skill_id=request.skills[0].skill.id,
                grade=2,
                domain=Domain.PHYSICAL,
                minutes=100,
                instructions="Contract test",
            ),
        ),
    )
    draft = MicroProposal(
        key="one",
        observable_verb="Compare",
        knowledge_object="liquid properties",
        cognitive_domain="applying",
        grade_band=GradeBand(minimum=2, maximum=3),
        estimated_minutes=10,
        evidence_of_mastery="Record two observations",
        source_refs=request.skills[0].source_refs,
        confidence=0.8,
        task="Compare two liquid samples",
        atomicity_reason="One comparison task",
    )
    return runtime, task, draft


def votes():
    return tuple(
        Verdict(role=r, score=0.9, passed=True, rationale="Voting rule contract case")
        for r in JudgeRole
    )


def test_voting_quorum_mandatory_roles_and_threshold():
    baseline = votes()
    assert aggregate(baseline, (), 0, LoopPolicy()).accepted
    for role in JudgeRole:
        changed = tuple(
            v.model_copy(update={"score": 0.79}) if v.role == role else v for v in baseline
        )
        assert aggregate(changed, (), 0, LoopPolicy()).accepted == (
            role not in {JudgeRole.GRAPH, JudgeRole.EVIDENCE}
        )
    assert not aggregate(baseline[:-1], (), 0, LoopPolicy()).accepted
    assert not aggregate(baseline, ("cycle",), 0, LoopPolicy()).accepted
    assert not aggregate((*baseline[:-1], baseline[0]), (), 0, LoopPolicy()).accepted
    with pytest.raises(ValidationError):
        LoopPolicy(max_revisions=4)


def test_judges_cannot_emit_content_and_coordinators_cannot_emit_micros():
    with pytest.raises(ValidationError):
        Verdict(**votes()[0].model_dump(), replacement_content="rewrite")
    with pytest.raises(ValidationError):
        RootPlan(assignments=(), rationale="Plan", source_refs=(), micros=[])
    with pytest.raises(ValidationError):
        DomainPlan(workers=(), rationale="Plan", source_refs=(), contexts=[])


def test_reducer_is_commutative_idempotent_and_rejects_conflicts():
    assert merge_keyed({"a": 1}, {"b": 2}) == merge_keyed({"b": 2}, {"a": 1})
    assert merge_keyed({"a": 1}, {"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        merge_keyed({"a": 1}, {"a": 2})


def test_graph_cycle_orphan_external_and_duplicate_checks():
    _, task, draft = case()
    micros = materialize(
        task,
        Decomposition(
            micros=(
                draft,
                draft.model_copy(
                    update={"key": "two", "knowledge_object": "other", "task": "Other task"}
                ),
            ),
            granularity_rationale="Test contract",
        ),
    )
    parents = {task.assignments[0].skill_id}
    assert graph_errors(micros, parents) == ()
    assert "orphan parent" in graph_errors(micros, set())
    assert "duplicate micro-skill IDs" in graph_errors(micros + micros, parents)
    cycle = tuple(
        m.model_copy(
            update={
                "skill": m.skill.model_copy(
                    update={
                        "prerequisites": (
                            PrerequisiteRef(id=micros[1 - i].skill.id, type=EdgeType.PREREQUISITE),
                        )
                    }
                )
            }
        )
        for i, m in enumerate(micros)
    )
    assert "prerequisite cycle" in graph_errors(cycle, parents)
    assert (
        canonical_json(
            materialize(task, Decomposition(micros=(draft,), granularity_rationale="Atomic"))[
                0
            ].skill
        ).find("contexts")
        == -1
    )


def test_evidence_rejects_irrelevant_or_fabricated_citations():
    runtime, _, _ = case()
    packet = runtime.evidence[0]
    anchor = packet.anchors[0]
    claim = Claim(
        id="x", text="Contract assertion", source_refs=(packet.chunk_id,), kind="alignment"
    )
    check = ClaimVerdict(
        claim_id="x",
        supported=True,
        source_ref=packet.chunk_id,
        paragraph_id=anchor.paragraph_id,
        quote=anchor.text[:30],
        rationale="Contract quotation case",
    )
    verdict = Verdict(
        role=JudgeRole.EVIDENCE, score=0.9, passed=True, rationale="Contract case", claims=(check,)
    )
    assert not evidence_errors(runtime, (claim,), verdict)
    assert evidence_errors(runtime, (claim,), verdict.model_copy(update={"claims": ()}))
    assert evidence_errors(
        runtime,
        (claim,),
        verdict.model_copy(
            update={"claims": (check.model_copy(update={"quote": "invented text"}),)}
        ),
    )
    assert evidence_errors(runtime, (claim.model_copy(update={"kind": "scientific"}),), verdict)


def test_curriculum_enforces_budget_coverage_and_preserves_assigned_grade():
    _, task, draft = case()
    micros = materialize(
        task, Decomposition(micros=(draft,), granularity_rationale="One atomic task")
    )
    l3 = task.model_copy(update={"level": "L3", "grade": 2, "micros": micros, "budget_minutes": 20})
    unit = UnitProposal(
        key="one",
        label="Unit",
        micro_skill_ids=(micros[0].skill.id,),
        contexts=(ContextTag(type="everyday", domain="environmental"),),
        contextualized_task="Compare water samples",
        instruction_minutes=8,
        assessment_minutes=2,
        review_minutes=0,
        setup_minutes=0,
    )
    proposal = CurriculumProposal(
        units=(unit,), rationale="Alignment", source_refs=draft.source_refs
    )
    assert not curriculum_errors(l3, proposal)
    assert curriculum_errors(l3, proposal.model_copy(update={"units": ()}))
    assert curriculum_errors(
        l3, proposal.model_copy(update={"units": (unit.model_copy(update={"setup_minutes": 30}),)})
    )
    assert curriculum_errors(
        l3,
        proposal.model_copy(
            update={"units": (unit.model_copy(update={"instruction_minutes": 1}),)}
        ),
    )
    # A suggested broad grade band never duplicates a skill across grade sequences.
    assert not curriculum_errors(
        l3.model_copy(update={"grade": 3}), proposal.model_copy(update={"units": ()})
    )


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L3"])
def test_every_level_stops_after_three_revisions(level):
    from goes_natural_science_kg.agents.hierarchy import finish, loop_route

    state = {
        "task": WorkItem(id="boundary/" + level, level=level, instructions="Boundary contract"),
        "candidate": None,
        "micros": (),
        "ballots": {},
        "accepted": False,
        "expected_children": (),
    }
    for revision in range(4):
        state["revision"] = revision
        state["ballots"][str(revision)] = aggregate(
            votes(), ("Unresolved issue",), revision, LoopPolicy()
        )
        assert loop_route(state, 3) == ("revise" if revision < 3 else "finish")
    outcome = finish(state)["outcomes"][state["task"].id]
    assert outcome.status == "needs_human_review" and outcome.revisions == 3
    assert len(outcome.ballots) == 4
    state["accepted"] = True
    state["revision"] = 0
    assert loop_route(state, 3) == "finish"


def test_local_prerequisite_keys_resolve_to_stable_ids():
    from goes_natural_science_kg.schemas.orchestration import ProposalPrerequisite

    _, task, draft = case()
    second = draft.model_copy(
        update={
            "key": "two",
            "task": "Compare observations",
            "knowledge_object": "observations",
            "prerequisites": (ProposalPrerequisite(id="one", type=EdgeType.PREREQUISITE),),
        }
    )
    result = materialize(
        task, Decomposition(micros=(draft, second), granularity_rationale="Two atomic tasks")
    )
    first = next(m for m in result if m.skill.identity.key == "one")
    dependent = next(m for m in result if m.skill.identity.key == "two")
    assert dependent.skill.prerequisites[0].id == first.skill.id
    reversed_result = materialize(
        task, Decomposition(micros=(second, draft), granularity_rationale="Two atomic tasks")
    )
    assert result == reversed_result


def test_quote_layout_normalization_preserves_original_offsets():
    from goes_natural_science_kg.agents.checks import verified_quote_span

    text = "Before: observable properties\n of liquids. After."
    start, end = verified_quote_span(text, "observable properties of liquids.")
    assert text[start:end] == "observable properties\n of liquids."
    assert verified_quote_span(text, "invisible properties of liquids.") is None
    assert verified_quote_span("same same", "same") is None


def test_parent_cannot_reset_terminal_child_revision_budget():
    from goes_natural_science_kg.agents.hierarchy import child_items, finish
    from goes_natural_science_kg.schemas.orchestration import WorkerDirective

    _, task, _ = case()
    failure = {
        "task": task,
        "candidate": None,
        "micros": (),
        "accepted": False,
        "revision": 3,
        "ballots": {str(i): aggregate(votes(), ("Unresolved",), i, LoopPolicy()) for i in range(4)},
        "expected_children": (),
    }
    outcome = finish(failure)["outcomes"][task.id]
    parent = {
        "task": task.model_copy(update={"id": "domain", "level": "L1"}),
        "revision": 1,
        "candidate": DomainPlan(
            workers=(
                WorkerDirective(
                    skill_id=task.assignments[0].skill_id, instructions="Changed instruction"
                ),
            ),
            rationale="Plan",
            source_refs=(),
        ),
        "outcomes": {outcome.item_id: outcome},
        "errors": (),
    }
    assert child_items(parent)[0].id == outcome.item_id


def test_dynamic_domain_and_skill_fanout_depends_on_input():
    from goes_natural_science_kg.agents.hierarchy import child_items
    from goes_natural_science_kg.schemas.orchestration import WorkerDirective

    _, task, _ = case()
    one = task.assignments[0]
    two = one.model_copy(
        update={"skill_id": "skill-second-0123456789abcdef", "domain": Domain.LIFE}
    )
    root = {
        "task": task.model_copy(update={"level": "L0", "id": "root"}),
        "revision": 0,
        "candidate": RootPlan(assignments=(one, two), rationale="Plan", source_refs=()),
        "errors": (),
    }
    children = child_items(root)
    assert len(children) == 2 and {c.assignments[0].domain for c in children} == {
        Domain.PHYSICAL,
        Domain.LIFE,
    }
    domain = {
        "task": task.model_copy(update={"level": "L1", "id": "team", "assignments": (one, two)}),
        "revision": 0,
        "candidate": DomainPlan(
            workers=tuple(
                WorkerDirective(skill_id=a.skill_id, instructions="Decompose") for a in (one, two)
            ),
            rationale="Plan",
            source_refs=(),
        ),
        "errors": (),
    }
    assert len(child_items(domain)) == 2
