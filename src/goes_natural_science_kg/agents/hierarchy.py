# src/goes_natural_science_kg/agents/hierarchy.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Implement bounded, evidence-audited hierarchical orchestration.
from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from goes_natural_science_kg.agents.checks import (
    aggregate,
    claims_for,
    compile_curriculum,
    curriculum_errors,
    evidence_errors,
    graph_errors,
    materialize,
    planning_errors,
)
from goes_natural_science_kg.agents.transport import generate
from goes_natural_science_kg.schemas.agent_state import JudgeState, WorkInput, WorkOutput, WorkState
from goes_natural_science_kg.schemas.base import Contract, content_hash
from goes_natural_science_kg.schemas.orchestration import (
    AgentRuntime,
    CurriculumProposal,
    Decomposition,
    Domain,
    DomainPlan,
    EvaluableMicro,
    ItemOutcome,
    JudgeRole,
    RootPlan,
    Verdict,
    WorkItem,
)

Graph = CompiledStateGraph[Any, Any, Any, Any]
CONTRACTS: dict[str, type[Contract]] = {
    "L0": RootPlan,
    "L1": DomainPlan,
    "L2": Decomposition,
    "L3": CurriculumProposal,
}


def scope_evidence(runtime: AgentRuntime, state: WorkState) -> tuple[dict[str, Any], ...]:
    task = state["task"]
    ids = {a.skill_id for a in task.assignments}
    refs = {
        r for s in runtime.request.skills if not ids or s.skill.id in ids for r in s.source_refs
    }
    refs.update(r for m in state.get("micros", task.micros) for r in m.skill.source_refs)
    return tuple(p.model_dump(mode="json") for p in runtime.evidence if p.chunk_id in refs)


def model_payload(runtime: AgentRuntime, state: WorkState) -> dict[str, object]:
    task = state["task"]
    task_data = task.model_dump(mode="json", exclude={"id"})
    ids = {a.skill_id for a in task.assignments}
    inputs = [
        s.model_dump(mode="json") for s in runtime.request.skills if not ids or s.skill.id in ids
    ]
    candidate = state.get("candidate")
    ballots = tuple(b.model_dump(mode="json") for _, b in sorted(state.get("ballots", {}).items()))
    children = [
        state["outcomes"][key].model_dump(
            mode="json", exclude={"item_id", "children", "logical_key"}
        )
        for key in state.get("expected_children", ())
        if key in state.get("outcomes", {})
    ]
    return {
        "task": task_data,
        "input_skills": inputs,
        "budgets": runtime.request.budget_minutes,
        "candidate": candidate.model_dump(mode="json") if candidate else None,
        "revision": state.get("revision", 0),
        "feedback": ballots,
        "children": children,
        "micros": [m.model_dump(mode="json") for m in state.get("micros", task.micros)],
        "claims": [
            c.model_dump(mode="json")
            for c in claims_for(candidate, state.get("micros", task.micros))
        ],
        "evidence": scope_evidence(runtime, state),
    }


async def optimize(runtime: AgentRuntime, state: WorkState) -> dict[str, object]:
    try:
        candidate = await generate(
            runtime,
            "optimizer-" + state["task"].level.lower(),
            runtime.settings.optimizer_model,
            model_payload(runtime, state),
            CONTRACTS[state["task"].level],
        )
        return {"candidate": candidate, "errors": ()}
    except ValueError as error:
        return {"candidate": None, "errors": ("invalid optimizer output: " + str(error),)}


def proposed_children(state: WorkState) -> tuple[WorkItem, ...]:
    task = state["task"]
    prefix = task.id + f"/r{state['revision']}"
    candidate = state.get("candidate")
    if state.get("errors"):
        return ()
    if isinstance(candidate, RootPlan):
        return tuple(
            WorkItem(
                id=prefix + "/" + d.value,
                level="L1",
                assignments=tuple(a for a in candidate.assignments if a.domain == d),
                instructions="Decompose the assigned skills; preserve atomic observable tasks.",
            )
            for d in Domain
            if any(a.domain == d for a in candidate.assignments)
        )
    if isinstance(candidate, DomainPlan):
        assignments = {a.skill_id: a for a in task.assignments}
        return tuple(
            WorkItem(
                id=prefix + "/" + w.skill_id,
                level="L2",
                assignments=(assignments[w.skill_id],),
                instructions=w.instructions,
            )
            for w in candidate.workers
        )
    return ()


def logical_key(task: WorkItem) -> str:
    return content_hash(
        {
            "level": task.level,
            "skills": sorted(a.skill_id for a in task.assignments),
            "grade": task.grade,
        }
    )


def child_items(state: WorkState) -> tuple[WorkItem, ...]:
    """A terminal human-review child cannot be silently retried by a parent revision."""
    terminal = {
        o.logical_key: o.item_id
        for o in state.get("outcomes", {}).values()
        if o.status == "needs_human_review"
    }
    return tuple(
        c.model_copy(update={"id": terminal[logical_key(c)]}) if logical_key(c) in terminal else c
        for c in proposed_children(state)
    )


def prepare(runtime: AgentRuntime, state: WorkState) -> dict[str, object]:
    candidate = state.get("candidate")
    errors = list(state.get("errors", ()))
    if candidate is not None:
        errors.extend(planning_errors(runtime, state["task"], candidate))
    updated = cast(WorkState, {**state, "errors": tuple(errors)})
    return {"errors": tuple(errors), "expected_children": tuple(c.id for c in child_items(updated))}


def assemble(runtime: AgentRuntime, state: WorkState) -> dict[str, object]:
    task = state["task"]
    candidate = state.get("candidate")
    errors = list(state.get("errors", ()))
    micros = task.micros
    if isinstance(candidate, Decomposition):
        micros, worker_errors = worker_materialization(task, candidate)
        errors.extend(worker_errors)
    elif task.level in {"L0", "L1"}:
        children = [state["outcomes"][key] for key in state.get("expected_children", ())]
        micros = tuple(m for child in children if child.status == "approved" for m in child.micros)
        if any(child.status != "approved" for child in children):
            errors.append("child requires human review")
    if micros:
        errors.extend(
            graph_errors(
                micros,
                {s.skill.id for s in runtime.request.skills},
                allow_external=task.level in {"L1", "L2"},
            )
        )
    if task.level == "L0" and {m.skill.parent_skill_id for m in micros} != {
        s.skill.id for s in runtime.request.skills
    }:
        errors.append("approved decomposition does not cover input map")
    if isinstance(candidate, CurriculumProposal):
        errors.extend(curriculum_errors(task, candidate))
    known = {p.chunk_id for p in runtime.evidence}
    if any(
        not set(c.source_refs) <= known or not c.source_refs for c in claims_for(candidate, micros)
    ):
        errors.append("claim has missing or unregistered source references")
    return {"errors": tuple(sorted(set(errors))), "micros": micros}


async def judge(runtime: AgentRuntime, packet: JudgeState) -> dict[str, object]:
    state = packet["state"]
    role = JudgeRole(packet["role"])
    if state.get("candidate") is None:
        verdict = Verdict(
            role=role, score=0.0, passed=False, rationale="No schema-valid candidate to evaluate."
        )
    else:
        try:
            verdict = cast(
                Verdict,
                await generate(
                    runtime,
                    role.value,
                    runtime.settings.judge_model,
                    model_payload(runtime, state),
                    Verdict,
                ),
            )
            if verdict.role != role or (role != JudgeRole.EVIDENCE and verdict.claims):
                raise ValueError("judge emitted another role or out-of-scope evidence verdicts")
        except ValueError as error:
            verdict = Verdict(
                role=role, score=0.0, passed=False, rationale="Invalid judge output: " + str(error)
            )
    return {"verdicts": {f"{state['revision']}/{role.value}": verdict}}


def vote(runtime: AgentRuntime, state: WorkState) -> dict[str, object]:
    revision = state["revision"]
    verdicts = tuple(state["verdicts"][f"{revision}/{r.value}"] for r in JudgeRole)
    errors = list(state.get("errors", ()))
    evidence = next(v for v in verdicts if v.role == JudgeRole.EVIDENCE)
    errors.extend(
        evidence_errors(
            runtime, claims_for(state.get("candidate"), state.get("micros", ())), evidence
        )
    )
    ballot = aggregate(verdicts, tuple(sorted(set(errors))), revision, runtime.settings.policy)
    return {"accepted": ballot.accepted, "ballots": {str(revision): ballot}}


def finish(state: WorkState) -> dict[str, object]:
    candidate = state.get("candidate")
    task = state["task"]
    approved = state["accepted"]
    micros = state.get("micros", ())
    curriculum = (
        compile_curriculum(task, candidate)
        if approved and isinstance(candidate, CurriculumProposal)
        else None
    )
    outcome = ItemOutcome(
        item_id=task.id,
        logical_key=logical_key(task),
        level=task.level,
        status="approved" if approved else "needs_human_review",
        revisions=state["revision"],
        ballots=tuple(state["ballots"][str(i)] for i in range(state["revision"] + 1)),
        candidate=candidate,
        micros=micros,
        curriculum=curriculum,
        expansion_ratio=(
            len(micros) / len(candidate.assignments)
            if isinstance(candidate, RootPlan) and candidate.assignments
            else len(micros) / len(task.assignments)
            if task.assignments
            else None
        ),
        children=state.get("expected_children", ()),
    )
    return {"outcomes": {task.id: outcome}}


def build_level(
    runtime: AgentRuntime, level: str, children: dict[str, Graph] | None = None
) -> Graph:
    builder = StateGraph(WorkState, input_schema=WorkInput, output_schema=WorkOutput)

    async def proposer(state: WorkState) -> dict[str, object]:
        return await optimize(runtime, state)

    async def specialist(state: JudgeState) -> dict[str, object]:
        return await judge(runtime, state)

    def dispatch(state: WorkState) -> list[Send] | str:
        items = tuple(c for c in child_items(state) if c.id not in state.get("outcomes", {}))
        if not items:
            return "assemble"
        return [
            Send(
                "domain-" + i.assignments[0].domain.value if level == "L0" else "worker",
                {"task": i},
            )
            for i in items
        ]

    builder.add_node(
        "initialize",
        lambda _: {
            "revision": 0,
            "candidate": None,
            "errors": (),
            "micros": (),
            "expected_children": (),
        },
    )
    builder.add_node("optimizer", proposer)
    builder.add_node("prepare", lambda s: prepare(runtime, s))
    builder.add_node("assemble", lambda s: assemble(runtime, s))
    builder.add_node("judge", specialist, input_schema=JudgeState)
    builder.add_node("vote", lambda s: vote(runtime, s))
    builder.add_node("revise", lambda s: {"revision": s["revision"] + 1})
    builder.add_node("finish", finish)
    for name, child in (children or {}).items():
        builder.add_node(name, child)
        builder.add_edge(name, "assemble")
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "optimizer")
    builder.add_edge("optimizer", "prepare")
    builder.add_conditional_edges("prepare", dispatch)
    builder.add_conditional_edges(
        "assemble", lambda s: [Send("judge", {"state": s, "role": r.value}) for r in JudgeRole]
    )
    builder.add_edge("judge", "vote")
    builder.add_conditional_edges(
        "vote", lambda s: loop_route(s, runtime.settings.policy.max_revisions)
    )
    builder.add_edge("revise", "optimizer")
    builder.add_edge("finish", END)
    return builder.compile()


def build_hierarchy(
    runtime: AgentRuntime,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    interrupt_before: list[str] | None = None,
) -> Graph:
    worker = build_level(runtime, "L2")
    teams = {"domain-" + d.value: build_level(runtime, "L1", {"worker": worker}) for d in Domain}
    root = build_level(runtime, "L0", teams)
    curriculum = build_level(runtime, "L3")
    builder = StateGraph(WorkState, input_schema=WorkInput, output_schema=WorkOutput)
    builder.add_node("root", root)
    builder.add_node("curricularization", curriculum)

    def dispatch(state: WorkState) -> list[Send] | str:
        outcome = state["outcomes"][state["task"].id]
        if outcome.status != "approved" or not isinstance(outcome.candidate, RootPlan):
            return END
        return [
            Send(
                "curricularization",
                {
                    "task": WorkItem(
                        id="L3/grade-" + str(g),
                        level="L3",
                        grade=g,
                        budget_minutes=b,
                        assignments=outcome.candidate.assignments,
                        micros=outcome.micros,
                        instructions="Sequence every assigned micro-skill within the grade budget; inject context here only.",
                    )
                },
            )
            for g, b in sorted(runtime.request.budget_minutes.items())
        ]

    builder.add_edge(START, "root")
    builder.add_conditional_edges("root", dispatch)
    builder.add_edge("curricularization", END)
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def worker_materialization(
    task: WorkItem, candidate: Decomposition
) -> tuple[tuple[EvaluableMicro, ...], list[str]]:
    micros: tuple[EvaluableMicro, ...] = ()
    errors: list[str] = []
    try:
        micros = materialize(task, candidate)
    except ValueError as error:
        errors.append(str(error))
    if any(
        not m.skill.grade_band.minimum <= task.assignments[0].grade <= m.skill.grade_band.maximum
        for m in micros
    ):
        errors.append("micro-skill grade band excludes assigned grade")
    if sum(m.skill.estimated_minutes for m in micros) > task.assignments[0].minutes:
        errors.append("worker time allocation exceeded")
    return micros, errors


def loop_route(state: WorkState, max_revisions: int) -> str:
    return "finish" if state["accepted"] or state["revision"] >= max_revisions else "revise"
