# src/goes_natural_science_kg/curriculum/continuity.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Turn explicit cross-grade diagnostic handoffs into audited scheduling constraints.
from goes_natural_science_kg.schemas.release import ContinuityProposal
from goes_natural_science_kg.schemas.sequencing import (
    ContinuityThread,
    LearningActivity,
    SequencingInput,
)


def add_continuity(
    data: SequencingInput, proposals: tuple[ContinuityProposal, ...]
) -> SequencingInput:
    """Add authored early-year diagnostics; do not infer prerequisite edges from similarity."""
    micros = {m.id: m for m in data.graph.micro_skills}
    activities = list(data.activities)
    threads = list(data.continuity)
    if len({p.id for p in proposals}) != len(proposals):
        raise ValueError("duplicate continuity proposal")
    for proposal in proposals:
        source, target = micros[proposal.source_micro_id], micros[proposal.target_micro_id]
        if (
            source.grade_band.maximum != proposal.target_grade - 1
            or target.grade_band.minimum != proposal.target_grade
        ):
            raise ValueError("continuity must connect adjacent grade bands")
        opening = next(
            a for a in data.activities if a.kind == "teach" and source.id in a.micro_skill_ids
        )
        activity_id = "continuity-" + proposal.id
        activities.append(
            LearningActivity(
                id=activity_id,
                micro_skill_ids=(source.id,),
                kind="bridge",
                minutes=proposal.minutes,
                cognitive_domain=source.cognitive_domain,
                inquiry=False,
                theme="cross-grade-diagnostic",
                task=proposal.task,
                mastery_criterion=proposal.scoring_rule,
                allowed_grades=(proposal.target_grade,),
                mandatory=True,
            )
        )
        threads.append(
            ContinuityThread(
                id=proposal.id, opening_activity_id=opening.id, continuation_activity_id=activity_id
            )
        )
    return SequencingInput(
        **{**data.model_dump(), "activities": tuple(activities), "continuity": tuple(threads)}
    )
