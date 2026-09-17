# src/goes_natural_science_kg/agents/release_replay.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Recover the first structurally valid draft from real cached generation and patch records.
from pathlib import Path

from goes_natural_science_kg.agents.release import apply_patch, design_errors
from goes_natural_science_kg.eval.experiment import read_observation
from goes_natural_science_kg.schemas.release import (
    ReleaseActivityBank,
    ReleaseUnitResult,
    UnitDesign,
    UnitPatch,
    UnitSkillDesign,
)


def first_structural_draft(
    result: ReleaseUnitResult, caches: tuple[Path, ...]
) -> ReleaseUnitResult:
    """Keep model rejection history while selecting a compileable, explicitly provisional draft."""
    design = None
    skills = None
    for key in result.request_ids:
        cache = next((p for p in caches if (p / (key + ".json")).exists()), None)
        if cache is None:
            raise FileNotFoundError("missing recorded request: " + key)
        observation = read_observation(cache, key)
        if observation.response is None:
            continue
        prompt = observation.request["prompt_id"]
        if prompt == "release-skills":
            skills = UnitSkillDesign.model_validate_json(observation.response)
        elif prompt == "release-activities" and skills is not None:
            bank = ReleaseActivityBank.model_validate_json(observation.response)
            design = UnitDesign(**skills.model_dump(), activities=bank.activities)
        elif prompt in {"release-design", "release-optimize"}:
            design = UnitDesign.model_validate_json(observation.response)
        elif prompt == "release-patch" and design is not None:
            design = apply_patch(design, UnitPatch.model_validate_json(observation.response))
        else:
            continue
        if design is not None and not design_errors(design, result.assignment):
            revision = int(observation.request["revision"])
            panels = tuple(p for p in result.panels if p.revision <= revision)
            return ReleaseUnitResult(
                assignment=result.assignment,
                initial=result.initial,
                final=design,
                panels=panels,
                revisions=revision,
                status="panel_passed" if panels and panels[-1].accepted else "needs_human_review",
                request_ids=result.request_ids[: result.request_ids.index(key) + 1],
                errors=result.errors,
            )
    raise ValueError("no structurally valid recorded draft: " + result.assignment.key)
