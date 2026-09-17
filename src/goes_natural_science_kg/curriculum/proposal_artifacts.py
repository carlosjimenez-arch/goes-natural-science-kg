# src/goes_natural_science_kg/curriculum/proposal_artifacts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Publish readable provisional curricula with assessment and source-location evidence.
from __future__ import annotations

import html
from pathlib import Path

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.curriculum.artifacts import write_build
from goes_natural_science_kg.curriculum.release import compile_proposal
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.release import (
    ContinuityProposal,
    PublishedActivityBank,
    PublishedReviews,
    PublishedUnitReview,
    ReleaseUnitResult,
)
from goes_natural_science_kg.schemas.sequencing import (
    LocalContext,
    SequencingSettings,
    SolverReport,
)


def proposal_html(results: tuple[ReleaseUnitResult, ...], status: str) -> str:  # noqa: C901 - nested grade/unit/skill audit rendering
    esc = html.escape
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Proposed Natural Science Curriculum</title><style>body{max-width:1100px;margin:2rem auto;padding:1rem;font:16px/1.6 system-ui;color:#19364a}h1,h2,h3{line-height:1.2}summary{cursor:pointer;font-weight:600}details{border:1px solid #ced9df;padding:1rem;margin:1rem 0}dt{font-weight:600}dd{margin:0 0 1rem}aside{background:#fff2d9;padding:1rem}code{overflow-wrap:anywhere}a{color:#075a87}@media print{details{break-inside:avoid}body{font-size:10pt}}</style><h1>Proposed Natural Science Curriculum</h1><aside>Provisional proposal for grades 2–6. Model review is not teacher validation or Ministry approval. Misconceptions are diagnostic hypotheses; durations require classroom trials. The 160-hour scenario uses clock hours, not a verified official timetable.</aside>'
    ]
    parts.append(
        "<p>Schedule status: "
        + esc(status)
        + '. <a href="report.html">Coverage, balance and solver audit</a></p><nav>'
    )
    parts.extend(f'<a href="#grade-{g}">Grade {g}</a> · ' for g in range(2, 7))
    parts.append("</nav>")
    for grade in range(2, 7):
        parts.append(f'<h2 id="grade-{grade}">Grade {grade}</h2>')
        for result in results:
            if result.assignment.grade != grade or result.final is None:
                continue
            design = result.final
            parts.append(
                "<h3>"
                + esc(result.assignment.title)
                + "</h3><p>"
                + esc(design.progression)
                + "</p>"
            )
            parts.append(
                f"<p>Review: {esc(result.status)} · optimizer revisions: {result.revisions}. Unit allocation: {result.assignment.weeks} planning weeks.</p>"
            )
            for skill in design.skills:
                parts.append(
                    "<h4>"
                    + esc(skill.label)
                    + "</h4><p>"
                    + esc(skill.pedagogical_rationale)
                    + "</p>"
                )
                for micro in skill.micros:
                    parts.append(
                        "<details><summary>"
                        + esc(micro.verb + " " + micro.knowledge_object)
                        + "</summary><dl>"
                    )
                    for label, value in [
                        ("Cognitive demand", micro.cognitive_domain.value),
                        ("Assessment task", micro.assessment_task),
                        ("Expected response", micro.expected_response),
                        ("Scoring rule", micro.scoring_rule),
                        ("Mastery evidence", micro.mastery),
                        ("Diagnostic hypotheses", "; ".join(micro.misconceptions)),
                        (
                            "Local prerequisites",
                            ", ".join(micro.prerequisites)
                            or "No dependency asserted within this unit",
                        ),
                    ]:
                        parts.append("<dt>" + label + "</dt><dd>" + esc(value) + "</dd>")
                    anchors = {a.key: a for a in result.assignment.anchors}
                    parts.append("<dt>Scientific support</dt><dd><ul>")
                    for support in micro.support:
                        anchor = anchors.get(support.anchor_key)
                        if anchor:
                            parts.append(
                                "<li>"
                                + esc(support.claim)
                                + ' · <a href="'
                                + esc(anchor.source_url, quote=True)
                                + "#page="
                                + str(anchor.page)
                                + '">Source page '
                                + str(anchor.page)
                                + "</a> · paragraph "
                                + esc(anchor.paragraph_id)
                                + "</li>"
                            )
                    parts.append("</ul></dd></dl></details>")
            parts.append(
                "<details><summary>Activity bank: materials, safety and differentiation</summary>"
            )
            for activity in design.activities:
                parts.append(
                    "<h4>"
                    + esc(activity.key)
                    + f" · {activity.minutes} min · "
                    + esc(activity.kind)
                    + "</h4><p>"
                    + esc(activity.task)
                    + "</p><dl>"
                )
                for label, value in [
                    ("Mastery criterion", activity.mastery_criterion),
                    ("Materials", ", ".join(activity.materials)),
                    ("Safety", activity.safety),
                    ("Access and differentiation", activity.differentiation),
                    ("Time rationale", activity.rationale_for_minutes),
                ]:
                    parts.append("<dt>" + label + "</dt><dd>" + esc(value) + "</dd>")
                parts.append("</dl>")
            parts.append("</details><p>Limitations: " + esc("; ".join(design.limitations)) + "</p>")
    return "".join(parts) + "</html>"


def write_proposal(
    output: Path,
    results: tuple[ReleaseUnitResult, ...],
    contexts: tuple[LocalContext, ...],
    settings: SequencingSettings,
    version: str,
    continuity: tuple[ContinuityProposal, ...] = (),
) -> SolverReport:
    """Write a reviewed-input snapshot; never overwrite an existing version directory."""
    from goes_natural_science_kg.agents.release import design_errors
    from goes_natural_science_kg.curriculum.solver import solve

    if output.exists():
        raise FileExistsError("proposal snapshot already exists: " + str(output))
    failures = {
        r.assignment.key: design_errors(r.final, r.assignment) if r.final else ("missing design",)
        for r in results
    }
    if any(failures.values()):
        raise ValueError("structurally invalid proposal: " + canonical_json(failures))
    data, pedagogy, bindings = compile_proposal(results, contexts, version)
    from goes_natural_science_kg.curriculum.continuity import add_continuity

    data = add_continuity(data, continuity)
    report = solve(data, settings)
    output.mkdir(parents=True)
    for name, value in [
        ("skills-map.json", data),
        ("graph.json", data.graph),
        ("settings.json", settings),
    ]:
        atomic_bytes(output / name, (canonical_json(value) + "\n").encode())
    for name, values in [("pedagogy.jsonl", pedagogy), ("evidence-bindings.jsonl", bindings)]:
        atomic_bytes(output / name, "".join(canonical_json(v) + "\n" for v in values).encode())
    # Public review records omit raw textbook paragraphs and full provider requests.
    reviews = PublishedReviews(
        units=tuple(
            PublishedUnitReview(
                unit_key=r.assignment.key,
                status=r.status,
                revisions=r.revisions,
                panels=r.panels,
                request_ids=r.request_ids,
                assignment_sha256=content_hash(r.assignment),
                errors=r.errors,
            )
            for r in results
        )
    )
    atomic_bytes(output / "reviews.json", (canonical_json(reviews) + "\n").encode())
    atomic_bytes(
        output / "continuity.json",
        (canonical_json([p.model_dump(mode="json") for p in continuity]) + "\n").encode(),
    )
    atomic_bytes(
        output / "activity-details.jsonl",
        "".join(
            canonical_json(
                PublishedActivityBank(unit_key=r.assignment.key, activities=r.final.activities)
            )
            + "\n"
            for r in results
            if r.final
        ).encode(),
    )
    write_build(output, data, settings, report)
    atomic_bytes(
        output / "curriculum-proposal.html", proposal_html(results, report.status).encode()
    )
    return report


def write_draft_attempt(output: Path, results: tuple[ReleaseUnitResult, ...]) -> None:
    """Retain a rejected first aggregate without manufacturing a valid calendar."""
    from goes_natural_science_kg.agents.release import design_errors
    from goes_natural_science_kg.curriculum.release import compile_records
    from goes_natural_science_kg.graph.linked_data import export_jsonld
    from goes_natural_science_kg.schemas.release import DraftAttempt

    if output.exists():
        raise FileExistsError(str(output))
    graph, activities, _, pedagogy, bindings = compile_records(
        results, (), "proposal-2026-09-17-v1"
    )
    errors = {
        r.assignment.key: design_errors(r.final, r.assignment) if r.final else ("missing design",)
        for r in results
    }
    if not any(errors.values()):
        raise ValueError("draft is not structurally rejected; use the normal proposal writer")
    attempt = DraftAttempt(
        graph_sha256=content_hash(graph),
        structural_errors=errors,
        limitation="First aggregate after bounded model optimization and recorded quote-format repairs. Structural defects prevent a certified calendar; no grade schedule is fabricated.",
    )
    output.mkdir(parents=True)
    for name, value in [
        ("graph.json", graph),
        ("graph.jsonld", export_jsonld(graph).model_dump(mode="json", by_alias=True)),
        ("attempt.json", attempt),
    ]:
        atomic_bytes(output / name, (canonical_json(value) + "\n").encode())
    for name, values in [
        ("pedagogy.jsonl", pedagogy),
        ("evidence-bindings.jsonl", bindings),
        ("activity-bank.jsonl", activities),
    ]:
        atomic_bytes(output / name, "".join(canonical_json(v) + "\n" for v in values).encode())
    atomic_bytes(
        output / "curriculum-proposal.html",
        proposal_html(results, attempt.status).encode(),
    )
    rows = "".join(
        "<li>" + html.escape(key + ": " + error) + "</li>"
        for key, values in errors.items()
        for error in values
    )
    atomic_bytes(
        output / "report.html",
        (
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Draft rejection</title><h1>First aggregate: scheduling rejected</h1><ul>'
            + rows
            + "</ul></html>"
        ).encode(),
    )
