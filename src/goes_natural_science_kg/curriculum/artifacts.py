# src/goes_natural_science_kg/curriculum/artifacts.py
# AI Depto
# Copyright 2026 Gobierno de El Salvador.
# SPDX-License-Identifier: Apache-2.0
# Mission: Publish audited solver outputs and generate a readable coverage report.
import html
import json
from pathlib import Path

from goes_natural_science_kg.corpus.fetch import atomic_bytes
from goes_natural_science_kg.graph.linked_data import export_jsonld
from goes_natural_science_kg.schemas.base import canonical_json, content_hash
from goes_natural_science_kg.schemas.graph import EvidenceGraphSnapshot, GraphSnapshot
from goes_natural_science_kg.schemas.sequencing import (
    LearningActivity,
    SequencingInput,
    SequencingSettings,
    SolverReport,
)


def graph_input(graph: GraphSnapshot) -> SequencingInput:
    """Minimal graph adapter: no invented review bank or unbounded spare-time filler."""
    return SequencingInput(
        graph=graph,
        approval="provisional",
        activities=tuple(
            LearningActivity(
                id="teach-" + content_hash(m.id)[:16],
                micro_skill_ids=(m.id,),
                kind="teach",
                minutes=m.estimated_minutes,
                cognitive_domain=m.cognitive_domain,
                theme=next(s.domain for s in graph.skills if s.id == m.parent_skill_id),
                task=m.observable_verb + " " + m.knowledge_object,
                mastery_criterion=m.evidence_of_mastery,
                allowed_grades=tuple(range(m.grade_band.minimum, m.grade_band.maximum + 1)),
            )
            for m in graph.micro_skills
        ),
    )


def read_input(path: Path) -> SequencingInput:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") == "graph-snapshot/2.0":
        return graph_input(EvidenceGraphSnapshot.model_validate(raw))
    if raw.get("schema_version") == "graph-snapshot/1.0":
        return graph_input(GraphSnapshot.model_validate(raw))
    return SequencingInput.model_validate(raw)


def report_html(data: SequencingInput, settings: SequencingSettings, report: SolverReport) -> str:
    esc = html.escape
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Science curriculum audit</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1200px;padding:1rem;color:#172a35;background:#f5f7f8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:.6rem;border-bottom:1px solid #ccd4d9;text-align:left}code{overflow-wrap:anywhere}details{margin:1rem 0}h1,h2{color:#084c61}.notice{border-left:5px solid #c87b17;padding:1rem;background:#fff4df}a{color:#07578c}</style><h1>Science curriculum audit</h1>'
    ]
    parts.append(
        '<p class="notice">Status: '
        + esc(report.status)
        + " · Input: "
        + esc(data.approval)
        + " · "
        + str(settings.budget_hours)
        + " h per grade. A solver certificate validates constraints; it does not validate pedagogy or source entailment.</p>"
    )
    parts.append("<p>Input SHA-256: <code>" + report.input_sha256 + "</code></p>")
    if report.hard_errors:
        parts.append(
            "<h2>Build failed</h2><ul>"
            + "".join("<li>" + esc(e) + "</li>" for e in report.hard_errors)
            + "</ul>"
        )
    parts.append(
        "<h2>Coverage and balance</h2><p>"
        + str(sum(report.coverage.values()))
        + " / "
        + str(len(data.graph.skills))
        + " input skills covered. Context cap: "
        + str(settings.context_max_percent)
        + "% by minutes and units.</p>"
    )
    parts.append(
        "<table><tr><th>Grade</th><th>Minutes</th><th>Knowing / Applying / Reasoning</th><th>Inquiry</th><th>Context</th><th>Content domains (minutes)</th></tr>"
    )
    for g, m in sorted(report.grade_metrics.items()):
        percentages = " / ".join(
            f"{100 * m['cognitive_minutes'][k] / m['total_minutes']:.1f}%"
            for k in ("knowing", "applying", "reasoning")
        )
        parts.append(
            f"<tr><td>{g}</td><td>{m['total_minutes']}</td><td>{percentages}</td><td>{m['inquiry_minutes']} min</td><td>{m['context_minutes']} min</td><td>"
            + esc(str(m["domain_minutes"]))
            + "</td></tr>"
        )
    parts.append(
        "</table><h2>Objective and gaps</h2><pre>"
        + esc(
            canonical_json(
                {
                    "weights": settings.weights.model_dump(),
                    "components": report.objective_components,
                    "objective": report.objective_value,
                    "best_bound": report.best_bound,
                }
            )
        )
        + "</pre>"
    )
    for grade in report.grades:
        parts.append(
            f"<details><summary>Grade {grade.grade}: {len(grade.units)} units · retrieval and themes</summary><pre>"
            + esc(canonical_json(report.grade_metrics[grade.grade]))
            + "</pre><table><tr><th>Period / minutes</th><th>Task and mastery criterion</th><th>Micro-skills</th><th>Context</th></tr>"
        )
        for u in grade.units:
            parts.append(
                f"<tr><td>{u.period + 1}: {u.start_minute}–{u.end_minute}</td><td>"
                + esc(u.task)
                + "<br><em>"
                + esc(u.mastery_criterion)
                + "</em></td><td>"
                + esc(", ".join(u.micro_skill_ids))
                + "</td><td>"
                + esc(u.context_id or "Abstract / no injected context")
                + "</td></tr>"
            )
        parts.append("</table></details>")
    parts.append("<h2>Local context evidence</h2>")
    for context in data.contexts:
        parts.append("<h3>" + esc(context.label) + "</h3><ul>")
        for source in context.evidence:
            parts.append(
                "<li>"
                + esc(source.claim)
                + ' <a href="'
                + esc(str(source.source_url), quote=True)
                + '">Primary source</a> · retrieved '
                + str(source.retrieved_at)
                + "</li>"
            )
        parts.append("</ul>")
    return "".join(parts) + "</html>"


def write_build(
    output: Path, data: SequencingInput, settings: SequencingSettings, report: SolverReport
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Archive only products of a previous certified build, never arbitrary user files.
    if (output / "build.json").exists():
        previous = SolverReport.model_validate_json((output / "build.json").read_bytes())
        archive = (
            output / "previous" / (previous.input_sha256[:16] + "-" + previous.settings_sha256[:16])
        )
        archive.mkdir(parents=True, exist_ok=True)
        for name in [
            "build.json",
            "report.html",
            "graph.jsonld",
            *(f"grade_{g}.json" for g in range(2, 7)),
        ]:
            path = output / name
            if path.exists():
                path.replace(archive / name)
    for grade in report.grades:
        atomic_bytes(output / f"grade_{grade.grade}.json", (canonical_json(grade) + "\n").encode())
    linked = export_jsonld(data.graph)
    atomic_bytes(
        output / "graph.jsonld",
        (canonical_json(linked.model_dump(mode="json", by_alias=True)) + "\n").encode(),
    )
    atomic_bytes(output / "report.html", report_html(data, settings, report).encode())
    atomic_bytes(output / "build.json", (canonical_json(report) + "\n").encode())
