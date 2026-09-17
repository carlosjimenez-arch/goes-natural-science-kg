# GOES Natural Science KG

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Managed with uv](https://img.shields.io/badge/managed_with-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

## Contents

- [Overview](#overview)
- [Current proposal](#current-proposal)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Evidence and licensing](#evidence-and-licensing)
- [Evaluation and performance](#evaluation-and-performance)
- [Development](#development)
- [License](#license)

## Overview

An evidence-backed AI engineering pipeline for designing Natural Science curricula
for **El Salvador, grades 2–6**. Models propose observable skills and finite learning
activities; specialist reviewers assess them; CP-SAT computes an annual calendar.
Source evidence, review judgments and scheduling certificates remain separate.

**Research prototype. The published curriculum is a proposal requiring teacher
review, not an approved national curriculum or a demonstrated learning intervention.**

## Current proposal

The September 17 release covers **30 official textbook units**, organized into
**60 proposed parent skills and 306 micro-skills**. Each micro-skill has an observable
assessment, expected response, scoring rule and source location. The final calendar
contains **9,600 minutes per grade**, with no violations of the declared hard constraints.

| Artifact | Purpose |
|---|---|
| [Final curriculum proposal](data/processed/proposals/2026-09-17/v5/curriculum-proposal.html) | Skills, micro-skills, assessments, materials, safety and differentiation |
| [Calendar and coverage report](data/processed/proposals/2026-09-17/v5/report.html) | Actual selected tasks, time allocation, cognitive/content balance and contexts |
| [Final reproducible input](data/processed/proposals/2026-09-17/v5/skills-map.json) | Graph, finite activity bank, local contexts and continuity constraints |
| [Grade calendars](data/processed/curriculum/) | Schema-validated `grade_2.json` through `grade_6.json` and JSON-LD export |
| [First snapshot](data/processed/proposals/2026-09-17/v1/) | Initial aggregate graph and rejected scheduling attempt; no fabricated calendar |
| [Intermediate snapshots](data/processed/proposals/2026-09-17/) | Versions 2–4 retain the context-selection failures diagnosed before final version 5 |
| [Critique](data/processed/proposals/2026-09-17/critique.json) and [semantic diff](data/processed/proposals/2026-09-17/semantic-diff.json) | Corrections, unresolved concerns and first-to-final graph changes |
| [Review and cost audit](data/processed/proposals/2026-09-17/run-audit.json) | Actual provider calls, failures, revisions, token usage and estimated cost |
| [Evidence audit](data/processed/proposals/2026-09-17/v5/binding-audit.json) | Original document hashes and 317 verified quotation locations |

Limits are material. Only **2 of 30 final candidates retain a passing model panel**;
others fail review or changed after verification. All require human validation.
The 5.1× decomposition ratio falls below the requested 6× target. Parent skills are
an authored abstraction of the books, not an independently supplied official skill
map. Some advanced topics, missing content-domain coverage and global prerequisite
completeness require curricular adjudication. A valid timetable does not resolve them.

The first aggregate had nine passing panels. A revised rubric passed three candidates
before further editorial corrections. These are different candidates and review
conditions, not a controlled estimate of improvement. Model votes are uncalibrated.

## Installation

Use **Python 3.13** and **uv 0.12.13**. uv manages the interpreter, environment,
application and development tools. Direct versions are exact in
[`pyproject.toml`](pyproject.toml); [`uv.lock`](uv.lock) fixes transitive dependencies.

Install uv using its [official guide](https://docs.astral.sh/uv/getting-started/installation/).
On macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install
uv sync --locked --group dev
uv run --locked goes-science --help
```

`.python-version` selects the tested **3.13.15** patch. No separate pip installation
or manual environment activation is needed. Initial installation requires network
access; offline tests do not require GCP credentials.

## Configuration

[`.env.example`](.env.example) documents the `GOES_NATURAL_SCIENCE_KG_` prefix and
nested `__` settings. Preserve existing credentials; local `.env` files are ignored.
Live generation uses Vertex AI with Application Default Credentials and an explicit
GCP project. Calls require network access and incur usage charges.

The proposal uses five criterion-specific judges across three Gemini models;
two judge roles share the generator's model family. This is not independent human
validation. Generation used temperature 0.2 and recorded responses: cached replay
is reproducible, fresh remote generation is not guaranteed byte-identical.

Scheduling settings are recorded beside each snapshot. The 160-hour scenario means
**160 clock hours, 0% tolerance**, using 32 planning weeks of 300 minutes. It is not a
verified official timetable. Cognitive targets, context caps and durations are
provisional design policies. Objective weights and the budget are configurable.

## Usage

### Rebuild the final proposal

```sh
uv run --locked goes-science build \
  --skills-map data/processed/proposals/2026-09-17/v5/skills-map.json \
  --config data/processed/proposals/2026-09-17/v5/settings.json \
  --budget-hours 160 --seed 42
```

Outputs include grade calendars, `graph.jsonld`, `build.json` and generated
`report.html`. `feasible` means the independent constraint audit passed;
`optimal` additionally requires an optimality proof. This release is **feasible**.
Failed builds return exit code 2 and retain diagnostics. No LLM fills missing hours.

Use `--budget-hours 140 --output data/interim/scenario-140h` to explore a smaller
budget. Change the recorded objective weights to prioritize inquiry. A bounded
search may return `unknown`; a weight change does not guarantee a better incumbent.

### Inspect and extend the pipeline

```sh
uv run --locked goes-science corpus-ingest --help
uv run --locked goes-science corpus-trace --help
uv run --locked goes-science proposal-prepare --help
uv run --locked goes-science proposal-generate --help
uv run --locked goes-science proposal-review --help
uv run --locked goes-science proposal-publish --help
uv run --locked goes-science proposal-diff --help
uv run --locked goes-science orchestrate --help
uv run --locked goes-science prompts-evaluate --help
```

`proposal-prepare` reconstructs textbook assignments from reviewed ingestion plans
and SHA-verified documents. The published scheduling input includes editorial
corrections and explicit continuity handoffs; rerunning a generator does not
reproduce those editorial decisions automatically.

## Architecture

```text
Discover → license gate → fetch/cache → layout parse → semantic chunks → evidence
                                                              ↓
Reviewed unit → skill design → finite activity bank → deterministic checks
                                      ↑                         ↓
                                bounded patch ← specialist panel
                                                              ↓
                                 editorial review → CP-SAT → independent audit
```

The experimental unit subgraph runs alongside the existing four-level LangGraph
hierarchy. Dynamic fan-out, explicit reducers, per-unit checkpoints, hashed response
caches and bounded transport retries support resumption. Every optimization loop
allows at most three revisions. A separate single-pass editorial verification never
resets that limit; later edits invalidate its approval for the changed candidate.

Five judges assess curricular alignment, cognitive demand, graph structure, evidence
and age appropriateness. Acceptance requires at least four passing votes at 0.8,
passing evidence and age judgments, and no critical finding. Judges do not rewrite.
These thresholds are explicit policy, not validated measures of learning quality.

Graph nodes are **context-free skill archetypes**. Context is injected into selected
curricular activities only. The solver enforces exact time, declared prerequisites,
coverage and four adjacent-grade diagnostic handoffs. Large fixed-grade banks receive
a deterministic finite-bank starting schedule; CP-SAT and the independent audit still
control acceptance. Context seeds use authored alternatives; a bounded conditional solve then optimizes
context selection under the same time, count, repetition and variety constraints.

| Path | Responsibility |
|---|---|
| `src/goes_natural_science_kg/schemas/` | Central Pydantic contracts and exported schemas |
| `corpus/`, `graph/` | Ingestion, provenance, semantic diffs and interchange |
| `agents/`, `eval/` | Orchestration, registry, reviews and evaluation |
| `curriculum/` | Constraint solving, independent audits and artifact generation |
| `prompts/`, `decisions/`, `findings.yaml` | Versioned prompts, decisions and primary-source research |
| `tests/` | Unit, integration, property, benchmark and attributed golden fixtures |

Package subdirectories above are relative to `src/goes_natural_science_kg/`.

## Evidence and licensing

**Reconstruct the corpus from [`data/manifests/`](data/manifests/).** Manifests record
URLs, license decisions, source hashes and reviewed segmentation. Rejected sources
retain a reason and do not stop the pipeline. Official-publication admission is not
a blanket open-copyright license or a guarantee of scientific correctness.

Full PDFs, extracted source text, source-bearing provider requests, embeddings,
credentials and checkpoints remain ignored. Published bindings retain source IDs,
URLs, page/paragraph locations, offsets and quote hashes. An exact quotation location
proves traceability, not semantic entailment. Three attributed cropped PDF fixtures
are the explicit test-only exception to the document exclusion.

The proposal uses MINED textbooks, the second-grade teacher guide and a narrowly
reviewed NASA background passage. The older international curriculum corpus supports
alignment. New textbook embeddings were not generated: this release uses reviewed
unit/page evidence. Retrieval quality needs a separately annotated benchmark before
embedding changes can be called an improvement.

## Evaluation and performance

Historical evaluations retain three prompt variants, three replicas, a seven-arm
model/prompt follow-up and a reviewer-agreement study. No decomposition prompt has
passed the human-review promotion gate. See
[`data/processed/prompt-evaluation/`](data/processed/prompt-evaluation/) and
[decisions 0013–0015](decisions/0015-reviewer-agreement.yaml).

The proposal audit contains **724 actual provider observations**, including failed
transport experiments. Known list-price estimates total **USD 46.83**; **123 calls
have unknown cost**, not zero cost. This is not an invoice. Latency is recorded per
call, separately from solver wall-clock time.

Final schedule generation and its byte-identical repeat are recorded in
[`solver-measurement.json`](data/processed/proposals/2026-09-17/solver-measurement.json).
The search reports its objective and bound without claiming optimality. Cognitive
and content-domain targets that were missed remain visible in the generated report.

Local validation: **204 tests**, **81% coverage**, strict typing, lint, Bandit,
dependency audit and package build. All 29 benchmark workloads execute, but the
20% comparison gate remains **failing**: license-workflow mean +78.1% and the
small proposal-compilation mean +24.5%. Other runs varied substantially; these
flags are retained rather than declared resolved. See the
[regression report](data/processed/benchmarks/proposal-regression-report.txt).

| Workload | Mean, macOS arm64 / Python 3.13.15 |
|---|---:|
| Three real cropped PDFs | 778.5 / 92.2 / 147.0 ms |
| Golden evidence index | 8.42 ms |
| Finite-bank starting schedule, 1,194 activities | 38.63 ms |
| Compile an eight-micro-skill contract fixture | 0.569 ms |
| Complete final schedule / identical repeat | 63.06 / 62.24 s |

The schema-export workload grew from 81 to 117 contracts and the prompt registry
from 35 to 50 artifacts. Only those two reference cases were refreshed for their
changed workloads; the previous baseline is retained. Unchanged-workload references
remain in the 20% regression gate.

Local benchmark results are retained under [`data/processed/benchmarks/`](data/processed/benchmarks/).
Compare performance only on matching workloads, hardware and interpreter versions.
The 20% comparison gate requires a compatible runner; historical Python 3.12 results
are not a valid Python 3.13 baseline. No hosted CI workflow is currently configured.

## Development

```sh
make ci          # locked dependencies, lint, strict types, offline tests and Bandit
make security    # dependency advisories; requires network access
make bench       # measured benchmark report
make build       # source distribution and wheel through uv
```

Commit dependency changes with `uv.lock`. Contract changes require version review
and regenerated exports; artifacts require checksum manifests. Tests disable Internet
sockets. [`CLAUDE.md`](CLAUDE.md) defines conventions. Code and documentation are in
English; prompt bodies use Salvadoran Spanish. Review staged files before committing:
`.gitignore` does not remove files already tracked.

## License

Repository code is licensed under [Apache 2.0](LICENSE). Source documents retain
their own terms; see [NOTICE](NOTICE) and the source manifests.
