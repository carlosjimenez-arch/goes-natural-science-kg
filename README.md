# GOES Natural Science KG

[![Python 3.13](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Managed with uv](https://img.shields.io/badge/managed_with-uv-DE5FE9)](https://docs.astral.sh/uv/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

## Contents

- [Overview](#overview)
- [Project status](#project-status)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Data, evidence and licensing](#data-evidence-and-licensing)
- [Quality and evaluation](#quality-and-evaluation)
- [Development](#development)
- [License](#license)

## Overview

An evidence-backed AI engineering pipeline for designing Natural Science curricula
for **El Salvador, grades 2–6**. The system ingests admissible source documents,
models observable skills, evaluates agent outputs and schedules authored learning
activities under an exact annual time budget.

LLMs propose and review content. A deterministic constraint solver schedules it.
Every published schedule preserves its input approval status; passing software
checks does not establish instructional quality or scientific validity.

## Project status

**Research and engineering prototype. No classroom-ready curriculum is approved.**

| Component | Implemented | Current limit |
|---|---|---|
| Research and contracts | Primary-source findings, typed schemas, stable IDs and semantic graph diffs | AND/OR prerequisite routes remain a proposed extension |
| Corpus | License gate, PDF extraction, semantic chunks, embeddings and traceable index | Eight accepted national documents per grade across six countries; TIMSS/NGSS corpus admission remains unresolved |
| Agent hierarchy | Four LangGraph levels, specialized judges, bounded revisions, checkpoints and caches | Rejected items require human review |
| Prompt evaluation | Three variants per critical role, three replicas, recorded provider responses and offline replay | Reference cases are agent-authored; decomposition has no eligible variant |
| Curricularization | CP-SAT scheduling, independent audit, JSON-LD and generated HTML reports | The provisional graph supplies only 640 of the required 9,600 minutes per grade |

The solver rejects that incomplete input. The schedules in
[`data/processed/curriculum/examples/`](data/processed/curriculum/examples/) are
**synthetic engineering fixtures**, not recommended lessons. Evidence, decisions
and measured results remain available in the repository for review.

## Installation

Requirements: **Python 3.13** and **uv 0.12.13**. uv manages the interpreter, virtual
environment, application dependencies and development tools. Direct dependencies
are pinned in [`pyproject.toml`](pyproject.toml); [`uv.lock`](uv.lock) fixes the
resolved dependency set and distribution hashes.

Install uv using its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
For macOS or Linux:

```sh
curl -LsSf https://astral.sh/uv/0.12.13/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

From the repository root:

```sh
uv python install
uv sync --locked --group dev
uv run --locked goes-science --help
```

`.python-version` selects the tested Python 3.13 patch release. No manual virtual
environment activation or separate pip installation is required. Offline tests use
recorded fixtures and do not require GCP credentials. Initial installation needs
network access to download the interpreter and packages.

## Configuration

[`.env.example`](.env.example) documents supported environment variables. Create a
local `.env` only when needed; retain any existing configuration. Settings use the
`GOES_NATURAL_SCIENCE_KG_` prefix and `__` for nested fields.

Live ingestion and agent experiments use **Google Cloud Vertex AI** with Application
Default Credentials and an explicitly configured project. They require network
access and can incur usage charges. Credentials, downloaded corpus bytes, vectors,
checkpoints and operational caches are excluded from Git.

Scheduling parameters and objective weights live in
[`defaults.yaml`](data/processed/curriculum/defaults.yaml) and
[`inquiry.yaml`](data/processed/curriculum/inquiry.yaml). Minutes are integer values;
the default budget is **160 hours = 9,600 minutes per grade**, with **0% tolerance**.

## Usage

### Build a curriculum

```sh
uv run --locked goes-science build \
  --skills-map path/to/sequencing-input.json \
  --budget-hours 160 --seed 42
```

Input is a graph snapshot or a `sequencing-input/1.0` wrapper containing a finite,
authored activity bank. A bare graph contributes only its initial-teaching minutes;
the solver never invents padding. The wrapper supplies practice, retrieval, bridges,
assessments and contextualized task alternatives.

A successful build writes to `data/processed/curriculum/`:

- `grade_2.json` through `grade_6.json`: schema-validated calendars.
- `graph.jsonld`: linked graph with lossless typed payloads and evidence references.
- `report.html`: generated coverage, cognitive/domain balance and schedule audit.
- `build.json`: solver status, input/settings hashes, objective and bound.

Failed builds return exit code **2**, retain a diagnostic report and archive previous
calendars outside the active output set. `feasible` is an audited solution;
`optimal` additionally requires a proof of optimality.

### Run a technical scenario

```sh
uv run --locked goes-science build \
  --skills-map tests/golden/sequencing/engineering.json \
  --budget-hours 140 --seed 42 \
  --config data/processed/curriculum/defaults.yaml \
  --output data/processed/curriculum/examples/140h
```

Use `--budget-hours 160` for the default budget or `inquiry.yaml` to increase the
inquiry objective weight. With bounded search, a different weight does not guarantee
a better incumbent for that criterion. Compare reported metrics and bounds; objective
values from different weight configurations are not directly comparable.

### Inspect evidence and changes

```sh
uv run --locked goes-science graph-diff --help
uv run --locked goes-science corpus-ingest --help
uv run --locked goes-science corpus-trace --help
uv run --locked goes-science orchestrate --help
uv run --locked goes-science prompts-evaluate --help
```

### Confidence estimates

Micro-skill `confidence` written by the generator is a self-report with no calibration
source and is never published as the estimate. Every orchestration run writes
`confidence.json` beside `report.json`: for each micro-skill it records the final
panel signals (vote fraction, mean judge score, supported-claim fraction, revisions,
hard errors), a raw score and the source of the value (`self_reported`,
`panel_derived` or `calibrated`). See
[decision 0012](decisions/0012-confidence-calibration.yaml).

```sh
uv run --locked goes-science confidence-fit samples.jsonl \
  --output data/processed/confidence-calibration.json --fitted-at 2026-09-14T00:00:00Z
uv run --locked goes-science confidence-estimate data/processed/orchestration/report.json \
  --output data/processed/orchestration --generated-at 2026-09-14T00:00:00Z \
  --calibration data/processed/confidence-calibration.json
```

`confidence-fit` needs at least 30 human-reviewed `calibration-sample/1.0` rows with both
accepted and rejected outcomes; it exits with code **2** otherwise. It fits a monotone
isotonic mapping and reports Brier score and expected calibration error. No calibration
record exists yet, so current estimates are uncalibrated panel scores and say so.

## Architecture

```text
Source discovery → license gate → fetch → parse → normalize → chunk → embed → index
                                         ↓ evidence anchors
L0 root → L1 domain teams → L2 decomposition → reviewed skill graph
                                                   ↓
                              L3 curricularization → CP-SAT → independent audit
```

Each agent level has specialized judges and at most **three revisions**. Judges emit
structured verdicts; optimizers rewrite content. Items that do not pass are marked
`needs_human_review`. Versioned Spanish prompts, typed state, cached requests and
checkpoints support review and resumption.

Graph nodes represent **context-free skill archetypes**. Context enters only during
curricularization. Hard constraints enforce exact budgets, prerequisite order,
joint co-requisite teaching, skill coverage and declared cross-grade continuity.
Content/cognitive balance, inquiry, spaced retrieval and thematic coherence are
configurable soft objectives.

Contextualization is capped at 60% of both time and units, with limits on repetition
and mandatory variety when used. These caps and cognitive targets are testable design
policies, not empirically established optimal ratios. See
[decision 0008](decisions/0008-constraint-curriculum.yaml) for assumptions and limits.

| Path | Responsibility |
|---|---|
| `src/goes_natural_science_kg/schemas/` | Pydantic contracts and exported JSON Schemas |
| `corpus/`, `graph/` | Evidence ingestion, provenance, semantic diffs and interchange |
| `agents/`, `eval/` | Orchestration, prompt registry, judges, evaluation harness and confidence calibration |
| `curriculum/` | Constraint model, calendar audit and generated reports |
| `prompts/`, `decisions/`, `findings.yaml` | Versioned prompts, architecture decisions and research evidence |
| `tests/` | Unit, integration, property, benchmark and attributed golden fixtures |

Source subdirectories in the table are relative to `src/goes_natural_science_kg/`.

## Data, evidence and licensing

The **corpus is reconstructed from manifests**, not committed as a document bundle.
[`data/manifests/`](data/manifests/) records source URLs, admission decisions,
licenses or official-publication status, checksums and processing outcomes. The
pipeline records rejected sources with reasons and continues.

Official-publication admission is not an open copyright license. Do not assume
that acceptance permits redistribution. The three cropped PDF test fixtures have
explicit attribution; full source documents remain in ignored `data/raw/`.
Semantic chunks preserve document, page, paragraph and offset anchors. A resolving
citation proves traceability, not that the cited passage entails the generated claim.

Trackable artifacts include contracts, research, processed outputs, manifests and
scrubbed recordings of actual provider responses. Local credentials, raw documents,
intermediate indexes, generated build archives and environment files are ignored.
Review artifacts before staging: `.gitignore` cannot remove files already tracked.

## Quality and evaluation

```sh
make setup       # uv-managed Python and locked development environment
make ci          # local full gate: lock consistency, lint, types, offline tests and Bandit
make security    # dependency advisories; requires network access
make bench       # benchmark report in reports/benchmark.json
make build       # source distribution and wheel via uv
```

Tests disable Internet sockets and replay recorded provider observations. The prompt
experiment contains **630 cells** across three replicas; human reference review and
production prompt promotion remain pending. Results and selection reasons are in
[`data/processed/prompt-evaluation/report.json`](data/processed/prompt-evaluation/report.json).

Local validation on **Python 3.13.15**: **159 tests passed**, **86% coverage** and
**22 benchmark cases passed**. Lint, strict types, static security checks and the
source/wheel build pass. No hosted continuous-integration workflow is configured at
this time; the gate runs locally through `make ci`.

| Benchmark workload | Mean on macOS arm64, Python 3.13.15 |
|---|---:|
| Three real cropped PDFs, parsed separately | 713.0 / 81.6 / 149.2 ms |
| Index construction from golden evidence | 7.8 ms |
| CP-SAT, 205 synthetic activities | 3.131 s (one measured round) |

Measurements and runner metadata are in
[`tests/benchmarks/baseline.json`](tests/benchmarks/baseline.json); prompt variant
results are in the [comparison table](data/processed/prompt-evaluation/comparison.csv).
Historical ingestion measurements are retained in
[`data/processed/benchmarks/corpus-benchmark.json`](data/processed/benchmarks/corpus-benchmark.json).
Benchmark records identify their Python version and runner. Compare regressions only
on equivalent workloads, hardware and interpreters. The 20% regression gate
(`make bench-compare`) requires a matching runner and is not yet automated. A Python
3.12 baseline is not a valid performance gate for Python 3.13.

## Development

Use uv for dependency changes and commit `pyproject.toml` and `uv.lock` together.
Run `uv run --locked pre-commit install` to enable the repository hooks. Changes to
contracts require explicit version review and regenerated schema exports; artifact
changes require refreshed checksums. Architecture decisions are schema-validated
YAML files with linked enforcement tests.

[`CLAUDE.md`](CLAUDE.md) defines repository conventions. Code and documentation are
in English; prompt bodies use Salvadoran Spanish. Keep `.env`, credentials, local
logs and downloaded documents out of commits.

## License

Project code is licensed under [Apache License 2.0](LICENSE); see [NOTICE](NOTICE).
External source documents and retained evidence remain subject to their own terms.
