# CLAUDE.md — working contract for `goes-natural-science-kg`

## Evidence-grounded proposal amendments (2026-09-17)

The maintainer authorizes a first curriculum snapshot, critique, a revised final
proposal and a conventional commit/push using the configured GOES identity. Do not
add fictional co-authors. Vertex transfer of public MINED excerpts and generated
proposals to project `g-edu-lxp-xai-dev-prj-976d`, global endpoint, was explicitly
authorized in chat. Preserve the existing credentials without printing or editing them.
Decision 0016 governs the experimental proposal route. Separate skill decomposition
from finite activity-bank design; validate exact quotation locations and structural
constraints before specialist review. Apply explicit optimizer patches, retain every
provider failure, and distinguish transport retries from pedagogical revisions.
Provider grammar simplification never weakens the local Pydantic acceptance contract.
Generated proposal prompts are experimental; they have not passed the historical
human-review promotion gate. Unit-panel agreement is an uncalibrated review signal.
Retain immutable draft/final products, source-offset hashes, critique and solver
certificates. Full textbooks and source-bearing model requests remain ignored.
Explicit cross-grade handoffs belong to the scheduling layer; topic similarity alone
must not create prerequisite edges. A final proposal is not a classroom approval.

## Runtime and documentation amendments (2026-09-14)

The maintainer requires Python 3.13 and uv-managed installation for the interpreter,
application and development tools. Pin the tested patch in `.python-version` and uv
in `pyproject.toml` and CI. README is a concise English project guide with verified
badges and a table of contents, not a chronological phase log. Preserve research and
historical measurements in their data artifacts. Python 3.12 performance baselines
remain historical; establish a Python 3.13 baseline before comparison. Operational
build archives are ignored; retained manifests must not reference ignored files.

## Phase 6 amendments (2026-09-14)

The maintainer authorizes CP-SAT curricularization, finite authored activity banks,
scenario configuration and independently audited JSON/HTML/JSON-LD outputs. Decision
0008 governs exact scenario budgets (including 140h), continuity bridges, co-requisite
joint teaching, soft targets and context caps. Linear loops constructing solver
variables and auditing typed calendars are permitted. Context remains outside the
graph. No LLM creates schedule content. Preserve provisional/engineering approval
labels, source chunk anchors and failed-build diagnostics; do not fabricate time
padding or call a synthetic calendar a classroom-ready curriculum. Historical
contracts remain frozen; new solver contracts are exported separately.

## Phase 5 amendments (2026-09-14)

The maintainer authorizes versioned Spanish prompt variants, real three-replica
Vertex experiments and retained provider observations. Decision 0006 governs
selection. Agent-authored cases must never be labeled human annotations; production
promotion requires identified human review. Prompt metadata and evaluation contracts
are centralized under schemas. Semantic reference scoring is explicitly uncertain.
Compressed real provider observations may be retained under data/processed for
reproducible offline evaluation; no fabricated response may be called a cassette.
The historical prompt-rendered requests remain compatible despite metadata migration.

## Phase 4 amendments (2026-09-14)

The maintainer authorizes four LangGraph levels with local evaluator–optimizer loops,
real Vertex calls and recorded responses for offline verification. Decision 0005
specifies domain routing, voting, bounded revision semantics and publication gates.
Root/domain coordinators revise plans and child instructions; they do not bypass
child evaluation by rewriting approved micro-skills. SQLite checkpoints and provider
caches remain ignored except attributed, scrubbed test cassettes. Sparse connected
components implement cycle checks; semantic duplication and pedagogical prerequisites
remain specialized judge responsibilities, not a claimed proof from graph topology.
Agent state, reducers and transport-resource holders belong in schemas. Versioned
agent-prompt metadata is separate from the frozen Phase 2 evidence-review metadata.

## Phase 3 amendments (2026-09-14)

The maintainer explicitly authorizes live ingestion, GCP embeddings, measured parsing
and indexing, and three cropped real PDF golden fixtures. Decision 0004 governs
these additions. Full corpus bytes remain ignored; `tests/golden/pdfs/*.pdf` are
attributed test-only exceptions. Pure record loops are permitted for PDF geometry,
network scheduling and manifests; vector similarity uses NumPy. Stateful transport
resources do not constitute domain contracts. Historical graph contracts remain
readable, while evidence-ready contracts require resolvable chunk anchors. The
requested phases in chat take precedence over the obsolete phase numbering below.

## Phase 2 amendments (2026-09-14)

The maintainer's explicit Phase 2 request authorizes repository scaffolding, frozen
Pydantic v2 contracts, semantic graph diffing and an executable license gate. The
following amendments take precedence over conflicting legacy sections below;
their enforcement is recorded in decisions 0002 and 0003.

- Keep the existing package/distribution name. Centralize **all** data contracts in
  `src/goes_natural_science_kg/schemas/`, including decisions and workflow state.
  Configuration is `config.py`; use `agents/`, `curriculum/` and `eval/` boundaries.
- Prompts are root `prompts/*.prompt` files with typed YAML front-matter and Spanish
  bodies. This replaces the earlier Python-constant prompt layout while retaining
  exactly two Markdown documents and English identifiers.
- IDs use an immutable canonical identity namespace/key, readable slug and a
  16-hex SHA-256 suffix. Mutable labels, grades, minutes, evidence and edge payloads
  belong to revision content, not identity. Full revision SHA-256 is separate.
  This replaces the grade-bearing node IDs and endpoint-only edge hashes below.
- The Phase 2 edge enum is PREREQUISITE / CO_REQUISITE / REFINES / TRANSFERS_TO.
  ContextTag belongs only to curricularization, never to a graph node or edge.
  Inline prerequisites are a checked view of snapshot edges. They do not encode
  alternative AND/OR groups; decision 0001 remains proposed for a later design phase.
- Graph contracts enforce referential integrity; pedagogical dependency inference,
  cycle/route solving and curriculum generation remain future work. Do not claim
  their unimplemented invariants have been tested in this phase.
- Small typed-record validation, manifest handling and keyed semantic diffing may
  use linear Python iteration. Numeric graph algorithms still follow Invariant 4.
  Benchmarks cover the implemented operations; compare >20% mean regressions only
  on a compatible reference runner, never a macOS baseline against a Linux runner.
- The single licence vocabulary is in `schemas/corpus.py`; the packaged policy is
  `corpus/allowlist.yaml`. Official-publication admission is not an open copyright
  licence. Unknown rights require explicit review. Curriculum alignment and node
  grounding remain distinct future checks; this gate does not establish grounding.
- SourceDocument manifests include pending/accepted/rejected lifecycle contracts;
  persisted corpus manifests contain completed verdicts. Hash/size/download time
  are nullable until bytes are fetched, and must then be supplied together.
- Preserve the existing ignored `.env`; do not migrate or rewrite its credentials.
  `.env.example` documents the new namespace and optional GCP ADC configuration.
- No git initialization, commit, push, cloud deployment or live corpus acquisition
  is implied by this scaffolding phase.

This file is the normative contract for any AI agent (and any human) working in this
repository. It is one of exactly two Markdown files allowed in the tree (the other is
`README.md`). Sibling repositories `goes-math-kg` and `goes-linguistics-kg` cited a
`CLAUDE.md` that was later deleted; here it stays versioned and tests reference it by
invariant number.

Conversation with the maintainer happens in Spanish (voseo). Everything that lands in
the repository is in English, except LLM prompt bodies (see Invariant 1).

---

## 0. Purpose of the repository

Given a high-level skill map as input, this repository **computes** a Natural Science
curriculum for El Salvador, grades 2 through 6, under a hard budget of **160 hours per
grade** (9 600 minutes). The deliverable is a reproducible pipeline and its versioned
artifacts, not a document.

"Computes" means: same input + same commit + same seed ⇒ byte-identical output.
Any stage that calls an LLM caches its response under a SHA-256 of the full request
and versions that cache. Nothing that exists only in a chat transcript counts as a
result.

---

## 1. Roles

Every agent acting here speaks from three profiles at once and **names the profile**
whenever they pull in different directions:

- **Curriculum designer** — fluent in international assessment frameworks: PISA
  (science competencies), ERCE/LLECE, TIMSS (cognitive domains), BNCC (Ciências da
  Natureza), NGSS (DCIs, SEPs, CCCs), plus the MINED El Salvador programa de estudio.
- **Graph-theory researcher** — knowledge structures as DAGs, prerequisite inference,
  learning-sequence optimisation, topological constraints, spectral/sparse methods.
- **AI software engineer** — the product is a deterministic, audited pipeline with
  tests, benchmarks and manifests. If a decision cannot be tested it is not a decision.

When a curriculum-design preference and a graph constraint conflict (e.g. a pedagogically
desirable order that creates a cycle), say so explicitly, write a `decisions/` entry, and
let the maintainer choose. Do not resolve it silently in code.

---

## 2. Invariants (non-negotiable)

Each invariant has a number. Tests and `decisions/` entries cite them as `Invariant N`.

### Invariant 1 — Language
All code, identifiers, docstrings, comments, commit messages, log messages, test names,
YAML keys and README prose are in **English**. LLM **prompt bodies** are in **Spanish**
(Salvadoran register) and are the only Spanish allowed in `src/`. Verbatim quotes from
Spanish sources inside docstrings or evidence fields are permitted because translating
evidence damages provenance. A meta-test scans `src/`, `tests/` **and** `scripts/`
(the siblings only scanned `src/`, which is how 106 Spanish test names and Spanish
script filenames leaked into `goes-math-kg`).

### Invariant 2 — Documentation lives in code
The only Markdown files are `README.md` and this `CLAUDE.md`. Everything else:
- architecture decisions → `decisions/NNNN-<slug>.yaml`, validated by a Pydantic schema
  (`src/goes_natural_science_kg/decisions/schema.py`) in a test;
- data contracts → Pydantic models in `src/goes_natural_science_kg/**/schemas.py`
  (JSON Schema is an **export** from those models, never hand-written);
- MLOps / DataOps → a section of `README.md`.
If you feel the urge to write an explanatory `.md`, the code is not explaining itself.
Fix the code. The siblings accumulated 13+ `docs/*.md` and cited ADR numbers with no
ADR files; we do not repeat that.

### Invariant 3 — Tests test the function, not the mock
Mocking domain logic is forbidden. Mocks are allowed **only** at two edges:
- network downloads (HTTP), and
- LLM calls — and there only through **recorded cassettes of real responses**
  (`pytest-recording` / VCR, cassettes under `tests/cassettes/`, secrets scrubbed).
  Invented response strings are not cassettes.
Everything else is exercised for real:
- **golden datasets** under `tests/golden/`, versioned, with a SHA-256 manifest;
- **property-based tests** (`hypothesis`) for graph invariants (acyclicity, closure
  under prerequisite transitivity, grade monotonicity, ID uniqueness, determinism
  under node/edge permutation);
- **constraint tests** for the curriculum (budget ≤ 9 600 min per grade, full
  coverage of the input skill map, acyclicity, no backward-grade prerequisite).
A test that would still pass if you deleted the body of the function under test is not
a test. The siblings never adopted `hypothesis`, benchmarks or cassettes; this repo does,
deliberately.

### Invariant 4 — Measured performance
Every function that walks the graph or the corpus has a `pytest-benchmark` case under
`tests/benchmarks/`. Baselines are committed (`tests/benchmarks/baseline.json`) and CI
fails on **>20 % regression** (`--benchmark-compare-fail=mean:20%`).
Implementation rules:
- vectorise with `numpy` / `polars`; no per-row Python loops over corpus tables;
- adjacency and reachability use `scipy.sparse` (CSR); `networkx` is allowed for
  algorithms with no sparse equivalent and must be wrapped, not spread through the code;
- OOP only where there is **state with invariants to protect**: the graph store and the
  solver. Pure transformations are plain functions with typed inputs and outputs.

### Invariant 5 — Git hygiene and what gets committed
`.gitignore` excludes at least: `data/raw/`, `data/interim/`, `.cache/`, `.venv/`,
`*.pdf`, LangGraph checkpoints (`.langgraph/`, `*.sqlite`, `checkpoints/`), `.env*`
except `.env.example`, `*.pem`, `*.key`, `service-account*.json`, `__pycache__/`,
`.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, `.uv-cache/`, `reports/`, `var/`.
- The **corpus is never committed**. Its **manifest** is:
  `data/manifests/corpus.jsonl`, one record per source with `resource_id`, `url`,
  `license`, `license_evidence`, `sha256`, `bytes`, `downloaded_at`, `status`
  (`accepted | rejected`), `rejection_reason`. The corpus is re-downloaded from it.
- `data/processed/` and the graph artifacts (`data/graph/`) **are committed**: they are
  the product. This deliberately departs from `goes-math-kg` ADR 0002 (bucket-only
  artifacts); the reason is auditability without GCS credentials. If an artifact
  exceeds 5 MB, split it or justify it in a `decisions/` entry — do not move it to a
  bucket silently.
- A test asks `git check-ignore` (not the `.gitignore` text) for every sensitive path,
  as `goes-linguistics-kg` does after its 2026-08-26 `.gitignore` regression.

### Invariant 6 — Licences
Only sources with an **open licence** or **official government publication** enter the
corpus. The allowlist of licence labels and trusted domains is defined once, in
`src/goes_natural_science_kg/corpus/licenses.py`, as canonical **enums** (no free-form
strings — `goes-math-kg` ended with `"CC BY 4.0"`, `"CC-BY-4.0"` and `"CC BY 4.0
(licencia indicada…)"` as three different labels). The concrete allowlist is fixed in
Phase 2 (corpus acquisition) and recorded as a `decisions/` entry. No verifiable
licence ⇒ the source is recorded in the manifest as `rejected` with a reason; it is
never downloaded. Curriculum documents (programas de estudio) are evidence for
alignment but never ground a node, per the circular-coverage argument in
`goes-math-kg`'s `national_corpus_policy.py`.

### Invariant 7 — Secrets
No credentials in code, config or committed data. Configuration is `pydantic-settings`
(`src/goes_natural_science_kg/settings.py`) reading `.env`; only `.env.example` is
tracked (names and comments, no values). Environment variable prefix:
`GOES_NATURAL_SCIENCE_KG_`. Nested settings use the `__` delimiter, as in
`goes-linguistics-kg`. Note: the `.env` already present in this directory was copied
from `goes-math-kg` and carries `GOES_MATH_KG_*` names; it must be replaced, and
`.gitignore` must exist before `git init`.

### Invariant 8 — Determinism
- Every run takes an explicit `--seed`; the seed, the git SHA, `git_dirty`, the input
  SHA-256 and the settings hash are written to every artifact manifest.
- No persistent identifier may derive from Python's randomised `hash()`; a test runs
  ID minting under two `PYTHONHASHSEED` values.
- Iteration over sets/dicts that reaches an artifact is sorted first.
- LLM calls: `temperature=0`, response cached at `.cache/llm/<sha256>.json` where the
  key is `sha256(json.dumps({prompt, input, model, schema, prompt_version,
  cache_version}, sort_keys=True))` — the pattern of `goes-math-kg`
  `expert_review/transport.py`. Cache misses in CI are a failure: CI runs offline.
- Timestamps in artifacts come from a single injected clock, never `datetime.now()`
  inline.

---

## 3. Consistency with sibling repositories

Both siblings were read in full before this file was written. We inherit what is
consistent between them and correct what is not. Divergences are listed; none is silent.

### 3.1 Inherited (same as `goes-math-kg`, which is the more disciplined of the two)
- **Layout**: src-layout, `src/goes_natural_science_kg/` (distribution
  `goes-natural-science-kg`), `tests/unit/`, `tests/integration/`, `scripts/`,
  `config/*.yml`, `Makefile`, `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`,
  `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`, `LICENSE` (Apache-2.0),
  `NOTICE`, `.python-version` pins a Python `3.13` patch.
- **Module header** on every `.py` file:
  ```python
  # <path>.py
  # AI Depto
  # Copyright 2026 Gobierno de El Salvador.
  # SPDX-License-Identifier: Apache-2.0
  # Mission: <one line>
  ```
- **Tooling**: `uv` for everything (`uv sync --locked --group dev`, `uv lock --check`);
  `ruff` with `line-length = 100`, `select = ["E","W","F","I","UP","TID","B","SIM",
  "RUF","C901"]`, `ban-relative-imports = "all"`; `mypy --strict` over the **whole**
  package from day one (the math repo's ratchet never closed); `bandit`, `pip-audit`.
- **pre-commit**: pre-commit-hooks (trailing-whitespace, end-of-file-fixer, check-yaml,
  check-toml, check-merge-conflict, detect-private-key, check-added-large-files
  `--maxkb=5000`), ruff + ruff-format, `uv-lock`, `conventional-pre-commit` on
  `commit-msg` with types `feat fix refactor perf docs style test build ci chore revert`.
- **Makefile targets**: `setup lock lint format types security test bench quality ci`
  plus domain targets prefixed `kg-`. `make ci` runs the **full** test suite; the
  siblings' hand-listed CI subsets (8 of 300 files; 35 of 228) are the reason their
  local and CI gates diverged.
- **Commits**: Conventional Commits, English, imperative subject ≤ 72 chars, body says
  which invariant or decision the change serves.
- **LLM stack**: Google Vertex AI via `google-genai`, ADC auth, models pinned in
  `config/goes.yml`; proposer and validator must be different models (both siblings
  enforce `proposer != validator`). LangGraph for orchestration; checkpoints are ignored
  by git. xAI/Grok as optional cross-model auditor, degrading gracefully without a key.
- **Prompts**: Python string constants in `src/goes_natural_science_kg/prompts/*.py`,
  English constant names and comments, Spanish bodies, `str.format` placeholders,
  registered in a frozen `PromptSpec(id, version, task, purpose, template)` catalog.
  Prompt output contracts are strict JSON validated by Pydantic.
- **Artifact formats**: JSON for manifests and single objects, **JSONL** for node/edge/
  record streams (`nodes.jsonl`, `edges.jsonl`), CSV only for human-facing tables.
  Every payload carries `schema_version: "<family>/<major>.<minor>"`
  (e.g. `learning-graph-contract/1.0`). Every artifact directory carries a
  `manifest.json` with `{schema_version, git_sha, git_dirty, seed, inputs:{path: sha256},
  artifacts:{path: {sha256, bytes}}}`. No Parquet in committed artifacts (neither
  sibling uses it); Parquet is allowed under `data/interim/` for speed.
- **Resource IDs**: `res_` + `sha256(url)[:12]`, regex `^res_[0-9a-f]{12}$` — the scheme
  shared by `goes-math-kg` and `cgl_core`; `goes-linguistics-kg`'s ingestion layer uses
  `sha1(url)[:16]`, which its own core regex rejects. That is a bug there; we use sha256.
- **Store boundary**: an absent artifact, empty evidence and a hash mismatch are three
  distinct exceptions (`ArtifactUnavailable`, `EmptyEvidence`, `DigestMismatch`), never
  a silent fallback.

### 3.2 Graph vocabulary
Anchored on `goes-math-kg`'s `schemas/learning_graph_contract_v1.json`, the only
vocabulary in either sibling that is written as a contract. Adapted for science.

**Node kinds** (`NodeKind`, `StrEnum`):
`curriculum_skill` (input skill map, high level) → `specific_skill` → `micro_skill`;
plus `indicator` (MINED achievement indicator), `phenomenon` (anchoring phenomenon,
NGSS-style; replaces math's `representation`), `misconception`, `practice` (scientific
and engineering practice, PISA/NGSS), `crosscutting_concept`.

**Edge kinds** (`EdgeKind`, `StrEnum`):
`decomposes_to`, `prerequisite_of`, `assesses`, `requires`, `anchored_in`
(replaces `uses_representation`), `targets_misconception`, `aligns_to`, `supported_by`,
`applies_practice`, `expresses_concept`.

**Prerequisite kinds** (`PrereqKind`, on `prerequisite_of` only), following the sibling
pattern `<domain>ly_required | pedagogically_recommended | appears_before`:
`scientifically_required`, `pedagogically_recommended`, `appears_before`.

**Statuses**: `candidate | reviewed | published | rejected` for nodes and edges.

**Cognitive fields** (from `micro_skill_draft_v2` in math, TIMSS/PISA-aligned):
`cognitive_domain ∈ {knowing, applying, reasoning}`, `dok_level ∈ {1,2,3,4}`,
`pisa_competencies ⊆ {explain_phenomena_scientifically,
evaluate_and_design_scientific_enquiry, interpret_data_and_evidence_scientifically}`.
No `bloom_level` field (neither sibling has one; Bloom appears only as
`bloom_references`).

**Identifiers**:
- micro-skill: `^MS_G(0[2-6])_[A-Z]{2,3}\d+_\d{3}$` — the `goes-math-kg` ontology regex
  with grade restricted to 2..6. Strand codes (`[A-Z]{2,3}`) are fixed in the Phase 1
  `decisions/` entry (candidates: NGSS domains PS/LS/ESS/ETS vs. MINED units).
- specific skill: `^SS_G(0[2-6])_[A-Z0-9_]+$`; curriculum skill: `^CS_[A-Z0-9_]+$`.
- indicator: `^G(0[2-6])-I\d+$`; misconception `^MC_[A-Z0-9_]+$`; phenomenon
  `^PH_[A-Z0-9_]+$`; edge id `e_` + `sha256(f"{src}|{kind}|{dst}")[:14]`.
- Exactly **one** ID scheme per entity, validated by the Pydantic model. `goes-math-kg`
  ships three incompatible micro-skill schemes; that is a defect we do not import.

**Grade and time**: `grade: int`, `ge=2, le=6`, one constant `GRADES = (2, 3, 4, 5, 6)`
(the siblings disagree between 1..6, 1..11, 1..12 and 2..6). Time is always integer
**minutes** (`minutes`, `budget_minutes`); no `hours` field, as in both siblings.
`BUDGET_MINUTES_PER_GRADE = 9_600`. Session length (45 vs 60 min) is **unresolved in
`goes-linguistics-kg`** (`status: pendiente_confirmacion`); here it is a `decisions/`
entry with `status: pending` until MINED confirms, and the solver treats it as a
parameter, never a literal.

### 3.3 Deliberate divergences from the siblings (each with its reason)
| Topic | Siblings | Here | Why |
|---|---|---|---|
| Decisions | `docs/adr/*.md` (math, 4 files, one in Spanish) / none (linguistics, 90+ dangling `ADR-00xx` citations) | `decisions/*.yaml`, schema-validated | Invariant 2; machine-checkable `enforced_by` |
| Docs | 13+ `.md` in `docs/` | `README.md` only | Invariant 2 |
| `data/` layout | `corpus/ reference/ graph/ release/ releases/` (math) / everything under ignored `var/artefactos/{bronze,silver,gold}` (linguistics) | `data/{raw,interim,processed,manifests,graph}` | Invariant 5; both siblings gitignore `data/**` entirely and rely on GCS |
| Artifacts in git | bucket-only + `ARTIFACTS.json` pointer | processed + graph committed | Invariant 5, auditability without credentials |
| Settings | `os.environ` ad hoc (math) / `pydantic-settings` (linguistics) | `pydantic-settings` | Invariant 7 |
| hypothesis / benchmark / VCR | none | required | Invariants 3, 4 |
| mypy | ratchet with hundreds of open errors (math) / none (linguistics) | `--strict`, whole package | avoid the ratchet never closing |
| Test dir | 228 flat files (linguistics) / `unit/` + `integration/` (math) | `unit/ integration/ benchmarks/ golden/ cassettes/` | discoverability, markers |
| Spanish test names | 106 (math), ~100 (linguistics) | zero, meta-tested | Invariant 1 |
| CI scope | hand-listed subsets | full suite, offline, coverage enforced | gate parity |
| `.env` in tree | present (untracked) in both | present only as `.env.example`; a test asserts | Invariant 7 |

### 3.4 Errors found in the siblings worth telling the maintainer
Listed so they can be fixed there; none is copied here.
1. `goes-math-kg`: `CLAUDE.md` deleted but cited by ADR 0003, `INVARIANTES.md` and
   tests; `ARTIFACTS.json` committed with mode 600 and 108/507 stale entries;
   `schemas/` gitignored via `*.json` so `make schemas-check` validates nothing on a
   fresh clone; README says "no licence" while `LICENSE` is Apache-2.0; three
   micro-skill ID schemes; grade bounds 1..11 / 1..12 / 1..6 / 2..6 across models;
   `mccabe max-complexity = 44`.
2. `goes-linguistics-kg`: no ADR files despite 90+ citations; two `res_` ID schemes
   (sha1 vs sha256); three node-type vocabularies with `TipoArista` defined twice with
   disjoint values; ~200 `var/candidates/*` runs containing copies of `src/` and `tests/`
   (427 stray `.py`); CI disables coverage with `-o addopts=`; personal e-mail as
   pipeline identity in tracked `config.yaml`; `reserva_de_la_clase_oficial` has an
   empty `source`.

---

## 4. Repository layout (target; not created until Phase 0 is approved)

```
goes-natural-science-kg/
├── CLAUDE.md  README.md  LICENSE  NOTICE
├── pyproject.toml  uv.lock  .python-version  Makefile
├── .env.example  .gitignore  .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── config/goes.yml                  # models, budgets, cache_version (no secrets)
├── decisions/                       # NNNN-<slug>.yaml, schema-validated
├── data/
│   ├── raw/        (ignored)        # downloaded corpus, re-fetchable from manifests
│   ├── interim/    (ignored)        # parquet/intermediate tables
│   ├── manifests/                   # corpus.jsonl, skill_map input + sha256
│   ├── processed/                   # committed product tables (jsonl/csv)
│   └── graph/<YYYYMMDD>_<label>/    # nodes.jsonl edges.jsonl manifest.json
├── src/goes_natural_science_kg/
│   ├── settings.py                  # pydantic-settings
│   ├── decisions/schema.py
│   ├── corpus/     (manifest, licenses, download, extract)
│   ├── graph/      (schemas.py, store.py [stateful], sparse.py, invariants.py)
│   ├── solver/     (budget allocation, sequencing; stateful)
│   ├── prompts/    (Spanish bodies, English names)
│   ├── pipeline/   (LangGraph stages, pure functions between them)
│   └── cli.py      (typer; console script `goes-science`)
├── scripts/                         # thin wrappers only; logic lives in src
└── tests/
    ├── unit/  integration/  benchmarks/  golden/  cassettes/
    └── conftest.py                  # autouse offline fixture, injected clock, seed
```

---

## 5. `decisions/` entry schema

```yaml
id: 0001                     # zero-padded, monotonic
title: <short English title>
status: proposed | accepted | superseded | pending
date: 2026-09-13
supersedes: null | 0000
invariants: [1, 5]           # which invariants this serves or constrains
context: |
  <why a decision was needed>
options:
  - name: <option>
    pros: [..]
    cons: [..]
criteria: [..]               # what the choice was judged on
decision: <chosen option and one-paragraph rationale>
consequences: [..]
enforced_by: [tests/unit/test_x.py::test_y]   # may be empty only while status=proposed
```
A test validates every file against the Pydantic model and checks that `enforced_by`
paths exist for `accepted` entries.

---

## 6. How we work

- **Phases** (each gated by the maintainer's explicit OK, given in chat): 0 scaffold and
  tooling → 1 schemas and graph vocabulary → 2 corpus manifest and licences → 3 graph
  construction → 4 solver and budget → 5 curriculum artifacts and README. Before each
  phase, present the plan and **wait**. Do not start the next phase on your own.
- Any decision with a real trade-off (data structure, framework, curricular
  granularity, ID scheme, session length) becomes a `decisions/` entry with options,
  criteria and the chosen option, written **before** the code that depends on it.
- If a maintainer requirement is ambiguous or wrong, say so **before** implementing.
  If the maintainer reaffirms it, implement it as stated and record the concern in the
  relevant `decisions/` entry.
- State the role you are speaking from whenever roles pull apart.
- Never report a result that lives only in the conversation; point to the artifact,
  test or decision file that carries it.
- Never `git init`, commit or push unless asked. When asked, commits end with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## 7. Definition of done for any change

1. `make quality` passes: `uv lock --check`, ruff, ruff-format, `mypy --strict`,
   bandit, full pytest with coverage, benchmarks within 20 % of baseline.
2. New graph or corpus function ⇒ a benchmark case and a hypothesis property.
3. New LLM stage ⇒ a cassette from a real call, a cache key test, a `PromptSpec`
   version bump if the prompt changed.
4. New artifact ⇒ `schema_version`, `manifest.json`, a golden test, and a line in the
   README artifact table.
5. New trade-off ⇒ a `decisions/` entry with `enforced_by` populated.
6. Two runs with the same seed produce byte-identical `data/graph/**` (asserted by
   `tests/integration/test_determinism.py`).
