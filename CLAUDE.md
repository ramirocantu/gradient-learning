# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Gradient is a **single-user, backend-only** multi-domain study system: a FastAPI + Postgres
service exposing a JSON API at `/api/v1/*` (plus `/healthz` and `/media/*`). It ships **no view
layer** — clients (native macOS app, Chrome capture extension, MCP host) are external repos that
consume the HTTP contract. Forked from the `mcat-coach` PoC and rescoped to a generalized
course-builder.

`SPEC.md` is the source of truth — its caveman-encoded sections (`§G` goal, `§A` architecture,
`§C` constraints, `§I` interfaces, `§V` invariants, `§T` tasks, `§B` bug log) govern intent.
`FORMAT.md` defines that encoding. Read `SPEC.md` before any non-trivial change; `docs/BACKEND_CORE.md`
catalogs the live API surface and `docs/openapi.json` is the machine-readable contract.

## Spec-driven workflow (non-obvious — this repo runs on it)

Work is governed by skills that read/write `SPEC.md`, not freeform editing:
- **`spec`** — sole mutator of `SPEC.md` (write spec, amend a §, add invariants).
- **`build`** — plan-then-execute a `§T` task (`build §T.3`, `build --next`); flips the §T status cell.
- **`check`** — read-only drift detector: diffs `SPEC.md` against code, reports `§V` violations.
- **`backprop`** — on a bug/test failure, trace cause and append a `§B` row + optional new `§V` invariant.

When fixing a bug or a failing test, follow backprop: every bug should become a `§B` row. Invariants
are numbered monotonically (`V<N>` / `V-<area><n>`) and **never reused**.

## Commands

mise drives everything (toolchain via uv, env, tasks). Run tasks with `mise run <task>`:

| Task | What |
|---|---|
| `mise run setup` | install deps + start Postgres + migrate (first-time) |
| `mise run dev` | uvicorn dev server (`--reload`) at `localhost:8000` |
| `mise run test` | full pytest suite |
| `mise run lint` | `ruff check .` |
| `mise run format` | `ruff format .` |
| `mise run typecheck` | `pyright` (basic mode, points at `.venv`) |
| `mise run check` | lint + typecheck + test (run before declaring done) |
| `mise run migrate` | `alembic upgrade head` against this branch's DB |
| `mise run db:psql` | psql into this branch's DB |

Run a single test (the venv bin is on PATH, so bare `pytest`/`ruff` work; or use `uv run`):
```bash
uv run pytest tests/test_kb_reads_api.py -k some_name -x
```

**Type checking: prefer the LSP.** Claude Code's official **pyright-lsp plugin is installed** and
auto-launches `pyright-langserver` from `.venv/bin` (via mise `_.path`). Use the **`LSP` tool** for
live diagnostics, hover types, and go-to-definition while editing — it's faster and more precise
than shelling out, and reads the same `[tool.pyright]` config. Reach for `mise run typecheck`
(`uv run pyright`) only for a full-repo batch check, e.g. before declaring work done.

Add a migration (review the generated file — autogenerate is imperfect with async engines + custom types):
```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

## Database per branch (gotcha)

`DATABASE_URL` is **derived per git branch** by mise — `main` → `gradient_main`, `feature/x` →
`gradient_feature_x`. Never set it manually. `gradient` (unsuffixed) is the pristine template that
`mise run worktree:setup` clones from on `git worktree add` (after `mise run hooks:install` once).
Tests pin to a separate `gradient_test` DB (set in `tests/conftest.py`, not branch-derived).
`mise run db:prune` drops orphan branch DBs (protects `gradient`/`gradient_test`).

## Configuration

- **Secrets** (`OPENAI_API_KEY`, `COACH_TOKEN`, `NOTION_*`) live **outside the repo** at
  `~/.config/gradient/secrets.json`, loaded via `mise.local.toml`. There is no `.env`.
- Non-secret knobs are in `mise.toml [env]`; everything else falls back to **`app/config.py`
  defaults** — that file is the canonical list of per-extractor knobs (model names, cache paths,
  scheduler intervals).
- `tests/conftest.py` sets `GRADIENT_DISABLE_DOTENV=1` and unsets secret env vars **before** the
  first `app.config` import (settings is an import-time singleton) — hence the intentional E402
  ignore on that file.

## Architecture: core + plugin

The engine is **domain-blind**; everything MCAT-specific is a registry-keyed plugin (proving the
seam, not privileged). AAMC outline / UWorld capture / AnKing tags are the *reference* plugins.

**Core:** `Course` + recursive `outline_nodes` tree (one table, arbitrary depth, `kind` label) ·
generic `<target>_tags(node_id)` over {question, anki_note, atomic_fact, notion_page} ·
`Question` + `Attempt` (open `source` discriminator) · atomic-fact store + PDF-ingest · pgvector
recall · `concept_edges` (cross-node links) · Anki sync/retention via AnkiConnect · Notion
write-out · LLM tagging behind `services/llm/`.

**Plugins (periphery):**
- **Source adapters** (`app/services/adapters/`) — `capture → {Question, Attempt}`, keyed by `source`. UWorld = reference.
- **Outline-schema importers** (`app/services/outline/importer.py`) — validate + materialize an uploaded `{course, nodes}` schema. AAMC = bundled example at `app/seeds/aamc_outline.schema.json`.
- **Anki tag-shape parsers** (`app/services/anki/tag_parser.py`) — `tag string → node ref`, per deck.

### Code layout (`app/`)
- `main.py` — FastAPI entry (`app.main:app`); mounts the `/api/v1` router + `/media` + `/healthz`. No sub-apps.
- `api/v1/*.py` — route handlers; each `router` is included in `main.py`. `api/deps.py` = settings/session/auth deps.
- `models/` — SQLAlchemy models · `schemas/` — Pydantic · `services/` — business logic.
- `services/llm/` — OpenAI client, content-hash cache, grounded tagging, logprobs calibrator.
- `services/kb/` — embeddings, recall, persist_tags, pdf_ingest, notion, inbox, jobs.
- `scheduler.py` — APScheduler jobs (anki sync/assignment/review, pdf_ingest, notion_sync, embed, grounded_tag); each writes a `TaskRun` row with an in-flight guard.

### LLM rules (hard constraints from §C)
- **OpenAI only**, single provider, behind `services/llm/`. `OPENAI_BASE_URL` stays configurable for OpenAI-compatible local servers.
- Calibrator model **must support `logprobs`** (standard chat model, **not** an o-series reasoning model).
- Pattern: content-hash cache + `extractor_version` + token-cost log + structured output.
- **Cognitive-safety rule:** AI tags/summarizes/links/drafts; it must **never** generate primary active-recall questions or flashcards.
- The LLM4Tag grounded path (`services/kb/` + `services/llm/grounded.py`) is the **sole** categorization engine — the legacy MCAT categorizer and anki topic-resolver were cut.

### Other invariants worth knowing
- **Backend-only:** never re-add a server-rendered or bundled view layer. The HTTP contract is the boundary.
- **Notion = write-out only:** one page per outline node, facts as blocks, pointer index. No read-back, no local content copy.
- **Auth:** most `/api/v1/*` routes require the `X-Coach-Token` header (`verify_coach_token` in `api/deps.py`); `/admin/*` mutations are localhost-only; `/healthz` + `/media/*` are open.
- **Outline creation = upload a schema**, not in-app PDF parsing. Gradient owns validate + materialize only.
- `Attempt.time_seconds` is **not** actionable (carried constraint — don't weight by it).

## CodeGraph

This repo has a CodeGraph MCP server (`codegraph_*` tools) — a tree-sitter knowledge graph of every
symbol/edge/file. Prefer it for **structural** questions (what calls what, where defined, impact of a
change, signatures) over grep. Use `codegraph_context` first, then one `codegraph_explore`. Use grep
only for literal text (string/comment contents). Trust its results; don't re-verify with grep. The
index lags writes ~500ms — don't re-query immediately after an edit.
