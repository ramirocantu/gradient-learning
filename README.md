# Gradient

A single-user, multi-domain study system. **Backend-only**: a FastAPI +
Postgres service exposing a JSON API at `/api/v1/*` (plus `/media/*` for
assets and an MCP tutor seam). It ships no view layer — clients (a native
macOS app, the Chrome capture extension, and the MCP host) are external and
consume the HTTP contract. Forked from the `mcat-coach` PoC and rescoped to a
generalized course-builder: import an outline schema per course (AAMC is one
example), link captured questions / Anki cards / lecture-PDF atomic facts to
nodes in that tree, draft a Notion wiki out of it.

`SPEC.md` is the source of truth — read that for the canonical goal,
architecture, invariants, and task list. `docs/BACKEND_CORE.md` catalogs the
API surface, MCP seam, and reusable services a client builds against
(`docs/openapi.json` is the machine-readable contract). This README covers
local setup.

## Requirements

- Python 3.12+
- Docker (for Postgres)
- An OpenAI API key (or any OpenAI-compatible local server — set
  `OPENAI_BASE_URL`)

## Setup

**1. Start Postgres:**
```bash
docker compose up -d
```

**2. Create your env file:**
```bash
cp .env.example .env
```
Edit `.env` and set `OPENAI_API_KEY`. All other defaults match the Docker
config and work as-is locally. Model selection (`OPENAI_MODEL`,
`OPENAI_CALIBRATOR_MODEL`) is documented inline in `.env.example`.

**3. Create the virtualenv and install dependencies:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**4. Run migrations:**
```bash
alembic upgrade head
```

**5. (Optional) Seed the AAMC outline:**
```bash
# Creates a course slug=aamc + materializes the 1554-node MCAT outline.
curl -X POST localhost:8000/api/v1/courses \
    -H 'content-type: application/json' \
    -d '{"slug":"aamc","name":"MCAT — AAMC Content Outline"}'

curl -X POST localhost:8000/api/v1/courses/1/outline:import \
    -H 'content-type: application/json' \
    --data-binary @app/seeds/aamc_outline.schema.json
```
The same endpoint accepts any schema upload — uploading any other
`{course, nodes}` file materializes that course. AAMC has no privileged
position; it's just the bundled example (§V-O3).

**6. Start the server:**
```bash
uvicorn app.main:app --reload
```

API at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Verifying it works

```bash
curl localhost:8000/healthz
# {"status":"ok"}
```

## Running tests

```bash
pytest
```

## Project layout

```
.
├── pyproject.toml        # dependencies and project metadata
├── alembic.ini           # Alembic config
├── alembic/
│   ├── env.py            # async migration environment
│   └── versions/         # migration files
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── config.py         # settings loaded from env
│   ├── database.py       # SQLAlchemy async engine and Base
│   ├── api/v1/           # route handlers (the public JSON API)
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # business logic (categorizer, anki, analyzer, …)
│   ├── seeds/            # bundled schema files (AAMC outline reference)
│   └── web/media.py      # /media/* asset file-server (no view layer)
├── docs/                 # BACKEND_CORE.md + generated openapi.json
├── scripts/              # one-shot CLIs (e.g. the V-L2 gate runner)
└── tests/                # pytest suite + fixtures
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Async Postgres URL (`postgresql+asyncpg://...`) |
| `OPENAI_API_KEY` | Yes | API key — used by every LLM-touching service (categorizer, anki topic resolver, feature extractor, synthesizer, calibrator) |
| `OPENAI_BASE_URL` | No | Optional OpenAI-compatible base URL — set this to swap to a local server (vLLM / lm-studio) without changing code |
| `OPENAI_MODEL` | No | Default chat model (T5 spike: `gpt-4.1-mini`) |
| `OPENAI_CALIBRATOR_MODEL` | No | Logprobs-capable chat model for confidence calibration (T5: `gpt-4.1-mini`). MUST NOT be an o-series reasoning model |
| `COACH_TOKEN` | Yes | Shared secret the Chrome extension sends in `X-Coach-Token` |

See `.env.example` for the full per-extractor knob set (cache paths, scheduler
intervals, Anki integration).

## Adding a migration

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Autogenerate compares your SQLAlchemy models against the live schema.
Always review the generated file before committing — autogenerate isn't
perfect with async engines and custom types.
