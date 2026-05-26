# Gradient — Backend

FastAPI backend for the Gradient Learning system. Receives question captures from the Chrome extension, stores them in Postgres, categorizes them against the AAMC content outline, and serves analytics to the dashboard.

## Requirements

- Python 3.12+
- Docker (for Postgres)

## Setup

**1. Start Postgres:**
```bash
# from the repo root
docker compose up -d
```

**2. Create your env file:**
```bash
# from the repo root
cp .env.example .env
```
Edit `.env` and set `ANTHROPIC_API_KEY` to your real key. All other defaults match the Docker config and work as-is locally.

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

**5. Start the server:**
```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

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
root/
├── pyproject.toml        # dependencies and project metadata
├── alembic.ini           # Alembic config
├── alembic/
│   ├── env.py            # async migration environment
│   └── versions/         # migration files
└── app/
    ├── main.py           # FastAPI app entry point
    ├── config.py         # settings loaded from env
    ├── database.py       # SQLAlchemy async engine and Base
    ├── api/              # route handlers
    ├── models/           # SQLAlchemy models
    ├── schemas/          # Pydantic schemas
    ├── services/         # business logic
    └── seeds/            # static data (AAMC outline, tag mappings)
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | Async Postgres URL (`postgresql+asyncpg://...`) |
| `ANTHROPIC_API_KEY` | Yes | API key for Claude — used by the feature extractor |
| `COACH_TOKEN` | Yes | Shared secret the Chrome extension sends in `X-Coach-Token` |

## Adding a migration

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

Autogenerate compares your SQLAlchemy models against the live schema. Always review the generated file before committing — autogenerate isn't perfect with async engines and custom types.
