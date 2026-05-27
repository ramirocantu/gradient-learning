# Backend Core — the seam clients build against

Gradient is **backend-only** (see `SPEC.md` §A, §C, V-D1). This repo ships no
view layer. Everything a client needs is reachable over HTTP at `/api/v1/*`
(+ `/healthz` and `/media/*`); the MCP tutor seam is a curated read/persist
subset of that API. Clients — a native macOS app, the Chrome capture
extension, and the MCP host — are external and built in a separate
phase/repo against this contract.

This doc is the curated catalog. `docs/openapi.json` is the machine-readable
contract (regenerate with the snippet at the bottom) for client codegen.

- **Stack:** Python 3.12+, FastAPI, SQLAlchemy async, Postgres 16 (asyncpg),
  OpenAI SDK, APScheduler. Entry point: `app/main.py:app`.
- **Auth:** most routes require the `X-Coach-Token` header (shared secret,
  `COACH_TOKEN`), enforced by `verify_coach_token` (`app/api/deps.py`).
  Course/outline reads are open; `/admin/*` mutation routes are localhost-only
  (no token); `/healthz` + `/media/*` are open.
- **CORS:** `chrome-extension://*` and `http(s)://localhost[:port]`
  (`app/main.py`). A native macOS app calling over loopback fits the localhost
  allowance; widen the regex if a client runs from another origin.

---

## 1. JSON API surface (`/api/v1/*`)

Auth column: 🔑 = `X-Coach-Token` required · 🌐 = open · 🏠 = localhost-only.

### Captures / ingest — `app/api/v1/captures.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/v1/captures` | 🔑 | Ingest a capture payload (extension); routes to the `source` adapter → `Question`/`Attempt` rows. |

### Courses + outline — `app/api/v1/outline.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/courses` | 🌐 | List courses. |
| POST | `/api/v1/courses` | 🌐 | Create a course. |
| POST | `/api/v1/courses/{course_id}/outline:import` | 🌐 | Validate-then-materialize an uploaded outline schema (atomic). Re-upload AAMC restores MCAT. |
| GET | `/api/v1/courses/{course_id}/outline` | 🌐 | Read the outline node tree. |

### Tutor (MCP read seam) — `app/api/v1/tutor.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/tutor/questions/by-qid/{qid}` | 🔑 | Question detail + tags + features. |
| GET | `/api/v1/tutor/questions/by-attempt-id/{attempt_id}` | 🔑 | Same, resolved via an attempt. |
| GET | `/api/v1/tutor/captures/recent` | 🔑 | Recent captures (default 5, max 50). Each row's `topics[]` = the question's resolved node tags (T38). |
| GET | `/api/v1/tutor/sessions/latest` | 🔑 | Latest session `test_id`. |
| GET | `/api/v1/tutor/sessions/recent` | 🔑 | Recent sessions (default 5, max 50). |
| GET | `/api/v1/tutor/sessions/{test_id}/summary` | 🔑 | Session summary + attempt aggregates + per-node breakdown (`by_topic`/`top_topics`, T38). |
| GET | `/api/v1/tutor/attempts/flagged` | 🔑 | Flagged attempts (default 20, max 100). |
| GET | `/api/v1/tutor/outline/nodes/search` | 🔑 | Search outline nodes (`q`, course slug, limit). |
| GET | `/api/v1/tutor/outline` | 🔑 | Outline tree for a course slug. |
| GET | `/api/v1/tutor/outline/nodes/{node_id}/subtree` | 🔑 | Subtree under a node (V-O1 set rollup). |
| GET | `/api/v1/tutor/healthz` | 🔑 | Tutor health (incl. DB validation). |

**Node-tag payload shapes (T38 — V-O1, V-T1, V-O5).** A question's canonical
`QuestionTag.node_id`s (non-overridden) resolve to outline-node labels via
`app/services/tutor/outline.py:resolve_node_labels`, which renders each
node's full `>>`-joined path (cross-course safe; unknown ids dropped):

- `captures/recent` → each row gains `topics: [{node_id, name, path, kind}]`,
  deduped per question, ordered by `node_id`. Empty list when untagged.
- `sessions/{test_id}/summary` → `by_topic: [{node_id, name, path, kind,
  attempt_count, correct_count, accuracy}]`. A question tagged to N nodes
  counts in each (set membership, V-O1 — ⊥ summed once); ordered by
  `node_id`. `top_topics` = same rows ranked by `attempt_count` desc
  (data ordering, ⊥ verdict — V-M1), capped at 5.

### PKM write-back — `app/api/v1/pkm.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/v1/pkm/discriminators` | 🔑 | Persist a discriminator factor (append-only, deduped by `(question_id, factor_text)`). |

### Attempt notes — `app/api/v1/attempts.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/attempts/{attempt_id}/notes` | 🔑 | List notes on an attempt. |
| POST | `/api/v1/attempts/{attempt_id}/notes` | 🔑 | Add a note. |
| DELETE | `/api/v1/attempts/notes/{note_id}` | 🔑 | Delete a note. |

### Anki — `app/api/v1/anki*.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/v1/anki/sync` | 🔑 | Trigger one-off AnkiConnect deck sync. |
| GET | `/api/v1/anki/review-queue` | 🔑 | Due/overdue cards. |
| GET | `/api/v1/anki/cards/by-qid/{qid}` | 🔑 | Cards tagged with a qid. |
| POST/GET | `/api/v1/anki/assignments` | 🔑 | Create / list assignments. |
| PATCH | `/api/v1/anki/assignments/{assignment_id}` | 🔑 | Mark skipped / completed-manual. |
| GET/POST | `/api/v1/anki/load-config` | 🔑 | Read / upsert daily load budget. |
| GET | `/api/v1/anki/load-adherence` | 🔑 | Load-adherence stats (default 30d). |
| POST/GET | `/api/v1/anki/reviews` | 🔑 | Create / list pending reviews. |

### Admin — `app/api/v1/admin.py`
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/admin/jobs` | 🔑 | Scheduler jobs + next-run times. |
| POST | `/api/v1/admin/jobs/{job_name}/trigger` | 🔑 | Run a job immediately (canonical job set = `_VALID_JOBS`). |
| POST | `/api/v1/admin/questions/{question_id}/recategorize` | 🏠 | Re-run the categorizer on a question. |
| POST | `/api/v1/admin/questions/{question_id}/tags` | 🏠 | Create a manual tag override. |
| DELETE | `/api/v1/admin/tags/{tag_id}` | 🏠 | Delete / override a tag. |

### Root
| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | 🌐 | Liveness (`{"status":"ok"}`). |
| GET | `/media/{file_path}` | 🌐 | Serve a hash-addressed asset (PDF/image) under `MEDIA_ROOT`. Path-traversal guarded. |

---

## 2. MCP tutor seam

The MCP server lives in a separate repo and is a thin HTTP client of the
`tutor` + `pkm` routes above (all `X-Coach-Token`). Tool → endpoint:

| MCP tool | Endpoint |
|---|---|
| `get_question` | `GET /api/v1/tutor/questions/by-qid/{qid}` |
| `get_question_by_attempt_id` | `GET /api/v1/tutor/questions/by-attempt-id/{attempt_id}` |
| `get_recent_captures` | `GET /api/v1/tutor/captures/recent` |
| `get_latest_session_id` | `GET /api/v1/tutor/sessions/latest` |
| `get_recent_sessions` | `GET /api/v1/tutor/sessions/recent` |
| `get_session_summary` | `GET /api/v1/tutor/sessions/{test_id}/summary` |
| `get_flagged_attempts` | `GET /api/v1/tutor/attempts/flagged` |
| `search_outline_nodes` | `GET /api/v1/tutor/outline/nodes/search` |
| `get_outline_tree` | `GET /api/v1/tutor/outline` |
| `get_node_subtree` | `GET /api/v1/tutor/outline/nodes/{node_id}/subtree` |
| `add_note` / notes | `app/api/v1/attempts.py` notes routes |
| `write_discriminator_factor` | `POST /api/v1/pkm/discriminators` |

Per V-M1: MCP tools are data-exposure + persist only — no verdicts/heuristics
in signatures; Socratic reasoning is host-side.

---

## 3. Reusable service entry points

Business logic a client-serving endpoint (or a future native-app backend
route) should reuse rather than reimplement:

| Concern | Module | Key entry point |
|---|---|---|
| Outline import (validate→materialize) | `app/services/outline.py` | schema validate + node materialization (atomic, V-O2) |
| Subtree rollup (V-O1 set, not sum) | `app/services/outline_subtree.py` | subtree membership for a node |
| Node lookup by path/slug | `app/services/categorizer/outline_lookup.py` | `OutlineLookup.load(session, course_slug=…)` |
| Capture ingestion / source routing | `app/services/ingest.py` | source-keyed adapter dispatch → `Question`/`Attempt` |
| Question tagging (OpenAI) | `app/services/categorizer/` | `tag_question(...)` |
| LLM client (retries baked in) | `app/services/llm/client.py` | `build_openai_client(max_retries=5)` (V41) |
| LLM result cache | `app/services/llm/cache.py`, `app/services/categorizer/cache.py` | content-hash SQLite cache |
| Anki sync / queries / assignments / reviews | `app/services/anki/*` | AnkiConnect client + sync/assignment/review services |
| Manual tag overrides | `app/services/admin_tags.py` | `create_manual_tag(...)` |
| Media storage | `app/services/media_store.py` | hash-keyed asset store (served by `/media/*`) |
| Tutor reads (MCP seam backing) | `app/services/tutor/*` | questions, captures, sessions, flags, outline, health, discriminators |

---

## 4. Data model (`app/models/`)

Core: `Course`, `OutlineNode` (recursive tree, `kind`/`depth`/`position`),
`Question`, `Attempt`, `Passage`, `RawCapture`, `QuestionTag` (target =
`node_id`), `AttemptNote`, `QuestionFeatures`, `DiscriminatorFactor`, `Media`,
`TaskRun`.

Anki: `AnkiNote`, `AnkiCard`, `AnkiNoteTag`, `AnkiCardReview`,
`AnkiAssignment`, `AnkiReview`, `AnkiWrite`, `AnkiLoadConfig`.

P2 KB substrate (models present, workflow lands later): `PdfSource`,
`AtomicFact`, `AtomicFactTag`, `ContentEmbedding` (pgvector), `ConceptEdge`,
`NotionPage`, `LlmBatchRun`.

Tag invariants (V-T1..V-T3): canonical `<target>_tags` shape, only target is
`node_id`, `source ∈ {schema_map, llm, manual}`, `confidence` required for
`llm` and `<0.5 ⇒ manual_review`.

---

## 5. Scheduler jobs (`app/scheduler.py`)

Started in the app lifespan when `SCHEDULER_ENABLED`. Each job records a
`TaskRun`; concurrent runs are guarded; transient OpenAI/Anki errors degrade
to partial success (V41).

| Job id | Trigger | Does |
|---|---|---|
| `run_categorizer` | every `CATEGORIZER_INTERVAL_MINUTES` (15) | Tag `needs_categorization` questions → `QuestionTag`. |
| `run_anki_sync` | every `ANKI_SYNC_INTERVAL_MINUTES` (15) | Sync deck + review state from AnkiConnect. |
| `run_anki_topic_resolver` | every `ANKI_TOPIC_RESOLVER_INTERVAL_MINUTES` (60) | Resolve Anki notes → outline nodes. |
| `run_anki_assignment_unlock` | every `…UNLOCK_INTERVAL_MINUTES` (60) | Unsuspend due assignments (+ addTags). |
| `run_anki_assignment_complete` | daily cron (05:15 UTC) | Mark fully-reviewed assignments complete. |
| `run_anki_review` | every `ANKI_REVIEW_PUSH_INTERVAL_MINUTES` (60) | Push due reviews (createFilteredDeck + addTags). |

`run_feature_extraction` exists but is **fenced** (not registered) — see V-RB1.

---

## 6. Where a native client plugs in

There is no in-repo UI to extend. A native macOS app (or any client):
1. Talks to `/api/v1/*` over HTTP with `X-Coach-Token`; generate a client from
   `docs/openapi.json`.
2. Fetches assets from `/media/{file_path}`.
3. For tutor/Socratic flows, either call the `tutor`/`pkm` routes directly or
   go through the MCP host (which wraps the same routes).
4. A view needing data the API doesn't expose **extends the public API**
   (`app/api/v1/*` + a service in `app/services/*`) — never a private/
   dashboard-only route (V-D1). The structural guard is
   `tests/test_backend_only_seam.py`.

---

## 7. Fenced / orphaned surfaces (NOT part of the live seam)

These survive in the tree but are **not** reachable from the live API — kept
per the frontend-only cull scope (SPEC V-RB1), not part of the contract a
client builds against. They are fenced (carry a `FENCED` marker, routes
commented out, jobs unregistered) and are candidates for a later cull.

| Surface | File(s) | Status |
|---|---|---|
| Mastery report service | `app/services/analytics.py` | FENCED; was called only by the deleted dashboard home route — now zero callers. |
| Study-next recommender | `app/services/recommender.py` | FENCED; same — zero callers after dashboard deletion. |
| Analytics / analyzer / recommendations routers | `app/api/v1/{analytics,analyzer,recommendations}.py` | Files present but **never mounted** (`include_router` calls commented out in `app/main.py`). Not in `/api/v1/*`. |
| Feature-extraction pipeline | `app/services/analyzer/*`, `run_feature_extraction_job` | Job **unregistered** in `start_scheduler` (FENCED, V-RB1). The `QuestionFeatures` it would populate is still surfaced in the tutor question payload, so the model + reader stay live; the writer (job) is off. |

Reactivation, if ever wanted, means porting onto `OutlineNode` + `outline_subtree`
(V-O5) and re-exposing via the public API — not a private route (V-D1).

Guards: `tests/test_fence_guards.py` asserts these stay fenced (FENCED marker
present, routers unmounted, feature-extraction job unregistered).

---

## Regenerating the contract

```bash
uv run python -c "import json; from app.main import app; \
  open('docs/openapi.json','w').write(json.dumps(app.openapi(), indent=2)+'\n')"
```

Interactive docs (when the server runs): `http://localhost:8000/docs`.
