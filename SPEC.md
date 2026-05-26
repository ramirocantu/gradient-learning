# SPEC — Gradient (generalized study tool)

Forward source of truth. Fork of `mcat-coach` (proof-of-concept, MCAT-bound) → **Gradient**: a domain-agnostic study engine. MCAT becomes one domain pack atop a thin core. Authored 2026-05-26 from interview + OpenAI pivot. The mcat-coach `SPEC.md` / `PLAN*.md` remain historical narrative for the PoC only; this file governs Gradient.

## §G — goal

A single-user, multi-domain study system. Add a domain (a course: biochem, anatomy, …) and the system links four things under it:
1. **ingress notes** (lecture PDFs) → grounded atomic facts → Notion wiki,
2. **practice questions** (web Qbank via extension, PDF sets, manual entry) → performance dashboard + resource links,
3. **Anki flashcards** → cards linked to course topics + retention,
4. **Notion** personal-knowledge system (write-out target).

Workflow:
1. Create a course → import an **outline schema** (user generates it from their own sources via a shipped prompt, then uploads) → base node tree + tags.
2. Ingress lecture PDFs → tag → grounded atomic facts → draft to Notion (one page per concept).
3. Ingress practice questions → tag → dashboard performance + resource links → Socratic review via MCP.

Architecture = **core + periphery**. The core is domain-blind. Everything MCAT-specific (AAMC outline, UWorld capture, AnKing tag shape) is a *plugin* — proving the seam, not privileged.

## §A — architecture (core vs plugin)

**CORE (engine, domain-blind):**
- `Course` + recursive `outline_nodes` tree (one table; arbitrary depth + `kind` label).
- Generic `Tag(node_id)` over any target {question, anki_card, atomic_fact, notion_page}.
- `Question` + `Attempt` with open `source` discriminator.
- Atomic-fact store + PDF-ingest pipeline.
- Vector recall (pgvector).
- `concept_edges` — cross-domain node↔node links.
- Anki sync/retention/assignment layer (AnkiConnect protocol).
- Notion writer (write-out + pointer index; no read-back).
- LLM tagging engine (OpenAI, behind `services/llm/`).
- MCP data + persist tools.
- JSON API (`/api/v1/*`) = the dashboard's sole data seam. Dashboard is a **client**, not core: P0 = server-rendered Jinja, P1 = React+Tailwind SPA (T16) over the same API. Swapping the view layer ⊥ touch core; the API contract is the boundary.

**PLUGINS (periphery, registry-keyed):**
- **Source adapters** — `capture → normalized {Question, Attempt}`, keyed by `source`. UWorld = reference adapter. Also: generic web-Qbank (extension), manual entry, PDF question-set parser.
- **Outline-schema importers** — validate + materialize an uploaded schema into `courses` + `outline_nodes`. AAMC outline = a bundled example schema file.
- **Anki tag-shape parsers** — `anki tag string → node ref`, keyed per deck/pack. AnKing-MCAT regex = reference parser.
- **Domain pack** = bundle of (outline schema + anki tag shape + question source) for one course.

Seam = the normalized internal model + adapter registries keyed on `source` / `course` / deck.

## §C — constraints

- Single user. Local-first **except Notion write-out** (sole cloud egress for storage) + **OpenAI API** (LLM + embeddings). Backend + Postgres stay local. ⊥ multi-user.
- Stack: Py 3.12 / FastAPI / SQLAlchemy async / Postgres 16 (asyncpg) / **OpenAI SDK** / APScheduler. Dashboard frontend = Jinja today → P1 migrates to a JS SPA (see frontend-stack carve-out below). Extension TS + MV3 (separate repo). Adds: pgvector (`vector` ext via SQLAlchemy), `notion-client`, PyMuPDF/pdfplumber.
- ⊥ new **backend** framework, ORM, queue. ⊥ a new ORM for pgvector — use existing SQLAlchemy.
- **Frontend-stack carve-out (amended 2026-05-26):** dashboard MAY adopt a JS framework (React + Tailwind) built via the `frontend-design` plugin. Backend stays FastAPI serving JSON; SPA consumes existing `/api/v1/*`. ⊥ new backend stack still holds; ⊥ the SPA reaching past the JSON API (no server-side render coupling).
- **LLM = OpenAI, single provider, behind `services/llm/`.** No local model required (vLLM dropped — logprobs are cloud-side). `OPENAI_BASE_URL` left configurable so an OpenAI-compatible local server can slot in later without code change.
- LLM use mirrors the proven pattern: content-hash cache + `extractor_version` + token-cost log + structured output. Anthropic-specific cache markers retired (OpenAI caching is automatic — see V38/V42).
- **Calibration** (LLM4Tag confidence) uses OpenAI logprobs. The calibrator model **must support `logprobs`** — i.e. a standard chat model (GPT-4o/4.1-class), **not** an o-series reasoning model. Tagging may use any model.
- Embeddings default = OpenAI `text-embedding-3-small` (pgvector dim 1536); BGE-local (`bge-base-en-v1.5`, dim 768) retained as a config swap. `embedding_version` stamps every row; provider/dim change ⇒ bump + re-embed. **(Open: confirm OpenAI vs local embeddings — not part of the explicit LLM pivot.)**
- Outline creation = **upload a schema**, not in-app PDF parsing. Gradient ships a prompt template; the user runs it against their own sources (PDF/screenshot/webpage) in their own LLM session, then uploads the resulting schema. Gradient owns validate + materialize only.
- Anki source = AnkiConnect HTTP. Read calls only by default; write allowlist (`unsuspend`, `addTags`, namespaced `createFilteredDeck`) carried from the PoC, ⊥ scheduler mutation. Per-deck tag-shape parser is a plugin.
- PDF corpus = user-uploaded classroom PDFs only; atomic-fact generation grounded to uploaded content. PyMuPDF/pdfplumber; local-dir poller.
- Notion = **write-out only**. One page per concept (outline node); atomic facts = blocks within. Store a pointer index (`notion_page_id`, `url`, `tags[]`, `node_id`) + back-link anchors. ⊥ read-back, ⊥ local content copy, ⊥ Notion-as-source-of-truth.
- Cognitive-safety (hard rule): AI tags / summarizes / links / drafts; ⊥ generate primary active-recall questions or flashcards.
- MCP role: data exposure + structured writes; LLM (host) = reasoner. ⊥ heuristics in tool signatures. Socratic dialogue host-side; discriminator tool persists only.
- `Attempt.time_seconds` ⊥ actionable (carried hard constraint).

## §I — interfaces

### Schema (target — generalized)

```
courses(id, slug UQ, name, description?, created_at)
outline_nodes(
  id, course_id FK→courses, parent_id FK→outline_nodes NULL,
  kind TEXT,           # per-course label: section|unit|lecture|concept|…
  name, depth, position, created_at)
  UQ(course_id, parent_id, name); IX(course_id), IX(parent_id)
  # AAMC = one course, a 4-deep instance (section→fc→cc→topic as kinds)

concept_edges(
  id, src_node_id FK, dst_node_id FK,
  kind CHECK IN ('similarity','manual'), score NUMERIC NULL, created_at)
  UQ(src_node_id, dst_node_id, kind)   # cross-domain links

questions(
  id, source TEXT, external_id TEXT,   # (source, external_id) UQ
  stem_html, stem_plain, choices JSONB, correct_choice,
  explanation_html?, explanation_plain?, source_meta JSONB?,
  needs_categorization BOOL, first_seen_at, last_updated_at)

attempts(
  id, question_id FK, attempted_at, selected_choice, is_correct,
  time_seconds?, flagged, session_ref TEXT?, source TEXT, created_at)

# canonical TAG shape — one table per target kind, all target node_id:
#   question_tags, anki_note_tags, atomic_fact_tags, notion_page_tags
#   (anki target = anki_note_tags per note-as-unit V75; anki_card_tags
#    dropped T95.  atomic_fact_tags + notion_page_tags = P2 — their target
#    tables don't exist yet; T2 retargets only question_tags + anki_note_tags.)
<target>_tags(
  id, <target>_id FK, node_id FK→outline_nodes,
  source CHECK IN ('schema_map','llm','manual'),
  confidence NUMERIC(3,2) NULL,   # required for llm, NULL otherwise
  rationale TEXT?, extractor_version TEXT?,
  manual_review BOOL DEFAULT false,   # Conf<0.5 flag (V69)
  is_overridden BOOL DEFAULT false, overridden_at?, created_at)
  UQ(<target>_id, node_id, source)

pdf_sources(id, course_id FK, filename, sha256 UQ, status, ingested_at)
atomic_facts(
  id, course_id FK, pdf_source_id FK, page?, text, node_id FK?,
  content_hash, created_at)

content_embeddings(
  id, entity_kind CHECK IN ('question','atomic_fact','outline_node'),
  entity_id, embedding vector(N), embedding_version)
  # pgvector; dim per EMBEDDING_MODEL

notion_pages(
  id, node_id FK UQ, notion_page_id, url, tags JSONB, last_synced_at)
discriminator_factors(
  id, question_id FK, factor_text, node_id?, notion_block_id?, created_at)

# carried (reuse): anki_notes, anki_note_tags, anki_cards, anki_card_reviews,
#   anki_assignments, anki_reviews, anki_load_config, task_runs
```

### Outline-schema import

```
# Shipped schema format (JSON/YAML), uploaded by user:
{ "course": {"slug","name","description?"},
  "nodes": [ {"path": ["Section","FC","CC","Topic"], "kind", "name",
              "disciplines?": [...], "position?"} , ... ] }
api: POST /api/v1/courses                         → Course
api: POST /api/v1/courses/{id}/outline:import     # body = schema; validate → materialize nodes
api: GET  /api/v1/courses/{id}/outline            → node tree
job: (sync) outline_import — validate, dedupe, build parent chain
docs: PROMPT_OUTLINE_SCHEMA.md — the template user runs against their own sources
seed: seeds/aamc_outline.schema.json — MCAT reference; uploading it restores MCAT outline
```

### Ingress (notes → atomic facts → Notion)

```
env: PDF_INBOX_DIR
api: POST /api/v1/pdf/ingest {course_id, file}    # or scheduler-polled inbox
job: run_pdf_ingest_job   # poll → extract → chunk → tag(node) → atomic facts → embed
job: run_notion_sync_job  # one-way: per node, upsert a Notion page; facts = blocks; back-links + pointer row
mcp: write_discriminator_factor(question_id, factor_text, node_id?)
api: POST /api/v1/pkm/discriminators → DiscriminatorFactor   # persist-only (X-Coach-Token)
```

### Practice questions + mastery

```
api: POST /api/v1/captures            # source-tagged; routes to source adapter
api: POST /api/v1/questions           # manual entry
api: GET  /api/v1/mastery/course/{id} → CourseSummary
api: GET  /api/v1/mastery/node/{id}   → NodeSummary   # subtree-membership rollup
job: run_categorizer_job   # tag needs_categorization questions vs course outline
job: run_embed_job         # embed new questions/facts/nodes
job: run_calibrate_job     # OpenAI logprob Conf; prune <0.5 → manual_review
ext: source adapters — uworld (reference) | web-qbank | (manual = api) | pdf-qset (later)
```

### Anki (reuse)

```
env: ANKICONNECT_URL=http://127.0.0.1:8765 ; ANKI_DECK_NAME ; ANKI_SYNC_INTERVAL_MINUTES
api: POST /api/v1/anki/sync ; GET /api/v1/anki/cards?node_id= ; GET review-queue ; load-adherence
mcp: sync_anki ; get_anki_review_queue ; get_anki_cards_for_node ; get_anki_performance(node_id?, window_days?)
job: run_anki_sync_job ; assignment/review jobs (carried)
plugin: anki tag-shape parser registry — AnKing-MCAT = reference; maps tag → node_id
```

### Env

```
DATABASE_URL                 # postgresql+asyncpg://…
OPENAI_API_KEY               # required
OPENAI_BASE_URL              # optional; OpenAI-compatible local server later
OPENAI_MODEL                 # tagging / facts / Socratic — pick in P0 spike
OPENAI_CALIBRATOR_MODEL      # MUST support logprobs (non-reasoning chat model)
EMBEDDING_MODEL              # default text-embedding-3-small (dim 1536)
COACH_TOKEN                  # X-Coach-Token shared secret (extension + MCP persist)
ANKICONNECT_URL ; ANKI_DECK_NAME ; ANKI_SYNC_INTERVAL_MINUTES
PDF_INBOX_DIR
NOTION_API_TOKEN ; NOTION_WIKI_DB_ID    # ⊥ commit
```

## §V — invariants

**Core / outline**
- V-O1: `outline_nodes` is the sole hierarchy. AAMC's 4 levels are expressed as `kind` labels on a 4-deep tree, ⊥ as dedicated tables. Rollup = subtree membership (set, not sum) — each item lives once at its most-specific node; a parent's set = union of descendants' + own direct items.
- V-O2: Outline import is **validate-then-materialize**. Reject (whole upload, atomically) on: missing course slug, duplicate node path, broken parent chain, depth/kind contradiction. ⊥ partial import.
- V-O3: An uploaded schema is **data**. Re-uploading AAMC restores MCAT; no MCAT logic is privileged in core.
- V-O4: Node-path delimiter reserved + ASCII; renderer + parser must agree (carried: ` >> `, ⊥ `/`,`-`,`.`,`,`). Schema importer rejects a node name containing the delimiter.

**Tags**
- V-T1: Canonical tag shape per target table; the only tag target is `node_id`. The PoC's `(topic|content_category|skill)` 3-target is retired.
- V-T2: `source ∈ {schema_map, llm, manual}` records HOW a tag was derived. Sync/regex/import write their own source rows only; re-run pattern = `DELETE WHERE target_id=X AND source='llm'; INSERT new`. `manual` + `schema_map` rows untouched by any LLM re-run (carried V24/V43).
- V-T3: `confidence` required for `source='llm'`, NULL otherwise; `< 0.5` ⇒ `manual_review=true`, ⊥ silently dropped at persist (changed from PoC: surface for review, don't discard).

**LLM (OpenAI)**
- V38 (RETIRED): no `cache_control` markers — OpenAI prompt caching is automatic. Delete the Anthropic ephemeral-cache attach logic.
- V42 (KEPT, stronger): candidate iteration ⊥ switch the cached-prefix dimension between adjacent calls. Automatic prefix caching still requires a stable prefix; sort candidates by the cache-key dimension (e.g. `course_id`/`cc`) for contiguous drain. With no manual control, ordering is the only lever — enforce it.
- V44 (KEPT): ship BOTH a numbered NL candidate list (model reasoning surface) AND an int-encoded enum `[1..N]` (grammar-constrained sampling). Re-measure jaccard on the chosen OpenAI model; the dual-surface insight is model-agnostic, the score is not.
- V45 (REWORKED): structured output via OpenAI structured outputs (`response_format: json_schema, strict:true`). Honor OpenAI's schema limits (enum count / total enum-string length / property count); for large enums apply V44 int-encoding **before** enabling strict. Server-side belt (id-range recheck, picks slice, confidence threshold) retained.
- V69 (AMENDED): confidence calibration = OpenAI logprobs. Discriminator Yes/No grade on a **plain completion** (not structured) so the single token is readable; `Conf = exp(L_yes)/(exp(L_yes)+exp(L_no))`; `<0.5` ⇒ `manual_review`. Calibrator model must support `logprobs`. **No local vLLM.**
- V41 (AMENDED): extractors survive transient OpenAI errors. (a) `AsyncOpenAI(max_retries≥5)`. (b) worker catches `openai.APIError`/`RateLimitError`/`InternalServerError` per item, logs WARN, breaks early, returns `partial_failure=True` + accumulated counts; scheduler always reaches `commit()`; `task_run.status='succeeded'` (partial — resumes next run via candidate filter).
- V-L1: token-cost log reads OpenAI `usage` incl. `prompt_tokens_details.cached_tokens`; cache-hit accounting from `cached_tokens`, not inferred.
- V16 (AMENDED): all LLM-touching code mocks **OpenAI** at the SDK boundary in tests; ⊥ real API calls in the suite.
- V-L2 (gate): P0 ships a measurement harness re-running the anki-topic-resolver / categorizer eval on the chosen OpenAI model. Tagging quality (jaccard / set-equality vs the PoC's Claude baseline) is recorded before any pivot is declared done; a regression blocks the pivot.

**Embeddings / recall**
- V-E1: `embedding_version` stamps every row; provider or dim change ⇒ bump + full re-embed. ⊥ mixed-dim vectors in one `content_embeddings` column.
- V-E2: similarity edges (`concept_edges.kind='similarity'`) are derived (cosine); manual edges are human-verified. Recall ⊥ weight `Attempt.time_seconds`.

**Notion (write-out)**
- V-N1: sync is one-way Postgres→Notion. ⊥ read Notion content back; ⊥ keep a local content copy. The only Notion state stored = `notion_pages` pointer (page_id, url, tags, node_id) for link + back-link.
- V-N2: one Notion page per outline node (concept granularity). Atomic facts render as blocks within the node page. Page identity keyed by `node_id` (UQ) — re-sync upserts, ⊥ duplicates.

**Anki (carried)**
- V13: AnkiConnect read calls + write allowlist (`unsuspend`, `addTags`, namespaced `createFilteredDeck`) only; ⊥ mutate scheduling (intervals/ease/due/position), ⊥ `suspend`/`removeNotes`/`forgetCards`/`deleteDecks`/etc.
- V21: `ANKICONNECT_URL` pins host `127.0.0.1` (⊥ `localhost` — IPv6 resolution → spurious unreachable).
- V22: AnkiConnect client split timeout `connect=5 / read=120`.
- V26/V27: `anki_card_reviews` append-only, incremental `startID=MAX+1`; retention windows {7d,30d,all}, pass = ease∈{2,3,4}, exclude `type='learn'`, computed locally.

**MCP / safety**
- V-M1: MCP tools = data exposure + persist only; ⊥ verdicts/heuristics in signatures. Socratic reasoning host-side; `write_discriminator_factor` persists (Postgres + Notion block + back-link).
- V-M2: AI ⊥ generate primary active-recall questions or flashcards (cognitive-safety hard rule).

**Dashboard / API seam**
- V-D1: dashboard is a client over the public JSON API (`/api/v1/*`) — its sole data seam. ⊥ dashboard-only backend endpoints; a view needing new data extends the public API, ⊥ a private route. View-layer swap (Jinja → SPA, T16) ⊥ touch core. The API contract is the boundary (carries §A).

## §P — phases

- **P0 — schema generalize + OpenAI pivot (≈wk1).** Collapse `Section/FC/CC/Topic` → `Course` + `outline_nodes`; retarget tags to `node_id`; open `source` enum on captures; swap `anthropic`→`openai` SDK across extractors; retire V38, rework V45, amend V41/V16/V69. Reseed AAMC as an uploaded schema. **Gate: V-L2 measurement harness green** (tagging quality vs Claude baseline). No new UX; unblocks all.
- **P1 — day-1 usable (≈wk2–3).** Outline-schema import endpoint + prompt template + validate/materialize. Anki sync/linking on the real course (reuse). → tag your real Anki deck to your real course. Both reuse-heavy, immediate value. **Dashboard redesign (T16):** whole `app/web/dashboard/` rebuilt as a React+Tailwind SPA via the `frontend-design` plugin, consuming existing `/api/v1/*` JSON; replaces Jinja. Plugin use: invoke the `frontend-design` skill per view, feed it the JSON contract + the current Jinja view as reference, iterate to a production-grade, non-generic aesthetic. Runs after T14 (read-services ported to `node_id`).
- **P2 — notes → atomic facts → Notion (early semester).** PDF ingest poller → grounded atomic facts → embed + tag → one Notion page per concept + pointer/back-links. Vector recall online.
- **P3 — practice questions.** Source adapters: web-Qbank (extension) → manual entry → PDF-qset parser (hardest, last). Dashboard performance + resource links per node.
- **P4 — Socratic MCP.** MCP tools over the data; `write_discriminator_factor`; host-side dialogue. Dashboard chat only if the MCP-host workflow proves clunky.

## §O — open items
- Embeddings provider: OpenAI `text-embedding-3-small` (dim 1536) vs BGE-local (dim 768). Default set to OpenAI for single-provider consistency; confirm.
- OpenAI model choices (`OPENAI_MODEL`, `OPENAI_CALIBRATOR_MODEL`) decided empirically in the P0 spike, not pinned here.
- Old MCAT/UWorld attempt *data*: assumed not migrated (fresh start). UWorld capture adapter + AnKing parser kept as example code plugins.
- `README.md` references a `backend/` subdir the repo doesn't have — fix during P0.

## §T — tasks

P0 — schema generalize + OpenAI pivot. Ids are monotonic, not positional: T12–T14 (dependent-module ports) are appended but run mid-phase. **Exec order (dependency-correct, I hand-drive — ⊥ `--next` id-order):** T1 → T2 → T15 → T3 → T12 → T13 → T14 → T4 → T5 → T6 → T7 → T8 → T9 → T10(gate) → T11. (T15 = DB rename, independent housekeeping, runs now.) Schema/tags + ports land before the OpenAI pivot so the suite compiles; gate last. **T16 = P1** (dashboard redesign), runs after T14 — listed here for monotonic id, not P0 exec order. See FORMAT.md for `st` legend.

| id | st | goal | cites |
|-----|----|------|-------|
| T1 | x | collapse Section/FC/CC/Topic → `courses` + recursive `outline_nodes` (kind/depth/position); migration + SQLAlchemy models | V-O1,V-O4,I.schema |
| T2 | x | retarget tags → `node_id`; canonical `<target>_tags` shape on `question_tags` + `anki_note_tags` (atomic_fact/notion_page tags = P2, tables not built); retire PoC 3-target (topic/cc/skill); `source` enum → {schema_map,llm,manual}; confidence NULL-able + `manual_review` | V-T1,V-T2,V-T3,I.schema |
| T3 | x | open `source` discriminator enum on questions/attempts; `/api/v1/captures` routes to source adapter registry | I.api,§A |
| T4 | x | swap `anthropic`→`openai` SDK in `services/llm/`; retire V38 `cache_control` markers; `AsyncOpenAI(max_retries≥5)`; mock OpenAI at SDK boundary in tests | V38,V41,V16,V-L1,§C |
| T5 | x | P0 spike: pick `OPENAI_MODEL` + `OPENAI_CALIBRATOR_MODEL` (logprobs-capable, non-reasoning chat model); record in `.env.example` | §C,§O |
| T6 | x | structured output rework: OpenAI `response_format: json_schema, strict:true`; int-encode large enums before strict (honor enum-count/length limits); dual-surface NL list + `[1..N]` enum; server-side belt retained | V44,V45 |
| T7 | x | calibration via OpenAI logprobs: discriminator Yes/No on plain completion; `Conf=exp(L_yes)/(exp(L_yes)+exp(L_no))`; `<0.5`→`manual_review` | V69,V-T3 |
| T8 | x | V41 worker partial-failure: per-item catch `openai.APIError`/`RateLimitError`/`InternalServerError`, log WARN, break early, return `partial_failure=True`+counts; scheduler reaches `commit()`; `task_run.status='succeeded'` | V41 |
| T9 | x | reseed AAMC as uploaded schema: `seeds/aamc_outline.schema.json` + validate-then-materialize importer (`POST /courses/{id}/outline:import`); re-upload restores MCAT | V-O2,V-O3,I.outline-import |
| T10 | . | V-L2 GATE: measurement harness re-runs categorizer + anki-topic-resolver eval on chosen OpenAI model; record jaccard/set-equality vs PoC Claude baseline; regression blocks pivot | V-L2,V44 |
| T11 | . | fix `README.md` `backend/` subdir reference (repo has no `backend/`) | §O |
| T12 | x | port categorizer + outline resolution → `node_id`: `app/services/categorizer/outline_lookup.py` (resolve node by ` >> ` path, ⊥ section/cc/topic codes) + categorizer job + `app/api/v1/recommendations.py` | V-O1,V-T1,V-T2 |
| T13 | x | port anki layer → `node_id`: `app/services/anki/{topic_resolver_worker,topic_resolver_batch,queries}.py` + assignment/review scope → node_id subtree rollup | V-O1,V-T1 |
| T15 | x | rename DB `mcat_coach`→`gradient` (+ `mcat_coach_test`→`gradient_test`): docker-compose.yml, .env/.env.example, conftest + schema-test DSNs/db_names; role `mcat` unchanged; stand up fresh `gradient` via `alembic upgrade head` | §C,I.env |
| T14 | x | port dashboard + read-services → `node_id`: `app/web/dashboard/services/*` (mastery, drilldown, sessions, anki_scope) + routes/questions + utils, `app/services/{analytics,recommender}.py`, `app/services/analyzer/*`, `app/services/tutor/*`; shared subtree-rollup helper (V-O1 set rollup) | V-O1,V-T1,V-E2 |
| T16 | . | (P1, after T14) redesign whole dashboard via `frontend-design` plugin → React+Tailwind SPA over existing `/api/v1/*` JSON (mastery, node drilldown, sessions, anki-scope, outline-import view); replaces Jinja `app/web/dashboard/`; backend stays FastAPI serving JSON; invoke `frontend-design` skill per view (feed JSON contract + current Jinja view as ref); node rollup = subtree set (V-O1) | §C,I.api,V-O1,V-D1 |

## §B — bug log

| id | date | cause | fix |
|-----|------|-------|-----|

