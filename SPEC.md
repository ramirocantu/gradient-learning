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

**Primary loop = PKM (rescoped 2026-05-26).** Question review → discriminator factors → grounded atomic facts → Notion write-back is the load-bearing workflow. Study-plan / recommender surfaces (`app/services/recommender.py`, plan/calendar views) are non-critical; candidate-for-cut unless they directly serve the PKM loop. Raw mastery + Anki + tutor/QBank facts survive (they feed the loop).

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
- JSON API (`/api/v1/*`) + MCP tutor seam = the sole client contract. **Repo ships no view layer (backend-only, 2026-05-26).** Clients — native macOS app, Chrome extension, MCP host — are downstream external consumers built against the HTTP contract in a separate phase/repo (§O). `/media/*` serves assets. ⊥ in-repo UI; the API contract is the boundary. (Jinja dashboard + viewer sub-apps DELETED; `app/main.py` mounts no sub-app.)

**PLUGINS (periphery, registry-keyed):**
- **Source adapters** — `capture → normalized {Question, Attempt}`, keyed by `source`. UWorld = reference adapter. Also: generic web-Qbank (extension), manual entry, PDF question-set parser.
- **Outline-schema importers** — validate + materialize an uploaded schema into `courses` + `outline_nodes`. AAMC outline = a bundled example schema file.
- **Anki tag-shape parsers** — `anki tag string → node ref`, keyed per deck/pack. AnKing-MCAT regex = reference parser.
- **Domain pack** = bundle of (outline schema + anki tag shape + question source) for one course.

Seam = the normalized internal model + adapter registries keyed on `source` / `course` / deck.

## §C — constraints

- Single user. Local-first **except Notion write-out** (sole cloud egress for storage) + **OpenAI API** (LLM + embeddings). Backend + Postgres stay local. ⊥ multi-user.
- Stack: Py 3.12 / FastAPI / SQLAlchemy async / Postgres 16 (asyncpg) / **OpenAI SDK** / APScheduler. **No in-repo UI (backend-only).** Extension TS + MV3 (separate repo). Adds: pgvector (`vector` ext via SQLAlchemy), `notion-client`, PyMuPDF/pdfplumber.
- ⊥ new **backend** framework, ORM, queue. ⊥ a new ORM for pgvector — use existing SQLAlchemy.
- **Backend-only (amended 2026-05-26):** this repo has no in-repo UI — Jinja dashboard + viewer sub-apps deleted, `jinja2`/`markdown` deps dropped, `app/main.py` mounts no sub-app (`/api/v1/*` + `/healthz` + `/media/*` only). Clients consume `/api/v1/*` (+ MCP tutor seam) over HTTP. ⊥ re-adding a server-rendered or bundled view layer to the backend. Native client = separate downstream phase/repo (§O). Supersedes the retired Jinja→SPA carve-out.
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
- **Residual P0 tech debt (rebaseline 2026-05-26):** `app/services/anki/{queries,state,retention}.py` still reference `topic_id` / `cc_code` / `topics` / `content_categories`; `app/services/{analytics,recommender,analyzer,tutor/outline}.py` + `app/web/dashboard/services/{mastery,drilldown,anki_scope}.py` self-document as stubs / partial ports; `app/startup.py` still calls `scripts/seed_outline.py` which is a no-op stub. Treat as blocking debt — P0.5 gate clears it (V-RB1..V-RB4).
- **KB substrate gap:** §I schema specifies `pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages` but no SQLAlchemy models, Alembic migrations, or `pyproject.toml` deps (pgvector / notion-client / PyMuPDF/pdfplumber) exist yet. P2 lands the substrate before P3 workflow (V-KB1, V-KB2).

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
- V-D1 (backend-only, reworked 2026-05-26): repo ships **no view layer**. Public JSON API (`/api/v1/*`) + MCP tutor seam = the sole client contract. ⊥ HTML / `TemplateResponse` endpoints; ⊥ mounted view sub-apps; ⊥ dashboard-only / private routes — a client needing data extends the public API. All clients external (native macOS app, extension, MCP host). The API contract is the boundary (carries §A). Structural guard: `tests/test_backend_only_seam.py`. (Prior Jinja-thin-client / SPA-swap wording retired.)

**Outline / node-based reads (extend)**
- V-O5: core read paths key on `node_id` / `outline_nodes` + subtree rollup via `app/services/outline_subtree.py`. ⊥ `topic_id` / `cc_code` / legacy `topics` / `content_categories` joins in `app/services/{categorizer,tutor,analytics,recommender,analyzer,anki}/...` or `app/web/dashboard/services/*`. Surfaces still on legacy joins = explicitly fenced off critical path or removed.
- V-O6: outline import (`POST /api/v1/courses/{id}/outline:import`) = sole canonical onboarding. `scripts/seed_outline.py` is no-op stub; `app/startup.py` ⊥ call it. Seed restoration = re-upload `seeds/aamc_outline.schema.json` via importer.

**Rebaseline (P0.5 gate)**
- V-RB1: no service in `app/services/{analytics,recommender,analyzer,tutor/outline}.py` or `app/web/dashboard/services/{mastery,drilldown,anki_scope}.py` self-documents as stub / partial port. Either ported to OutlineNode + subtree rollup, or explicitly fenced (commented + route-disabled + test-skipped) per rescope.
- V-RB2: `app/services/anki/{queries,state,retention}.py` contain zero references to `topic_id`, `cc_code`, `topics`, `content_categories` — or surface fenced out of critical path.
- V-RB3: `app/startup.py` ⊥ call `scripts/seed_outline.py`; startup behavior reconciled with `app/api/v1/outline.py` import flow (no implicit seed; explicit upload required).
- V-RB4: legacy tests referencing `Topic` / `ContentCategory` / `cc_code` rewritten to OutlineNode + subtree rollup, or removed if testing pruned surfaces.
- V-RB5: post-pivot (T4), ⊥ `from anthropic` / `import anthropic` anywhere in `app/` or `scripts/`. OpenAI is the single LLM provider (§C, V16). Legacy harness/eval scripts in `scripts/` (e.g. `eval_categorizer_models.py`, `compare_categorizer_v5_baseline.py`, `extract_features*.py`, `compare_topic_resolver_v3_v4.py`, `run_categorizer.py`, `run_anki_topic_resolver_batch.py`) must be ported to the OpenAI SDK or deleted; their associated tests follow. Recurrence trap: `uv sync` cleans a transient `anthropic` install → suite collection-errors at the next `from anthropic` import. See B2.

**Knowledge-base substrate (P2 gate)**
- V-KB1: P2 substrate (`pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages`) ships with SQLAlchemy models + Alembic migrations + idempotent re-run tests **before** any P3 retrieval / grounded-generation work lands. New service seams (PDF ingest/parse, embedding write, similarity-edge derivation, Notion write-out) live under `app/services/` with mocked-SDK contract tests.
- V-KB2: `pyproject.toml` gains `pgvector`, `notion-client`, PyMuPDF/pdfplumber before P3. Config plumbing (env vars from §I) wired and validated at startup.

**Retrieval (LLM4Tag Phase 1)**
- V-L3: tagging prompts for atomic facts / questions are constrained by retrieved outline-node candidates (embeddings + `concept_edges.kind='similarity'` + optional few-shot exemplars from prior calibrated tags). ⊥ raw free-form judgment over the full outline. Recall layer feeds candidates; calibrator (V69) scores them.

**MCP write-back**
- V-M3: discriminator writes via tutor/MCP seam append-only. ⊥ duplicate prior notes (dedupe by `(question_id, factor_text)` hash); question ↔ factor links preserved across re-writes. Notion mirror update (V-N1, V-N2) idempotent — block append, ⊥ page rewrite.

**Alembic migrations**
- V-MIG1: any alembic migration that introduces a named ENUM TYPE (via `sa.Enum(..., name=...)` or `postgresql.ENUM(..., name=...)` inside `op.create_table`) MUST issue `op.execute("DROP TYPE IF EXISTS <name>")` in the downgrade after the matching `op.drop_table()`. Postgres dialect auto-creates the TYPE on first CREATE TABLE; `DROP TABLE` leaves it orphaned. Subsequent `upgrade head` after `downgrade base` then hits `DuplicateObjectError: type "<name>" already exists`. Safer alternative: declare with `postgresql.ENUM(..., create_type=False)` and manage TYPE DDL explicitly via `op.execute("CREATE TYPE ...")` / `op.execute("DROP TYPE ...")` symmetrically in `upgrade()` / `downgrade()`. See B1.

## §P — phases

Re-sequenced 2026-05-26 per rescope: insert P0.5 stabilization gate; substrate (P2) lands before workflow automation (P3). SPA path CUT (backend-only pivot 2026-05-26) — see V-D1, §O.

- **P0 — schema generalize + OpenAI pivot (≈wk1).** [DONE] Collapse `Section/FC/CC/Topic` → `Course` + `outline_nodes`; retarget tags to `node_id`; open `source` enum on captures; swap `anthropic`→`openai` SDK across extractors; retire V38, rework V45, amend V41/V16/V69. Reseed AAMC as an uploaded schema. **Gate: V-L2 measurement harness green** (tagging quality vs Claude baseline). No new UX; unblocks all.
- **P0.5 — rebaseline (gate before P1).** Finish `node_id` port for residual read paths (analytics / recommender / analyzer / tutor-outline / dashboard mastery+drilldown+anki_scope) or explicitly fence cut surfaces per rescope (T17). Port `app/services/anki/{queries,state,retention}.py` off `topic_id`/`cc_code` (T18). Reconcile `app/startup.py` + `scripts/seed_outline.py` with outline-import flow — remove stale seed call (T19). Rewrite legacy Topic/CC tests (T20). **Gate: V-RB1..V-RB4 green.** Treats post-P0 stubs as blocking debt, ⊥ hidden detail.
- **P1 — usable course onboarding + node-based reads (≈wk2–3).** Complete `/api/v1/courses/*` + outline import route/service/test coverage (T21). Extend `app/api/v1/tutor.py` + backing services for node search + outline_subtree traversal without AAMC-only tree shape (T22). Normalize dashboard + anki consumers to public `/api/v1/*` contracts (T23). Anki sync/linking on real course (reuse). (View layer since DELETED — backend-only, 2026-05-26.)
- **P2 — knowledge-base substrate.** Add SQLAlchemy models + Alembic for `pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages` (T24). Add pgvector + notion-client + PyMuPDF/pdfplumber to `pyproject.toml` + config plumbing (T25). Service seams under `app/services/` for PDF ingest/parse, embedding write+versioning, similarity-edge derivation, Notion write-out (T26). Migration + idempotent-re-run + dim-change contract tests (T27). **Substrate lands before workflow automation.**
- **P3 — LLM4Tag retrieval + grounded generation.** Recall layer: candidate retrieval from embeddings + `concept_edges` similarity edges + optional few-shot exemplars (T28). Grounded generation + calibrated tagging over PDFs / atomic facts via existing OpenAI patterns + `app/services/llm/calibrator.py` (T29). Persist calibrated outputs to atomic-fact/tag tables with version + `manual_review` (T30). Domain-blind workflow — MCAT/AAMC = domain pack example, not privileged branch.
- **P4 — QBank synthesis + MCP/Notion write-back.** Discriminator-factor persistence via tutor/MCP seam, append-only, link-preserving (T31). Notion page/block append+update as one-way replica over `notion_pages` pointer, backlinks to question/node (T32). Expand source adapters (manual entry / web-Qbank / PDF question-set) under `app/services/adapters/` after write-back stable (T33). **T34 (done): SPA path PRUNED — backend-only.** Native client = separate downstream phase/repo (§O); ⊥ in-repo UI.

## §O — open items
- Embeddings provider: OpenAI `text-embedding-3-small` (dim 1536) vs BGE-local (dim 768). Default set to OpenAI for single-provider consistency; confirm.
- OpenAI model choices (`OPENAI_MODEL`, `OPENAI_CALIBRATOR_MODEL`) decided empirically in the P0 spike, not pinned here.
- Old MCAT/UWorld attempt *data*: assumed not migrated (fresh start). UWorld capture adapter + AnKing parser kept as example code plugins.
- `README.md` references a `backend/` subdir the repo doesn't have — fix during P0.
- **Native client (backend-only pivot 2026-05-26):** a native macOS app (and any other frontend) is a separate downstream phase/repo, built against the documented seam — `docs/BACKEND_CORE.md` (curated catalog) + `docs/openapi.json` (machine-readable contract). Out of scope for this backend repo. Supersedes the retired T16/T34 SPA path.

## §T — tasks

P0 — schema generalize + OpenAI pivot. Ids are monotonic, not positional: T12–T14 (dependent-module ports) are appended but run mid-phase. **P0 exec order (dependency-correct, I hand-drive — ⊥ `--next` id-order):** T1 → T2 → T15 → T3 → T12 → T13 → T14 → T4 → T5 → T6 → T7 → T8 → T9 → T10(gate) → T11. (T15 = DB rename, independent housekeeping, runs now.) Schema/tags + ports land before the OpenAI pivot so the suite compiles; gate last. **T16** (dashboard SPA redesign) PRUNED by T34 — backend-only pivot 2026-05-26; no view layer in repo (V-D1, §O).

**P0.5+ exec order (rescope 2026-05-26):** P0.5: T19 → T17 → T18 → T20 → T35 (seed/startup cleanup → service ports → anki ports → legacy-schema test prune → OpenAI-SDK test reshape). P1: T21 → T22 → T23. P2: T24 → T25 → T26 → T27 (models+migrations → deps → service seams → tests). P3: T28 → T29 → T30 (recall → grounded gen → persist). P4: T31 → T32 → T33 → T34 (discriminator persist → Notion write-back → adapter expansion → SPA prune, backend-only); P4 follow-up: T37 (PDF-qset adapter — the "hardest, last" source deferred from T33). See FORMAT.md for `st` legend.

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
| T10 | x | V-L2 GATE: measurement harness re-runs categorizer + anki-topic-resolver eval on chosen OpenAI model; record jaccard/set-equality vs PoC Claude baseline; regression blocks pivot | V-L2,V44 |
| T11 | x | fix `README.md` `backend/` subdir reference (repo has no `backend/`) | §O |
| T12 | x | port categorizer + outline resolution → `node_id`: `app/services/categorizer/outline_lookup.py` (resolve node by ` >> ` path, ⊥ section/cc/topic codes) + categorizer job + `app/api/v1/recommendations.py` | V-O1,V-T1,V-T2 |
| T13 | x | port anki layer → `node_id`: `app/services/anki/{topic_resolver_worker,topic_resolver_batch,queries}.py` + assignment/review scope → node_id subtree rollup | V-O1,V-T1 |
| T15 | x | rename DB `mcat_coach`→`gradient` (+ `mcat_coach_test`→`gradient_test`): docker-compose.yml, .env/.env.example, conftest + schema-test DSNs/db_names; role `mcat` unchanged; stand up fresh `gradient` via `alembic upgrade head` | §C,I.env |
| T14 | x | port dashboard + read-services → `node_id`: `app/web/dashboard/services/*` (mastery, drilldown, sessions, anki_scope) + routes/questions + utils, `app/services/{analytics,recommender}.py`, `app/services/analyzer/*`, `app/services/tutor/*`; shared subtree-rollup helper (V-O1 set rollup) | V-O1,V-T1,V-E2 |
| T16 | x | PRUNED (backend-only pivot 2026-05-26, T34 decision): SPA dashboard redesign abandoned; Jinja `app/web/dashboard/` + viewer sub-apps DELETED, no view layer in repo. Native macOS client supersedes — separate downstream phase/repo (§O) | §C,V-D1 |
| T17 | x | (P0.5) port `app/services/{analytics,recommender,analyzer,tutor/outline}.py` + `app/web/dashboard/services/{mastery,drilldown,anki_scope}.py` off stubs onto OutlineNode + `outline_subtree`, or explicitly fence off critical path per rescope | V-RB1,V-O5,V-O1 |
| T18 | x | (P0.5) port `app/services/anki/{queries,state,retention}.py` off `topic_id` / `cc_code` / legacy `topics`/`content_categories` joins onto OutlineNode + `outline_subtree`; or fence off critical path | V-RB2,V-O5,V-O1 |
| T19 | x | (P0.5) reconcile `app/startup.py` + `scripts/seed_outline.py` with `POST /api/v1/courses/{id}/outline:import` flow; remove stale seed call from startup; seed restoration = explicit re-upload of `seeds/aamc_outline.schema.json` | V-RB3,V-O6 |
| T20 | x | (P0.5) rewrite or remove legacy tests referencing `Topic` / `ContentCategory` / `cc_code`; suite reflects generalized OutlineNode schema | V-RB4 |
| T21 | x | (P1) complete `/api/v1/courses/*` + outline import route/service/test coverage: create, re-upload (idempotent), validation failure (atomic reject), node-tree reads | V-O2,V-O3,I.outline-import |
| T22 | x | (P1) extend `app/api/v1/tutor.py` + backing services for node search + outline_subtree traversal without AAMC-only tree shape; MCP/tutor flows speak `node_id` only | V-O1,V-O3,V-D1,V-M1 |
| T23 | x | (P1) normalize dashboard + anki consumers to public `/api/v1/*` contracts only; ⊥ dashboard-only backend seam; Jinja stays as thin client until T34 | V-D1 |
| T24 | x | (P2) add SQLAlchemy models + Alembic migrations for `pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages` (+ `discriminator_factors` if not yet modeled); register in `app/models/__init__.py` | V-KB1,I.schema |
| T25 | x | (P2) add `pgvector`, `notion-client`, `PyMuPDF`/`pdfplumber` to `pyproject.toml`; wire config plumbing for `PDF_INBOX_DIR`, `NOTION_API_TOKEN`, `NOTION_WIKI_DB_ID`, `EMBEDDING_MODEL`; startup validation | V-KB2,§C,I.env |
| T26 | x | (P2) add service seams under `app/services/` for PDF ingest/parse, embedding write+versioning, similarity-edge derivation, Notion write-out; mocked-SDK contract tests per `tests/_openai_mocks.py` pattern; Notion + embedding clients mocked at SDK boundary | V-KB1,V-E1,V-E2,V-N1,V-N2,V16 |
| T27 | x | (P2) migration + contract tests for new substrate: idempotent re-run; dim change → version bump + re-embed; Notion write idempotent (one-way, append-only) | V-KB1,V-E1,V-N1,V-N2 |
| T28 | x | (P3) recall layer: candidate retrieval from `content_embeddings` + `concept_edges.kind='similarity'` + optional few-shot exemplars from prior calibrated tags; feeds tagging prompts | V-L3,V-E2 |
| T29 | x | (P3) grounded generation + calibrated tagging over PDFs / atomic facts via existing OpenAI patterns + `app/services/llm/calibrator.py`; constrained by retrieved candidates (⊥ free-form full-outline judgment) | V-L3,V69,V45,V44 |
| T30 | x | (P3) persist calibrated outputs to `atomic_facts` + `<target>_tags` tables with `embedding_version` / `extractor_version` stamps + `manual_review` (Conf<0.5) | V-T2,V-T3,V-E1 |
| T31 | x | (P4) discriminator-factor persistence via tutor/MCP seam: `write_discriminator_factor` + `POST /api/v1/pkm/discriminators`; append-only dedupe by `(question_id, factor_text)` hash; question↔factor links preserved | V-M1,V-M3 |
| T32 | x | (P4) Notion page/block append+update as one-way replica over `notion_pages` pointer; backlinks question/node anchors; idempotent re-sync; ⊥ read-back | V-N1,V-N2,V-M3 |
| T33 | x | (P4) expand source adapters under `app/services/adapters/`: manual entry → web-Qbank (extension) → PDF question-set parser (hardest, last); only after write-back stable | I.captures,§A |
| T34 | x | (P4) DECISION (2026-05-26): backend-only — prune T16 SPA; repo ships JSON API + MCP seam only. View layer DELETED; native client = downstream phase/repo (§O) | V-D1,§A |
| T35 | x | (P0.5) rewrite per-extractor tests (`test_categorizer_llm`, `test_anki_topic_resolver`, `test_feature_extractor`, `test_synthesizer`, `test_analyzer_endpoint`, `test_scheduler`, `tests/web/dashboard/test_insights`) onto OpenAI SDK boundary via `tests/_openai_mocks.py`; drop V38 `cache_control` asserts; `ToolUseBlock` isinstance → `response_format` json_schema content reads; mark `tests/test_llm_batch.py` skip-all (batch retired in T4); T4 follow-up the smoke test stood in for | V16,V38,V45 |
| T36 | x | (P0.5 follow-up) port-or-delete the 7 stale anthropic harness scripts in `scripts/` (`eval_categorizer_models.py`, `compare_categorizer_v5_baseline.py`, `extract_features.py`, `extract_features_sample.py`, `compare_topic_resolver_v3_v4.py`, `run_categorizer.py`, `run_anki_topic_resolver_batch.py`); drop `tests/test_eval_script.py` + `tests/conftest.py` `collect_ignore_glob` workaround once the importer is gone | V-RB5,V16,§C |
| T37 | . | (P4 follow-up) PDF question-set source adapter under `app/services/adapters/` (`source='pdf-qset'`), deferred from T33 as "hardest, last": file-upload payload (⊥ extension `CapturePayload`); pdfplumber/PyMuPDF parse → MANY `Question` rows, ⊥ `Attempt` (bulk import, no recorded answer); register in adapter registry; reuse §A seam proven by T33 | I.captures,§A |
| T38 | x | (P4 follow-up) surface node tags on tutor reads: resolve `QuestionTag.node_id` (+ path via `OutlineLookup`) per question in `tutor/captures/recent` + `tutor/sessions/{id}/summary` (`topics`/`by_topic` are empty TODO stubs — `app/services/tutor/captures.py:30`, `sessions.py:85`). Return `node_id`(+path) in payload. Unblocks client node-labels on captures + session per-node breakdown (desktop `desktop/SPEC.md` ¶T1/¶T2) | V-O5,V-O1,V-D1,V-T1 |
| T39 | . | (P4 follow-up) system-status read endpoint `GET /api/v1/admin/status` (or `/healthz` extension): probe AnkiConnect reachability (`version`/`deckNames`), Notion token validity, OpenAI key reachability; fold in per-job last `TaskRun` status (`/admin/jobs` returns only `next_run_time`, no last-run outcome). Lets a client show real connection health ⊥ "scheduled"/"unknown" guesses. Unblocks desktop `desktop/SPEC.md` ¶T3 settings panel | V-D1,I.api,§A |

## §B — bug log

| id | date | cause | fix |
|-----|------|-------|-----|
| B1 | 2026-05-26 | alembic 0001_initial.py declared `task_runs.status` as `sa.Enum('running','succeeded','failed', name='task_run_status')` inside `op.create_table`. Postgres dialect auto-created the named TYPE on first CREATE TABLE, but the downgrade only called `drop_table('task_runs')` — the TYPE leaked. Re-running `upgrade head` after `downgrade base` then hit `DuplicateObjectError: type "task_run_status" already exists`. Surfaced by T27's full-roundtrip migration test; also explained the pre-existing failure of `test_question_features_schema::test_migration_apply_and_rollback`. | Added `op.execute("DROP TYPE IF EXISTS task_run_status")` in 0001_initial.py downgrade after `drop_table('task_runs')`. Recurrence guarded by V-MIG1. |
| B2 | 2026-05-26 | T4 swapped `anthropic`→`openai` across `app/services/llm/` but left 7 harness/eval scripts in `scripts/` (`eval_categorizer_models.py`, `compare_categorizer_v5_baseline.py`, `extract_features.py`, `extract_features_sample.py`, `compare_topic_resolver_v3_v4.py`, `run_categorizer.py`, `run_anki_topic_resolver_batch.py`) with `from anthropic import AsyncAnthropic` at module top-level — they survived as residual P0 debt because `anthropic` lingered in the venv from pre-pivot installs. T26's `uv sync` (triggered by T25 deps) trimmed that transient install; `tests/test_eval_script.py` then collection-errored with `ModuleNotFoundError: No module named 'anthropic'`, aborting the whole suite before any test ran. | Short-term: T26 added `collect_ignore_glob = ["test_eval_script.py"]` in `tests/conftest.py` to restore collection. Long-term: T36 ports or deletes the 7 scripts + their test + removes the collect_ignore_glob workaround. Recurrence guarded by V-RB5. |

