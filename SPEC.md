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

**Primary loop = PKM (rescoped 2026-05-26).** Question review → discriminator factors → grounded atomic facts → Notion write-back is the load-bearing workflow. Study-plan / recommender surfaces (`app/services/recommender.py`, plan/calendar views) are non-critical; candidate-for-cut unless they directly serve the PKM loop. Raw mastery + Anki + tutor/QBank facts survive (they feed the loop). **Categorization redesign (2026-05-27):** transition MCAT-only PoC → general coursework tool — legacy AAMC categorizer (`categorizer/{llm,worker}`, `SUBJECT_TO_SECTION`) + anki topic-resolver CUT; the LLM4Tag grounded path is the categorization engine (V-L4; T53 removes the old, T50 wires the new).

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
- Embeddings default = OpenAI `text-embedding-3-small` (pgvector dim 1536); BGE-local (`bge-base-en-v1.5`, dim 768) retained as a config swap. `embedding_version` stamps every row; provider/dim change ⇒ bump + re-embed. **(Confirmed 2026-05-28, V-L5: OpenAI `text-embedding-3-small`; BGE-local stays a swap.)**
- Outline creation = **upload a schema**, not in-app PDF parsing. Gradient ships a prompt template; the user runs it against their own sources (PDF/screenshot/webpage) in their own LLM session, then uploads the resulting schema. Gradient owns validate + materialize only.
- Anki source = AnkiConnect HTTP. Read calls only by default; write allowlist (`unsuspend`, `addTags`, namespaced `createFilteredDeck`) carried from the PoC, ⊥ scheduler mutation. Per-deck tag-shape parser is a plugin.
- PDF corpus = user-uploaded classroom notes/slidedecks (incl. handwritten). Parse = render each page → image → OpenAI **vision** transcription (⊥ rely on embedded extractable text — handwriting / image slides have none). Atomic facts = OpenAI structured-output extraction grounded to the page transcription (⊥ regex sentence-split). **PyMuPDF** renders pages; pdfplumber retained for digital-text PDFs. local-dir poller. (Redesign 2026-05-28 — V-KB3/V-KB4, T54.)
- Notion = **write-out only**. One page per concept (outline node); atomic facts = blocks within. Store a pointer index (`notion_page_id`, `url`, `tags[]`, `node_id`) + back-link anchors. ⊥ read-back, ⊥ local content copy, ⊥ Notion-as-source-of-truth.
- Cognitive-safety (hard rule): AI tags / summarizes / links / drafts; ⊥ generate primary active-recall questions or flashcards.
- MCP role: data exposure + structured writes; LLM (host) = reasoner. ⊥ heuristics in tool signatures. Socratic dialogue host-side; discriminator tool persists only.
- `Attempt.time_seconds` ⊥ actionable (carried hard constraint).
- **Residual P0 tech debt (rebaseline 2026-05-26):** `app/services/anki/{queries,state,retention}.py` still reference `topic_id` / `cc_code` / `topics` / `content_categories`; `app/services/{analytics,recommender,analyzer,tutor/outline}.py` + `app/web/dashboard/services/{mastery,drilldown,anki_scope}.py` self-document as stubs / partial ports. (`app/startup.py` + `scripts/seed_outline.py` since DELETED 2026-05-27 — the implicit-seed-call debt is gone; see V-O6/V-RB3.) Treat as blocking debt — P0.5 gate clears it (V-RB1..V-RB4). **Categorizer redesign (2026-05-27):** the legacy MCAT categorizer + anki topic_resolver are no longer debt-to-port — CUT/deleted (V-L4, T53); fenced `recommender`/`analyzer` deleted as orphans too; shared `OutlineLookup` relocated out of `categorizer/`.
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
  id, source TEXT, external_id TEXT,   # (source, external_id) UQ = TARGET; code today keys on `qid` (globally-UQ TEXT), external_id rename deferred (app/models/captures.py:97)
  stem_html, stem_plain, choices JSONB, correct_choice?,   # NULLABLE (V-CAP1, T55) — NULL = answer pending for a deferred-answer capture source; current adapters always supply it
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
# Shipped schema format (JSON), uploaded by user:
{ "course": {"slug","name","description?"},
  "nodes": [ {"path": ["Section","FC","CC","Topic"], "kind", "name",
              "disciplines?": [...], "position?"} , ... ] }
api: POST /api/v1/courses                         → Course
api: POST /api/v1/courses/{id}/outline:import     # body = schema; validate → materialize nodes
api: GET  /api/v1/courses/{id}/outline            → node tree
job: (sync) outline_import — validate, dedupe, build parent chain
docs: docs/PROMPT_OUTLINE_SCHEMA.md — the template user runs against their own sources (NOT YET WRITTEN — T52)
seed: app/seeds/aamc_outline.schema.json — MCAT reference; uploading it restores MCAT outline
```

### Ingress (notes → atomic facts → Notion)

```
env: PDF_INBOX_DIR
api: POST /api/v1/pdf/ingest {course_id, file}    # or scheduler-polled inbox (UNBUILT — T51)
job: run_pdf_ingest_job   # poll PDF_INBOX_DIR → render pages (PyMuPDF) → vision transcribe (OpenAI) → LLM extract atomic facts (structured) → embed → grounded-tag (categorizer) (kb/pdf_ingest.py REWRITTEN by T54; job UNREGISTERED — T51)
job: run_notion_sync_job  # one-way: per node, upsert a Notion page; facts = blocks; back-links + pointer row (service kb/notion.py built; job UNREGISTERED — T51)
mcp: write_discriminator_factor(question_id, factor_text, node_id?)
api: POST /api/v1/pkm/discriminators → DiscriminatorFactor   # persist-only (X-Coach-Token)
```

### Practice questions + mastery

```
api: POST /api/v1/captures            # source-tagged; routes to source adapter (manual entry = source='manual', app/services/adapters/manual.py — ⊥ standalone /questions route)
api: GET  /api/v1/outline/courses/{id}/mastery → CourseSummary   # (T44; was /mastery/course/{id})
api: GET  /api/v1/outline/nodes/{id}/mastery   → NodeSummary     # subtree-membership rollup (T44)
job: run_categorizer_job   # REMOVED — legacy MCAT categorizer CUT (V-L4, T53); run_grounded_tag_job (T50) replaces it
job: run_grounded_tag_job  # LLM4Tag: recall→grounded→persist over needs_categorization Qs + untagged atomic_facts (seams kb/recall+llm/grounded+kb/persist_tags; REGISTERED T50 (app/scheduler.py) via kb/jobs.tag_pending)
job: run_embed_job         # embed new questions/facts/nodes → content_embeddings (REGISTERED T50 (app/scheduler.py) via kb/jobs.embed_pending)
job: run_calibrate_job     # calibration runs INLINE in run_grounded_tag_job via generate_grounded_tags→calibrate_tag, V69; DB CHECK enforces conf<0.5⇒manual_review; standalone re-grade sweep DEFERRED by T50 decision 2026-05-28 — only needed on calibrator-model rotation
ext: source adapters — uworld (reference) | web-qbank | manual   # pdf-qset CUT (T37, 2026-05-28) — notes-only PDF ingress, ⊥ practice-PDF import
```

### Anki (reuse)

```
env: ANKICONNECT_URL=http://127.0.0.1:8765 ; ANKI_DECK_NAME ; ANKI_SYNC_INTERVAL_MINUTES
api: POST /api/v1/anki/sync ; GET /api/v1/anki/cards/by-qid/{qid} ; GET review-queue ; load-adherence   # /cards?node_id= FENCED (app/api/v1/anki.py:91)
mcp: sync_anki (live) ; get_anki_review_queue (live) ; get_anki_cards_for_node + get_anki_performance (FENCED — backing /cards?node_id= + /performance commented, anki.py:91/172)
job: run_anki_sync_job ; assignment/review jobs (carried)   # run_anki_topic_resolver REMOVED — MCAT AAMC card→topic LLM resolver CUT (V-L4, T53)
plugin: anki tag-shape parser registry — AnKing-MCAT = reference; maps tag → node_id
```

### Other live API surface (documented from code — were EXTRA at /check §I 2026-05-27)

```
admin:    GET  /api/v1/admin/status ; GET /api/v1/admin/jobs ; DELETE /api/v1/admin/tags/{id}    # T39 (POST /recategorize REMOVED — drove legacy categorizer, V-L4/T53)
notes:    GET/POST /api/v1/attempts/{id}/notes ; DELETE /api/v1/attempts/notes/{id}
kb-reads: GET /api/v1/concept-edges ; GET /api/v1/atomic-facts ;
          GET /api/v1/notion/pages ; GET /api/v1/pdf-sources                                 # T45–T48
tutor:    GET /api/v1/tutor/{questions/by-qid/{qid}, questions/by-attempt-id/{id},
          captures/recent, sessions/latest, sessions/recent, sessions/{id}/summary,
          attempts/flagged, outline/nodes/search, outline, outline/nodes/{id}/subtree}       # node_id-only data seam (§A); T22/T38/T42
```

### Env

```
DATABASE_URL                 # postgresql+asyncpg://…
OPENAI_API_KEY               # required
OPENAI_BASE_URL              # optional; OpenAI-compatible local server later
OPENAI_MODEL                 # tagging / facts / extraction — default gpt-5.4-nano (V-L5)
OPENAI_VISION_MODEL          # PDF page transcription — default gpt-5.4-mini (⊥ nano; V-L5, V-KB3)
OPENAI_CALIBRATOR_MODEL      # logprobs Yes/No grade — gpt-5.4-nano reasoning OFF (V69/V-L5)
EMBEDDING_MODEL              # default text-embedding-3-small (dim 1536) — confirmed V-L5
OPENAI_SERVICE_TIER          # default flex — async KB jobs, ≈50% off standard (V-L5)
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
- V-L5 (model selection, decided 2026-05-28): per-task model pinned by capability+cost; all KB jobs are async ⇒ default `service_tier='flex'` (≈50% off Standard; Batch where a job tolerates queueing). **`OPENAI_MODEL` = `gpt-5.4-nano`** for tagging / atomic-fact extraction / grounded-tag / ranking — beats the retired `gpt-5-mini` on text reasoning (SWE-Bench Pro 52.4 vs 45.7) at ~4× lower cost, and OpenAI recommends nano for exactly classification/extraction/ranking ⇒ no nano→mini escalation tier. **`OPENAI_VISION_MODEL` = `gpt-5.4-mini`**, ⊥ nano — nano vision is weaker than even `gpt-5-mini` (OSWorld 39.0; "not built for vision") and a bad page transcription poisons every downstream fact/tag/embedding (V-KB3). **`EMBEDDING_MODEL` = `text-embedding-3-small`** (dim 1536) confirmed — no 5.4-class embedding model. Embeddings stay **synchronous** (`embed_pending`); Batch-API embedding (50% off) was evaluated 2026-05-28 and DEFERRED (see §O) — pennies saved at single-user scale vs a polling subsystem + ≤24h recall-index latency. **`OPENAI_CALIBRATOR_MODEL` = `gpt-5.4-nano` with reasoning OFF** — 5.4 models run reasoning-disabled DO expose `logprobs`, satisfying V69's logprob + non-reasoning constraint; same model as `OPENAI_MODEL` but the reasoning-off flag is mandatory for the Yes/No grade. Mechanism (wired): `calibrator.grade_yes_no` sets `reasoning_effort='none'` on the chat.completions call — on GPT-5.x this both disables reasoning AND is *required* to get `logprobs` back at all (reasoning mode returns none). `reasoning_effort=None` omits the flag for a legacy `gpt-4o*`/`gpt-4.1*` calibrator. Prices (Standard /1M tok): mini $0.75/$4.50, nano $0.20/$1.25, both 400k ctx. Any model rotation re-triggers the V-L2 harness. (ID note: V-L4 is the categorizer-cut invariant below — distinct.)

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
- V-O6: outline import (`POST /api/v1/courses/{id}/outline:import`) = sole canonical onboarding. `scripts/seed_outline.py` + `app/startup.py` both DELETED (2026-05-27) — ⊥ implicit seed exists; startup lifespan lives in `app/main.py`. Seed restoration = re-upload `seeds/aamc_outline.schema.json` via importer.

**Rebaseline (P0.5 gate)**
- V-RB1: no service in `app/services/{analytics,recommender,analyzer,tutor/outline}.py` or `app/web/dashboard/services/{mastery,drilldown,anki_scope}.py` self-documents as stub / partial port. Either ported to OutlineNode + subtree rollup, or explicitly fenced (commented + route-disabled + test-skipped) per rescope.
- V-RB2 (widened B4 2026-05-27): `app/services/anki/{queries,state,retention,sync,tag_parser}.py` contain zero references to `topic_id`, `content_category_id`, `skill_number`, `cc_code`, legacy `topics`/`content_categories` — anki tag resolution targets `node_id` only (V-T1, note-as-unit V75). `OutlineLookup` exposes node_id resolution only (`node_id_by_path`/`node`/`path_of`) — ⊥ legacy `content_category_id`/`topic_id`/`skill_number` lookup methods. Surfaces still on legacy joins = fenced off critical path or removed. (The original list omitted `sync.py`+`tag_parser.py` — the gap that let B4 hide behind env noise.)
- V-RB3: neither `app/startup.py` nor `scripts/seed_outline.py` exists (both DELETED 2026-05-27) — invariant satisfied structurally, ⊥ implicit seed; explicit upload via `app/api/v1/outline.py` import flow required. Startup lifespan in `app/main.py`.
- V-RB4: legacy tests referencing `Topic` / `ContentCategory` / `cc_code` rewritten to OutlineNode + subtree rollup, or removed if testing pruned surfaces.
- V-RB5: post-pivot (T4), ⊥ `from anthropic` / `import anthropic` anywhere in `app/` or `scripts/`. OpenAI is the single LLM provider (§C, V16). Legacy harness/eval scripts in `scripts/` (e.g. `eval_categorizer_models.py`, `compare_categorizer_v5_baseline.py`, `extract_features*.py`, `compare_topic_resolver_v3_v4.py`, `run_categorizer.py`, `run_anki_topic_resolver_batch.py`) must be ported to the OpenAI SDK or deleted; their associated tests follow. Recurrence trap: `uv sync` cleans a transient `anthropic` install → suite collection-errors at the next `from anthropic` import. See B2.
- V-RB6 (2026-05-30, B6+B7): no LIVE write path — a mounted route, a scheduled job, or any service they transitively call — references the retired 3-target tag shape: `topic_id`/`content_category_id`/`skill`/`skill_number`/`cc_code`/`parsed_kind IN ('aamc_topic','aamc_cc')` or the legacy `topics`·`content_categories` tables — whether by constructing it, forwarding it as kwargs to a `node_id`-only service (`admin_tags.create_manual_tag(*, node_id, …)`), reading it off a node_id-only model (`QuestionTag`/`AnkiNoteTag`), or joining it in SQL. Tag writes target `node_id` only (V-T1). Behavior-scoped — supersedes V-RB2's per-file list, which is anki-only (`{queries,state,retention,sync,tag_parser}.py`) and missed `app/api/v1/admin.py` (`ManualTagBody`/`create_manual_tag`/`_tag_row_payload`, B6) + `app/services/anki/assignment.py` (`_CANDIDATE_SQL_TOPIC`/`_CANDIDATE_SQL_CC`, B7). Fenced + route-disabled surfaces exempt (V-RB2 escape-clause, T49). Guard: `tests/test_v_rb6_no_retired_3target_writes.py` (xfail-strict until T57).

**Knowledge-base substrate (P2 gate)**
- V-KB1: P2 substrate (`pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages`) ships with SQLAlchemy models + Alembic migrations + idempotent re-run tests **before** any P3 retrieval / grounded-generation work lands. New service seams (PDF ingest/parse, embedding write, similarity-edge derivation, Notion write-out) live under `app/services/` with mocked-SDK contract tests.
- V-KB2: `pyproject.toml` gains `pgvector`, `notion-client`, `pymupdf` (page render — V-KB3), `pdfplumber` (digital-text PDFs) before P3. Config plumbing (env vars from §I) wired and validated at startup.
- V-KB3 (notes-ingress redesign 2026-05-28): PDF ingress renders each page to an image and transcribes it via an OpenAI **vision** call; ⊥ assume embedded extractable text (handwriting / image-only slides yield none). Vision-transcription + fact-extraction OpenAI calls are injected and mocked at the SDK boundary in tests (V16); cost read from `usage` incl. `prompt_tokens_details.cached_tokens` (V-L1); each persisted fact stamped with `extractor_version`. Page render (PyMuPDF) is injectable so tests skip a real PDF/vision round-trip.
- V-KB4 (notes-ingress redesign 2026-05-28): atomic facts extracted via OpenAI structured output (`response_format: json_schema, strict:true` — V45) grounded to the page transcription; ⊥ regex sentence-split as the fact source. Dedup by `content_hash` per course (carried, `UQ(course_id, content_hash)`). `atomic_facts.node_id` stays NULL until the grounded-tag categorizer (V-L3/V69) runs — ingress persists facts, T50 categorizes.

**Captures (questions)**
- V-CAP1 (2026-05-28): `questions.correct_choice` is NULLABLE (shipped — T55, migration `0006_correct_choice_nullable`). A capture source where the correct answer isn't available at capture time — the motivating case is a **web-capture adapter** that scrapes a question before the answer is revealed — records `correct_choice=NULL` = answer pending (⊥ gradeable; surfaced via `WHERE correct_choice IS NULL`, until filled). Today's adapters (extension/uworld/manual) still always supply it on capture, and `ParsedCapture.correct_choice` stays a required `str` — only a deferred-answer source produces NULL. No question-level flag column yet — NULL is the signal; add an indexed flag when a deferred-answer source actually lands. Blast radius scoped: 4 non-test reads of `correct_choice` (model col, `ParsedCapture` required, extension writer, tutor passthrough emits null). (Originated in the cut T37 pdf-qset design; kept as the one reusable idea.)
- V-CAP2 (2026-05-28): every capture is **course-scoped at ingest**. `CapturePayload` carries `course_slug` (known field — `extra='forbid'` stays); the adapter resolves it against `courses.slug` and stamps `course_id` (FK→courses) on the persisted `RawCapture` + `Question`. The grounded-tag categorizer (`run_grounded_tag_job` / `kb/jobs.tag_pending`) scopes a question's recall to ITS `course_id` — retires the legacy "tag questions only when exactly one course exists" guard (T50) for course-stamped questions (uncourse-stamped legacy rows keep the single-course fallback). Unknown slug → 422 (⊥ silent drop / wrong-course tag). `course_slug` stays Optional only for back-compat with the pre-course extension + the single-course case; a multi-course install REQUIRES it (⊥ ambiguous course → categorizer can't scope). Mirrors extension SPEC ¶O-1/¶T5 — Gradient Capture sends the user-picked course; `GET /api/v1/courses` already feeds its dropdown.

**Retrieval (LLM4Tag Phase 1)**
- V-L3 (amended 2026-05-28, V-L6): tagging prompts for atomic facts / questions are constrained by retrieved outline-node candidates. Recall merges THREE LLM4Tag meta-paths, deduped by node: **C2T** (cosine vs outline-node vectors), **C2C2T** (similar already-tagged content → its tag, V-L6), **T2T** (`concept_edges.kind='similarity'` neighbours of the C2T hits), plus optional few-shot exemplars from prior calibrated tags. ⊥ raw free-form judgment over the full outline. Recall layer feeds candidates; calibrator (V69) scores them.
- V-L6 (recall-completeness fixes, 2026-05-28): the recall layer (`kb/recall.py`, `kb/jobs.py`) closes the gaps that silently capped tagging recall vs the LLM4Tag paper (the candidate set is the HR#k ceiling — a node never recalled can never be tagged). Five parts:
  - **C2C2T content recall (A):** `_content_candidates` borrows tags from embedding-similar already-tagged `atomic_facts` in the course — the paper's hard-case rescuer (Fig 6), absent before. Second hop is **gold/silver weighted**: `source∈{manual,schema_map}` → weight 1.0 (`via='content-gold'`); `source='llm' ∧ ¬manual_review ∧ confidence≥δ_silver` → `silver_factor` (`via='content-silver'`, damps the echo/feedback loop). Computed on-the-fly (no content↔content edge table yet). The calibrator (V69) remains the hard backstop — C2C2T only proposes.
  - **node-path embedding (B):** outline nodes embed their full `>>` path, not the bare leaf name — the cosine matches a fact's prose against the tag's *meaning*. Re-embed under `embedding_version` `-v2` (V-E1).
  - **exemplars ON in the live job (C):** `tag_pending` calls recall with `exemplars_per_node=3` — SRKI was inert (default 0) before.
  - **δ floor (D):** embedding + content candidates below `min_score` (provisional 0.25) are dropped, not handed up as least-bad picks; retune on the V-L2 harness against real `text-embedding-3-small` cosine scale.
  - **fan-out caps (E):** T2T ≤ `edge_top_n` (5), C2C2T ≤ `content_node_cap` (5) — mirror the paper's meta-path caps so the candidate list can't balloon and dilute the pick.
  - **Dependency flagged:** C2C2T's on-the-fly content↔content cosine scans all course content vectors per recall — Python cosine is fine at one-outline scale but is the first thing that forces the pgvector ANN swap (see V-E1 / similarity.py note). C2C2T is corpus-dependent: contributes ~nothing on a cold KB, strengthens as content tags up.
- V-L4 (redesign CUT, 2026-05-27): legacy MCAT categorizer REMOVED. ⊥ `app/services/categorizer/{llm,worker,cache,_text}.py`, ⊥ `SUBJECT_TO_SECTION` / `uworld_aamc_tags`-driven per-section tagging, ⊥ `app/services/anki/topic_resolver*.py`. The LLM4Tag grounded path (`kb/recall` + `llm/grounded` + `kb/persist_tags` + `llm/calibrator`, V-L3/V69) is the SOLE categorization/tagging seam, wired live by T50. `OutlineLookup` + `normalize_typographic` are domain-blind core infra — live OUTSIDE `categorizer/` (relocated `app/services/outline/lookup.py`), ⊥ deleted with the categorizer. Categorization may be non-functional until T50; gap is intentional (PoC→general-tool transition). See §G, T53.

**MCP write-back**
- V-M3: discriminator writes via tutor/MCP seam append-only. ⊥ duplicate prior notes (dedupe by `(question_id, factor_text)` hash); question ↔ factor links preserved across re-writes. Notion mirror update (V-N1, V-N2) idempotent — block append, ⊥ page rewrite.

**Alembic migrations**
- V-MIG1: any alembic migration that introduces a named ENUM TYPE (via `sa.Enum(..., name=...)` or `postgresql.ENUM(..., name=...)` inside `op.create_table`) MUST issue `op.execute("DROP TYPE IF EXISTS <name>")` in the downgrade after the matching `op.drop_table()`. Postgres dialect auto-creates the TYPE on first CREATE TABLE; `DROP TABLE` leaves it orphaned. Subsequent `upgrade head` after `downgrade base` then hits `DuplicateObjectError: type "<name>" already exists`. Safer alternative: declare with `postgresql.ENUM(..., create_type=False)` and manage TYPE DDL explicitly via `op.execute("CREATE TYPE ...")` / `op.execute("DROP TYPE ...")` symmetrically in `upgrade()` / `downgrade()`. See B1.
- V-MIG2: every alembic revision id (the `revision:` string, also the migration filename stem) MUST be ≤ 32 chars. `alembic_version.version_num` is `varchar(32)`; an over-length id truncation-errors at `upgrade head` on the `UPDATE alembic_version SET version_num=...` that stamps the new head (`asyncpg StringDataRightTruncationError: value too long for type character varying(32)`). Keep ids short + numbered (`000N_short_slug`). See B5.

**Test isolation / config**
- V-TC1 (stabilization gate, 2026-05-27): the test suite ⊥ read the real `.env` file or hit the network for config. Test config comes from fixtures only — `COACH_TOKEN` set to a known dummy, `NOTION_API_TOKEN`/`OPENAI_API_KEY` unset (probes report unconfigured, ⊥ live calls), DB URL pinned to the test DB. A clean checkout's `uv run pytest` is green ⊥ depending on a developer's `.env`. Mechanism caveat: `app.config.settings` is a module-level singleton built at first import (`app/config.py:10`, `env_file=_BACKEND_ROOT/.env`, `env_ignore_empty=True`), so an autouse fixture alone runs *after* the singleton exists — env must be set at `tests/conftest.py` import (before any `app.config` import) and/or the singleton rebuilt. Guards B3. (V16 already bars live OpenAI; V-TC1 extends the bar to Notion + auth + DB config.)

## §P — phases

Re-sequenced 2026-05-26 per rescope: insert P0.5 stabilization gate; substrate (P2) lands before workflow automation (P3). SPA path CUT (backend-only pivot 2026-05-26) — see V-D1, §O.

- **P0 — schema generalize + OpenAI pivot (≈wk1).** [DONE] Collapse `Section/FC/CC/Topic` → `Course` + `outline_nodes`; retarget tags to `node_id`; open `source` enum on captures; swap `anthropic`→`openai` SDK across extractors; retire V38, rework V45, amend V41/V16/V69. Reseed AAMC as an uploaded schema. **Gate: V-L2 measurement harness green** (tagging quality vs Claude baseline). No new UX; unblocks all.
- **P0.5 — rebaseline (gate before P1).** Finish `node_id` port for residual read paths (analytics / recommender / analyzer / tutor-outline / dashboard mastery+drilldown+anki_scope) or explicitly fence cut surfaces per rescope (T17). Port `app/services/anki/{queries,state,retention}.py` off `topic_id`/`cc_code` (T18). Reconcile `app/startup.py` + `scripts/seed_outline.py` with outline-import flow — remove stale seed call (T19). Rewrite legacy Topic/CC tests (T20). **Gate: V-RB1..V-RB4 green.** Treats post-P0 stubs as blocking debt, ⊥ hidden detail.
- **P1 — usable course onboarding + node-based reads (≈wk2–3).** Complete `/api/v1/courses/*` + outline import route/service/test coverage (T21). Extend `app/api/v1/tutor.py` + backing services for node search + outline_subtree traversal without AAMC-only tree shape (T22). Normalize dashboard + anki consumers to public `/api/v1/*` contracts (T23). Anki sync/linking on real course (reuse). (View layer since DELETED — backend-only, 2026-05-26.)
- **P2 — knowledge-base substrate.** Add SQLAlchemy models + Alembic for `pdf_sources`, `atomic_facts`, `content_embeddings`, `concept_edges`, `notion_pages` (T24). Add pgvector + notion-client + PyMuPDF/pdfplumber to `pyproject.toml` + config plumbing (T25). Service seams under `app/services/` for PDF ingest/parse, embedding write+versioning, similarity-edge derivation, Notion write-out (T26). Migration + idempotent-re-run + dim-change contract tests (T27). **Substrate lands before workflow automation.**
- **P3 — LLM4Tag retrieval + grounded generation.** Recall layer: candidate retrieval from embeddings + `concept_edges` similarity edges + optional few-shot exemplars (T28). Grounded generation + calibrated tagging over PDFs / atomic facts via existing OpenAI patterns + `app/services/llm/calibrator.py` (T29). Persist calibrated outputs to atomic-fact/tag tables with version + `manual_review` (T30). Domain-blind workflow — MCAT/AAMC = domain pack example, not privileged branch.
- **P4 — QBank synthesis + MCP/Notion write-back.** Discriminator-factor persistence via tutor/MCP seam, append-only, link-preserving (T31). Notion page/block append+update as one-way replica over `notion_pages` pointer, backlinks to question/node (T32). Expand source adapters (manual entry / web-Qbank / PDF question-set) under `app/services/adapters/` after write-back stable (T33). **T34 (done): SPA path PRUNED — backend-only.** Native client = separate downstream phase/repo (§O); ⊥ in-repo UI.

## §O — open items
- Embeddings provider: RESOLVED 2026-05-28 (V-L5) — OpenAI `text-embedding-3-small` (dim 1536) confirmed; BGE-local (dim 768) stays a config swap.
- OpenAI model choices: RESOLVED 2026-05-28 (V-L5) — `OPENAI_MODEL`=gpt-5.4-nano, `OPENAI_VISION_MODEL`=gpt-5.4-mini, `OPENAI_CALIBRATOR_MODEL`=gpt-5.4-nano (reasoning OFF → exposes logprobs, satisfies V69), async chat jobs on Flex (`OPENAI_SERVICE_TIER`). Re-measure per V-L2 on any rotation.
- Embeddings Batch-API (50% off): EVALUATED + DEFERRED 2026-05-28. Embeddings stay synchronous (`embed_pending`). The only confirmed embedding discount is the Batch API (`service_tier='flex'` on `/v1/embeddings` is undocumented); Batch is a two-phase async subsystem (state table + migration + submit/collect jobs + file upload/download + ≤24h poll) whose single-user ROI is pennies (text-embedding-3-small is the cheapest model) and whose ≤24h latency would stall the recall index after an outline import. Revisit only if embedding volume grows materially. (V-L5)
- Old MCAT/UWorld attempt *data*: assumed not migrated (fresh start). UWorld capture adapter + AnKing parser kept as example code plugins.
- `README.md` references a `backend/` subdir the repo doesn't have — fix during P0.
- **Native client (backend-only pivot 2026-05-26):** a native macOS app (and any other frontend) is a separate downstream phase/repo, built against the documented seam — `docs/BACKEND_CORE.md` (curated catalog) + `docs/openapi.json` (machine-readable contract). Out of scope for this backend repo. Supersedes the retired T16/T34 SPA path.

## §T — tasks

P0 — schema generalize + OpenAI pivot. Ids are monotonic, not positional: T12–T14 (dependent-module ports) are appended but run mid-phase. **P0 exec order (dependency-correct, I hand-drive — ⊥ `--next` id-order):** T1 → T2 → T15 → T3 → T12 → T13 → T14 → T4 → T5 → T6 → T7 → T8 → T9 → T10(gate) → T11. (T15 = DB rename, independent housekeeping, runs now.) Schema/tags + ports land before the OpenAI pivot so the suite compiles; gate last. **T16** (dashboard SPA redesign) PRUNED by T34 — backend-only pivot 2026-05-26; no view layer in repo (V-D1, §O).

**P0.5+ exec order (rescope 2026-05-26):** P0.5: T19 → T17 → T18 → T20 → T35 (seed/startup cleanup → service ports → anki ports → legacy-schema test prune → OpenAI-SDK test reshape). P1: T21 → T22 → T23. P2: T24 → T25 → T26 → T27 (models+migrations → deps → service seams → tests). P3: T28 → T29 → T30 (recall → grounded gen → persist). P4: T31 → T32 → T33 → T34 (discriminator persist → Notion write-back → adapter expansion → SPA prune, backend-only); P4 follow-up: T37 (PDF-qset adapter — CUT 2026-05-28, notes-only PDF ingress). See FORMAT.md for `st` legend.

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
| T37 | x | CUT (2026-05-28): pdf-qset PDF-question-set adapter ABANDONED — backend does lecture-notes PDF ingress only (`kb/pdf_ingest` → `atomic_facts`). Practice questions enter via the extension capture adapters (uworld/manual/web-qbank) WITH attempts; a bulk PDF import has no attempts + needs manual answer-key fill-in + vision question-structure parse = lowest-value/highest-cost source. KEPT from the design: `correct_choice` nullability (V-CAP1, shipped via T55) for future deferred-answer web captures. Adapter list drops pdf-qset (§A/§I.ext); `adapters/__init__.py` deferral comment updated to CUT. | V-CAP1,§A |
| T38 | x | (P4 follow-up) surface node tags on tutor reads: resolve `QuestionTag.node_id` (+ path via `OutlineLookup`) per question in `tutor/captures/recent` + `tutor/sessions/{id}/summary` (`topics`/`by_topic` are empty TODO stubs — `app/services/tutor/captures.py:30`, `sessions.py:85`). Return `node_id`(+path) in payload. Unblocks client node-labels on captures + session per-node breakdown (desktop `desktop/SPEC.md` ¶T1/¶T2) | V-O5,V-O1,V-D1,V-T1 |
| T39 | x | (P4 follow-up) system-status read endpoint `GET /api/v1/admin/status` (or `/healthz` extension): probe AnkiConnect reachability (`version`/`deckNames`), Notion token validity, OpenAI key reachability; fold in per-job last `TaskRun` status (`/admin/jobs` returns only `next_run_time`, no last-run outcome). Lets a client show real connection health ⊥ "scheduled"/"unknown" guesses. Unblocks desktop `desktop/SPEC.md` ¶T3 settings panel | V-D1,I.api,§A |
| T40 | x | (stabilization gate) isolate test config from developer `.env`: set test env at `tests/conftest.py` import — before any `app.config` import — (dummy `COACH_TOKEN`, unset `NOTION_API_TOKEN`/`OPENAI_API_KEY`, test DB URL) and/or rebuild the `app.config.settings` singleton; seed-or-skip AAMC outline so `OutlineLookup` resolves (⊥ implicit startup seed — V-O6; fixture materializes via importer or test skips). An autouse fixture alone is too late (singleton built at import). Gate (amended B4): clears the env-noise failure CLASSES (auth 401 from real `COACH_TOKEN`, `test_settings` Notion default, `OutlineNotSeeded` in `test_anki_sync`) — clean-checkout `uv run pytest` 27 failed/21 errors → 12 failed/0 errors; the residual 12 are a genuine unported-anki defect reassigned to T41 (B4). ⊥ whole-suite-0 here (needs T41). | V-TC1,V16,V-O6 |
| T41 | x | (stabilization, B4) port `app/services/anki/{tag_parser,sync}.py` off the retired 3-target onto `node_id`: `ParsedTag` carries `node_id` (resolved via `OutlineLookup` path, ⊥ `cc_code`/`topic_id`/`content_category_id`/`skill_number`); `sync` writes `AnkiNoteTag(node_id=…)` incl. the unparsed-row re-parse branch (`sync.py:423-430`); rewrite `tests/test_anki_sync.py` + `tests/test_anki_note_schema.py` assertions/constructors onto `node_id` (V-RB4); `/api/v1/anki/sync` route resolves-or-skips when outline unseeded (⊥ 500 on `OutlineNotSeeded`). Gate: residual 12 failures → 0 (whole suite green) | V-O5,V-RB2,V-T1,V-RB4 |
| T42 | x | (client-unblock, desktop ¶T5) question review detail: extend `GET /tutor/questions/by-qid/{qid}` (or sibling) with answer-distribution (aggregate `Attempt.selected_choice` count per choice over all attempts of the qid) + the user's `picked` choice + attempt-history (`[{attempted_at, is_correct, selected_choice, time_seconds}]`). Data-only, ⊥ verdict (V-M1) | V-M1,I.api |
| T43 | x | (client-unblock, desktop ¶T6) anki review surface: include per-card retention/retrievability on `GET /anki/review-queue` payload; extend `GET /anki/load-adherence` with a per-day reviewed-count series (`[{date, reviewed}]`, default 30d) alongside the existing projection — feeds the load-adherence chart ⊥ client-sampled array | V13,I.api,§A |
| T44 | x | (client-unblock, desktop ¶T7) per-node/subtree **mastery** read endpoint `GET /api/v1/outline/nodes/{id}/mastery` (+ course-level): port the FENCED `app/services/analytics.py` mastery rollup onto `OutlineNode` + `outline_subtree` (V-O1 set rollup, ⊥ legacy topic/cc joins) and re-expose via public API (⊥ private route). Unfences the surface behind V-RB1 | V-O1,V-O5,V-D1,V-RB1 |
| T45 | x | (client-unblock, desktop ¶T8) `GET /api/v1/concept-edges` read API (recent edges + by-node): expose `concept_edges` (`from`/`to` node, `kind`, `score`, `created_at`) for the connections feed. Substrate + similarity-edge derivation already built (T24,T28 = x); ⊥ infra-gated. Blocker = build the route + populate `concept_edges` (run embedding + similarity on real content; empty until then) | V-E2,V-O1,V-D1 |
| T46 | x | (client-unblock, desktop ¶T9) `GET /api/v1/atomic-facts` read API (by node / by pdf): expose `atomic_facts` (`text`, `node_id`, `pdf_source`, `page`, `extractor_version`). Substrate + ingest/extract/persist already built (T24,T26,T30 = x); ⊥ infra-gated. Blocker = build the route + populate `atomic_facts` (ingest PDFs → extract; empty until then) | V-KB1,V-T1,V-D1 |
| T47 | x | (client-unblock, desktop ¶T10) `GET /api/v1/notion/pages` read API: expose the `notion_pages` pointer index (`node_id`, `title`, `url`, `block_count`, `last_synced`, `status`) for the page-index view. Read of OUR pointer index per V-N1 (⊥ Notion read-back); one page per node (V-N2). Substrate + Notion write-out already built (T24,T32 = x); ⊥ infra-gated. Blocker = build the route + populate `notion_pages` (run write-out; empty until then) | V-N1,V-N2,V-D1 |
| T48 | x | (client-unblock, desktop ¶T11) `GET /api/v1/pdf-sources` read API (inbox): expose `pdf_sources` (`filename`, `pages`, `status`, `facts_count`, `ingested_at`, `node_id`, `sha`). Substrate + PDF ingest seam already built (T24,T26 = x); ⊥ infra-gated. Blocker = build the route + populate `pdf_sources` (drop PDFs → ingest runs; empty until then) | V-KB1,V-D1 |
| T49 | . | (deferred, `/check` §V drift 2026-05-27) strip residual legacy token names from FENCED anki reads `app/services/anki/{queries,state,retention}.py`: `topic_id`/`cc_code`/`content_category_id`/`skill_number` survive as commented-out / route-disabled stub params+returns (fenced imports `app/api/v1/anki.py:33-40`) → V-RB2 holds via fence escape-clause but breaches literal "zero references"; same fenced-but-named surface that hid B4. Rename → `node_id` or delete dead stubs. Low priority — fenced, ⊥ critical path | V-RB2,V-O5 |
| T50 | x | (orchestration, `/check` §I 2026-05-27) wire the built-but-idle LLM4Tag pipeline into the scheduler: define + register `run_embed_job` (embed new questions/facts/`outline_nodes` → `content_embeddings`, V-E1 version stamp), `run_grounded_tag_job` (recall→grounded→persist over `needs_categorization` Qs + untagged `atomic_facts`), `run_calibrate_job` (prune Conf<0.5→`manual_review`) in `app/scheduler.py`. Seams (`kb/recall`, `llm/grounded`, `kb/persist_tags`, `llm/calibrator`) built+tested — this is the runner + V41 partial-failure/`task_run` discipline. Lights up substrate so T45/T46 stop returning `[]`. Runners in `kb/jobs.py` (embed_pending + tag_pending); atomic_facts always tagged (course from fact), questions only when exactly one course exists. run_calibrate_job NOT a standalone job — calibration inline in run_grounded_tag_job (V69), standalone re-grade deferred (decision 2026-05-28) | V-L3,V69,V-E1,V-T2,V-T3,V41,I.job |
| T51 | x | (orchestration, `/check` §I 2026-05-27) PDF ingest + Notion write-out runners + endpoint: add `POST /api/v1/pdf/ingest {course_id, file}`; define + register `run_pdf_ingest_job` (poll `PDF_INBOX_DIR` → `kb/pdf_ingest` → extract → chunk → `atomic_facts` → embed) + `run_notion_sync_job` (one-way per-node upsert over `notion_pages` pointer, idempotent). Services `kb/pdf_ingest.py` + `kb/notion.py` built — wires scheduler + API. Lights up T47/T48 substrate | V-KB1,V-N1,V-N2,I.api,I.job |
| T52 | x | (docs, `/check` §I 2026-05-27) write `docs/PROMPT_OUTLINE_SCHEMA.md` — shipped template the user runs against own sources (PDF/screenshot/webpage) to generate an upload outline schema; §I outline-import references it, file absent | I.outline-import,§G |
| T53 | x | (redesign CUT, 2026-05-27) remove legacy MCAT categorizer + anki topic_resolver + orphaned dead code; relocate shared `OutlineLookup`. CUT: `app/services/categorizer/{llm,worker,cache,_text}.py`, `SUBJECT_TO_SECTION` in `outline_render.py`, `app/services/anki/topic_resolver{,_worker,_batch,_cache}.py`, `run_categorizer`+`run_anki_topic_resolver` scheduler jobs + `POST /api/v1/admin/questions/{id}/recategorize`; orphans `app/api/v1/{analyzer,recommendations}.py`, `app/services/analyzer/*`, `app/services/recommender.py`, `app/services/topic_subtree.py`, `app/services/llm/batch.py`, dead `run_feature_extraction_job` wiring + commented router lines; their tests (`test_mechanical_features`, `test_llm_batch`, `test_v_l2_harness`, relevant `test_fence_guards`/`test_anki_queries_smoke` entries). RELOCATE `OutlineLookup`+`normalize_typographic` → `app/services/outline/lookup.py` (domain-blind); update importers (scheduler, admin, admin_tags, anki sync+tag_parser, tutor flags/questions/outline). KEEP: AAMC seed, uworld/manual/web-qbank adapters, AnKing `tag_parser`. Gate: suite green after. | V-L4,V-O5,V-RB1,V-L3 |
| T54 | x | (notes-ingress redesign 2026-05-28) rewrite `app/services/kb/pdf_ingest.py`: render each PDF page→image (PyMuPDF, injectable) → OpenAI **vision** transcription per page → OpenAI structured-output atomic-fact extraction grounded to transcription → persist `atomic_facts` (`content_hash` dedup, `node_id` NULL). REPLACE pdfplumber-text parse + `split_atomic_candidates` regex. Vision + extraction clients injected + mocked at SDK boundary (V16); cost-logged (V-L1); `extractor_version` stamp. Rewrite `tests/test_kb_pdf_ingest.py`. T51 wires the endpoint/job over it; T50 categorizes its output (`node_id`). | V-KB3,V-KB4,V-KB2,V16,V45,V-L1,V-MIG2 |
| T55 | x | (V-CAP1, 2026-05-28) make `questions.correct_choice` NULLABLE: model `app/models/captures.py` `Mapped[Optional[str]]` nullable=True; migration `0006_correct_choice_nullable` (ALTER COLUMN, id≤32 V-MIG2); `ParsedCapture.correct_choice` stays required `str` (extension always supplies). NULL = answer pending for a future deferred-answer capture source. The one idea kept from cut T37. | V-CAP1,V-MIG2,I.schema |
| T56 | x | (V-CAP2, extension ¶O-1/¶T5, 2026-05-28) course-scope captures at ingest: add `course_slug: Optional[str]` to `CapturePayload` (`app/schemas/captures.py`, `extra='forbid'` stays); add `course_id` FK→courses on `questions` + `raw_captures` (`app/models/captures.py`, nullable back-compat; ondelete align w/ course-scoped tables — CASCADE, or SET NULL to keep attempt history — decide at build); migration `0007_question_course_id` (id≤32, V-MIG2); adapter `normalize_capture` (`app/services/adapters/extension_capture.py`) resolves slug→`course_id` (unknown slug → 422), stamps it on RawCapture + Question; scope `run_grounded_tag_job` / `kb/jobs.tag_pending` recall to `question.course_id` (drop "exactly one course" guard for course-stamped Qs, keep single-course fallback for NULL). | V-CAP2,V-L3,V69,I.schema,I.api,§A,V-MIG2 |
| T57 | x | (stabilization, B6+B7) port the two write paths still on the retired 3-target onto `node_id` (V-RB6): (1) `app/api/v1/admin.py` — `ManualTagBody` carries `node_id: int` (drop `topic_id`/`content_category_id`/`skill` + `_exactly_one`), `create_manual_tag` forwards `node_id=`, `_tag_row_payload` reads `row.node_id` (drop dropped-column reads); (2) `app/services/anki/assignment.py` — `_CANDIDATE_SQL_TOPIC`/`_CANDIDATE_SQL_CC` → one `node_id` subtree-rollup query over `outline_subtree` (V-O5/V-O1), `_fetch_candidates` takes a `node_id` scope, drop `anki_note_tags.topic_id`/`content_category_id`/`parsed_kind` + legacy `topics`/`content_categories` joins; drop vestigial `AnkiCardTagOut.topic_id`. Gate: `POST /api/v1/admin/questions/{id}/tags` + `POST /api/v1/anki/assignments` exercised green; `tests/test_v_rb6_no_retired_3target_writes.py` xfail flips to pass (remove markers); suite green. | V-RB6,V-T1,V-O5,V-O1,V-RB4 |

## §B — bug log

| id | date | cause | fix |
|-----|------|-------|-----|
| B1 | 2026-05-26 | alembic 0001_initial.py declared `task_runs.status` as `sa.Enum('running','succeeded','failed', name='task_run_status')` inside `op.create_table`. Postgres dialect auto-created the named TYPE on first CREATE TABLE, but the downgrade only called `drop_table('task_runs')` — the TYPE leaked. Re-running `upgrade head` after `downgrade base` then hit `DuplicateObjectError: type "task_run_status" already exists`. Surfaced by T27's full-roundtrip migration test; also explained the pre-existing failure of `test_question_features_schema::test_migration_apply_and_rollback`. | Added `op.execute("DROP TYPE IF EXISTS task_run_status")` in 0001_initial.py downgrade after `drop_table('task_runs')`. Recurrence guarded by V-MIG1. |
| B2 | 2026-05-26 | T4 swapped `anthropic`→`openai` across `app/services/llm/` but left 7 harness/eval scripts in `scripts/` (`eval_categorizer_models.py`, `compare_categorizer_v5_baseline.py`, `extract_features.py`, `extract_features_sample.py`, `compare_topic_resolver_v3_v4.py`, `run_categorizer.py`, `run_anki_topic_resolver_batch.py`) with `from anthropic import AsyncAnthropic` at module top-level — they survived as residual P0 debt because `anthropic` lingered in the venv from pre-pivot installs. T26's `uv sync` (triggered by T25 deps) trimmed that transient install; `tests/test_eval_script.py` then collection-errored with `ModuleNotFoundError: No module named 'anthropic'`, aborting the whole suite before any test ran. | Short-term: T26 added `collect_ignore_glob = ["test_eval_script.py"]` in `tests/conftest.py` to restore collection. Long-term: T36 ports or deletes the 7 scripts + their test + removes the collect_ignore_glob workaround. Recurrence guarded by V-RB5. |
| B3 | 2026-05-27 | test suite has no env override. `Settings` (`app/config.py:10`) sets `env_file=_BACKEND_ROOT/.env` + `env_ignore_empty=True`, and `app.config.settings` is a module-level singleton built at first import — so a developer's real `.env` bleeds into every test. Live `COACH_TOKEN` flips `verify_coach_token` 401 paths (`test_scheduler` list_jobs/trigger assert 200, get 401); real `NOTION_API_TOKEN` makes the field non-None (`test_settings::test_kb_substrate_notion_vars_default_none`) and would point `/admin/status` Notion probes at the live API; unseeded AAMC outline → `OutlineNotSeededError` (`app/services/categorizer/outline_lookup.py:61`) errors `test_anki_sync`. 27 failed / 21 errors from a clean checkout. Surfaced at T38 + T39 verification: a polluted baseline can't tell a true regression from env noise — the next real break hides in the 27. | V-TC1 (suite ⊥ read real `.env`/network; config from fixtures only) + T40 (conftest sets test env before `app.config` import / rebuilds singleton; seed-or-skip outline). |
| B5 | 2026-05-28 | T54's new migration declared revision id `0005_atomic_fact_extractor_version` (34 chars). `alembic_version.version_num` is `varchar(32)`, so `alembic upgrade head` failed at the `UPDATE alembic_version SET version_num='0005_atomic_fact_extractor_version'` with `asyncpg.exceptions.StringDataRightTruncationError: value too long for type character varying(32)`. Surfaced in the full migration-roundtrip tests (`test_kb_substrate_contract`, `test_kb_substrate_schema`, `test_question_features_schema` — each runs `alembic upgrade head`). Earlier ids (`0004_atomic_fact_tags`=21, `0003_kb_substrate`=17) stayed under the cap by luck. | Renamed revision id → `0005_atomic_fact_extractor` (26 chars), file `alembic/versions/0005_atomic_fact_extractor.py`. Recurrence guarded by V-MIG2 (revision id ≤32 chars). |
| B4 | 2026-05-27 | T40's `.env` isolation + AAMC seed fixture unmasked a pre-existing defect env noise had hidden (the B3 prediction). The AnKing anki tag-parse→sync path was never ported off the retired 3-target despite T18/V75 marking anki done. `app/services/anki/tag_parser.py`: `ParsedTag` still carries `topic_id`/`content_category_id`/`skill_number` and `parse_tag` calls `outline_lookup.content_category_id(cc_code)` — a method the generalized `OutlineLookup` no longer has (only `node_id_by_path`/`node`/`path_of`) → `AttributeError` on aamc_cc tags. `app/services/anki/sync.py:427-429,436-446` writes `AnkiNoteTag(topic_id=…, content_category_id=…, skill_number=…)` but the model (V75 note-as-unit) dropped those columns → `TypeError: 'topic_id' is an invalid keyword argument for AnkiNoteTag` on every sync write. Stale tests assert/construct the dropped columns (`test_anki_sync` ~L264/285/286/308/309, `test_anki_note_schema`). Hid because the `lookup` fixture raised `OutlineNotSeededError` before the broken sync body ran; T40 seeding made it run. Invariant gap: V-RB2 listed `{queries,state,retention}.py` but omitted `sync.py`+`tag_parser.py`. Suite 27 failed/21 errors → 12 failed/0 errors after T40. | Widened V-RB2 to cover `sync.py`+`tag_parser.py` (+ `skill_number`/`content_category_id` + the `OutlineLookup`-methods note). Port + test rewrite + route resolve-or-skip = T41. T40 gate amended to clear env-noise classes only; residual 12 → T41. |
| B6 | 2026-05-30 | `POST /api/v1/admin/questions/{question_id}/tags` never ported off the retired 3-target (same class as B4, different file — outside V-RB2's anki-only list). `app/api/v1/admin.py:83-98` `ManualTagBody` still carries `topic_id`/`content_category_id`/`skill` + an `_exactly_one` validator; the route (`admin.py:119-138`) forwards them as kwargs `topic_id=/content_category_id=/skill=` to `app/services/admin_tags.py:create_manual_tag`, whose T14/T2-ported signature is `(session, question_id, *, node_id, rationale=None)` (node_id-only) → `TypeError: create_manual_tag() got an unexpected keyword argument 'topic_id'` on any call. `_tag_row_payload` (`admin.py:101-112`) also reads dropped `row.topic_id`/`content_category_id`/`skill` off `QuestionTag` (node_id-only, V-T1) → `AttributeError`. Hid because no live caller + no test exercises the route (admin mutation routes also lack `verify_coach_token`). Surfaced documenting the API surface (wiki). | Recurrence guarded by V-RB6 (behavior-scoped: ⊥ retired 3-target in any live write path). Port route + `ManualTagBody` + `_tag_row_payload` → `node_id` = T57. |
| B7 | 2026-05-30 | `POST /api/v1/anki/assignments` candidate selection never ported off the retired 3-target (same class as B4/B6). `app/services/anki/assignment.py:77-139` `_CANDIDATE_SQL_TOPIC`/`_CANDIDATE_SQL_CC` (run by `_fetch_candidates:147-159`) `JOIN topics`/`content_categories` and reference `anki_note_tags.topic_id`/`content_category_id`/`parsed_kind IN ('aamc_topic','aamc_cc')` + a `:cc_code` param — none exist in the live node_id-only schema (V-T1, note-as-unit V75, §B4) → assignment creation fails at the SQL layer (`UndefinedTableError`/`UndefinedColumnError`). `AnkiCardTagOut.topic_id` is vestigial (always null). Lifecycle (`mark_skipped`/`mark_completed_manual`/`run_unlock_due`/`run_complete_unlocked`) unaffected — they operate on stored `card_ids`/`note_ids`. Invariant gap: V-RB2 lists only `{queries,state,retention,sync,tag_parser}.py`, omitting `assignment.py`. | Recurrence guarded by V-RB6. Port candidate SQL → `node_id` subtree rollup (`outline_subtree`, V-O5/V-O1) = T57. |

