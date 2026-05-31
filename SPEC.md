# SPEC — RCA-10 · Test workflows and features of backend

One issue = one worktree = one spec. Linear `RCA-10` (High, milestone **MVP local-only**) is the
ledger; this file is the encoded detail. Authored 2026-05-31 from interactive scoping. Scope: add
**workflow-level** testing above the ~60 existing unit/schema/contract tests — two phases per
workflow: (1) manual exploratory pass against the live backend, log breakage; (2) codify the
critical green path as E2E pytest driving the HTTP contract.

## §G — goal

Prove three end-to-end backend pipelines work as a whole — not just their units — and leave behind
durable E2E coverage that drives the public `/api/v1/*` HTTP surface:

1. **Capture → Attempt → Grounded Tag** — `POST /captures` (uworld source adapter) → `Question` + `Attempt`
   rows (+ media), `needs_categorization=true`; then the SAME grounded arc as PDF over the question:
   `embed_pending` (question `stem_plain`) → `retrieve_candidates` recall → `generate_grounded_tags`
   (structured pick + V69 calibration) → persisted `question_tags` (node_id, source='llm', confidence) +
   `needs_categorization` flipped to `false`.
2. **PDF → Grounded Tag** — full LLM4Tag arc: `POST /pdf/ingest` → `atomic_facts` (node_id NULL) →
   `embed_pending` (fact + outline-node vectors) → `retrieve_candidates` recall → `generate_grounded_tags`
   (structured pick + V69 logprob calibration) → persisted `atomic_fact_tags` (node_id, source='llm', confidence).
3. **Outline → Mastery** — `POST /courses` + `outline:import` → materialize tree → tag targets → mastery rollup.

Each workflow runs the loop: **manual pass first** (find breakage on the live stack) → **backprop** any
bug to §B → **E2E test** locks the green path. Manual-pass findings drive the test assertions.

## §A — approach

- **Two test altitudes.** Manual pass = `mise run dev` + HTTP/curl against a real local stack (real
  OpenAI for embed/grounded-tag where unavoidable). E2E pytest = httpx/TestClient against the app,
  OpenAI mocked at the SDK boundary (`tests/_openai_mocks.py`), pinned to `gradient_test`.
- **HTTP-contract-first** for E2E. Tests assert the wire contract (routes, status, response shape) +
  observable DB state via read endpoints. Sanctioned in-process exceptions: the grounded-tag arc has
  **no HTTP trigger** (scheduler-only, `_do_run_grounded_tag`→`tag_pending`) and `kb/recall.py` has no
  route — T5 drives `embed_pending`→`tag_pending` in-process, then observes via `GET /atomic-facts?node_id=`.
- **PDF→Tag step order (carry-correct):** ingest writes facts with `node_id` NULL; recall's C2T path
  needs outline-node vectors, so `embed_pending` (embeds facts **and** nodes) MUST run before `tag_pending`.
  Fixture imports the outline first so nodes exist to embed + recall against.
- **One grounded arc, two entities.** `embed_pending` + `tag_pending` are shared: they embed/tag outline
  nodes, atomic_facts, AND questions in one pass. A question's `entity_text = stem_plain`; it is scoped to
  its OWN `course_id` (V-CAP2, stamped by the capture adapter), with a single-course fallback for an
  unscoped (NULL course_id) question and a skip-with-log when the course is ambiguous. Success flips
  `needs_categorization → false`. So the Capture and PDF workflows converge on the same tagging code —
  the question path needs the question's course outline imported + embedded for recall to be non-empty.
- **Fixtures land first.** No sample capture payload or test PDF exist in the repo; T1 crafts them +
  reusable conftest helpers before any workflow task.
- **Out of scope, untouched:** Anki cycle (sync/assign/review/retention — avoids AnkiConnect dep) and
  Notion write-out. ⊥ new tests there; ⊥ edits to existing Anki/Notion tests.

## §C — constraints

- Per-branch DB derived by mise (⊥ set `DATABASE_URL`); the suite pins `gradient_test` via `tests/conftest.py`.
- `OPENAI_API_KEY` present at `~/.config/gradient/secrets.json` — usable by the manual pass only.
  E2E ⊥ real OpenAI calls (carry V16): mock at the SDK boundary.
- `app/seeds/aamc_outline.schema.json` = the Outline-import fixture (validate-then-materialize).
- ⊥ committed test PDF. Manual pass (T4) uses a real lecture PDF kept **local + uncommitted** (gitignore the smoke dir). E2E (T5) needs no PDF file: vision is mocked and `ingest_pdf(renderer=...)` is injectable, so a fake renderer returns stub page-images and the upload bytes are synthetic.
- `Attempt.time_seconds` ⊥ actionable — ⊥ asserted as a performance signal (carried hard constraint).
- ⊥ new backend framework/ORM/route to enable testing. Tests consume the existing surface as-is; a
  gap that needs a new route is logged to §B + Linear, not silently patched here.

## §I — surfaces under test

```
# Capture → Attempt
POST /api/v1/captures                      CapturePayload{source=uworld,...} → IngestResponse
                                           (UnknownSource/UnknownCourse → 422)
                                           dedup key = questions.qid UNIQUE: repeat qid → same
                                           question_id + NEW attempt_id (⊥ (source,external_id))
                                           course_slug→course_id; ABSENT slug→fallback (NULL course_id),
                                           UNKNOWN slug→422 (UnknownCourseError) — not the same path
                                           real wire (ext 0.4.0, verified T2): uworld_aamc_tags = 6-level
                                           hierarchy strings ("Chapter:|Lesson:|Skill:|SubSkill:|Subject:|Unit:"),
                                           ⊥ flat names; media[] → `media` rows + bytes on disk (hash-sharded);
                                           attempt carries time_seconds + flagged; clean parse → parse_warnings NULL
GET  /api/v1/attempts/{attempt_id}/notes   observe attempt persisted
   grounded tag (shared arc, no HTTP trigger — scheduler-only):
   embed_pending  app/services/kb/jobs.py   in-process: embeds question stem_plain + outline nodes
   tag_pending    app/services/kb/jobs.py   in-process: entity_kind=QUESTION, course-scoped (V-CAP2),
                                           single-course fallback for NULL course_id; flips needs_categorization
   persists question_tags(node_id, source='llm', confidence, manual_review per V-T3)
GET  /api/v1/questions/by-qid/{qid}         question detail; resolves QuestionTag.node_id → tags + >> paths
GET  /api/v1/questions/by-attempt-id/{id}   same, keyed by attempt

# PDF → Grounded Tag (full LLM4Tag arc)
POST /api/v1/pdf/ingest                    multipart course_id:Form + file:UploadFile → PdfIngestResponse
                                           (unknown course → 404)
GET  /api/v1/pdf-sources                   pdf_sources rows + status='ingested'
GET  /api/v1/atomic-facts?pdf_source_id=&node_id=&limit=   facts; node_id NULL pre-tag, set post-tag
   embed_pending      app/services/kb/jobs.py    in-process: embeds facts + outline nodes → content_embeddings
   retrieve_candidates app/services/kb/recall.py  in-process: C2T/C2C2T/T2T → RecallResult (no route)
   generate_grounded_tags app/services/llm/grounded.py  structured pick + V69 calibration → GroundedResult
   tag_pending        app/services/kb/jobs.py    in-process orchestrator (_tag_one); persists atomic_fact_tags
                                           (scheduler entry _do_run_grounded_tag — no HTTP trigger)

# Outline → Mastery
POST /api/v1/courses                       {slug,name,description?} → 201 course
POST /api/v1/courses/{course_id}/outline:import   {course,nodes} schema → materialize (atomic, V-O2)
GET  /api/v1/courses/{course_id}/outline   node tree
GET  /api/v1/outline/nodes/{node_id}/mastery        compute_node_mastery
GET  /api/v1/outline/courses/{course_id}/mastery    compute_course_mastery (subtree rollup, V-O1)
```

All routes above are `X-Coach-Token`-gated (`verify_coach_token`).

## §V — invariants

- V1: E2E tests drive only the public HTTP surface (httpx/TestClient) — assert the contract + observable
  DB via read endpoints, ⊥ call services directly. Sole exception: `kb/recall.py` (no public route).
- V2: E2E ⊥ real OpenAI — mock at the SDK boundary (`build_openai_client` / `tests/_openai_mocks.py`),
  carry V16. The PDF→Tag arc mocks **four** call sites: vision-transcribe + fact-extract (ingest),
  embeddings (`embed_pending`), tagging structured-output + calibrator logprobs (`generate_grounded_tags`).
  Calibrator mock returns logprobs so `calibrated_confidence` is real-shaped (V69). Real key — manual pass only.
- V3: Each E2E test self-seeds its fixtures (own course/nodes) against `gradient_test`; ⊥ cross-test
  shared state, ⊥ ordering dependence between tests.
- V4: Every manual-pass breakage → a §B row; if recurrence is assertion-catchable, the fixing E2E test
  carries a regression assertion (and a new §V when it generalizes). ⊥ silent fix.
- V5: Out-of-scope surfaces (Anki cycle, Notion write-out) get ⊥ new tests and ⊥ edits to existing tests.
- V6 (corrected from T2 manual pass): ABSENT `course_slug` → fallback (`course_id` NULL, single-course
  `tag_pending` rule); UNKNOWN `course_slug` → 422 (`UnknownCourseError`). Distinct paths — the E2E
  asserts both: absent persists with NULL course, unknown rejects with 422.
- V7: The PDF→Tag E2E asserts the **persisted tag**, not just recall: an `atomic_fact_tags` row with
  `source='llm'`, non-NULL `confidence`, a resolved `node_id`, and `manual_review` consistent with
  the `<0.5` threshold (V-T3). Empty recall ⇒ no LLM call ⇒ no tag row (assert the empty path too).
- V8: `embed_pending` runs before `tag_pending` in the arc; the test ⊥ assert a tag before embeddings +
  node vectors exist (recall would return empty and the assertion would be vacuous).
- V10 (question grounded tagging): a captured question with `needs_categorization=true` runs the shared
  grounded arc (`embed_pending`→`tag_pending`, entity_kind QUESTION, `entity_text=stem_plain`). On a
  non-empty recall it persists ≥1 `question_tags` row (`source='llm'`, non-NULL `confidence`, resolved
  `node_id`, `manual_review` per V-T3) and flips `needs_categorization→false`. Scoped to the question's
  own `course_id` (V-CAP2); an unscoped question falls back to the sole course iff exactly one exists,
  else is skipped (course ambiguous) with `needs_categorization` left `true`. The E2E asserts both the
  tagged path and the ambiguous-skip path; recall must be non-empty (V8 — outline imported + embedded).
- V9 (snapshot semantics — confirmed intended, B1): a capture is a **full snapshot** of the source
  question. ∀ re-capture(same `qid`): `questions.uworld_aamc_tags` ← incoming tags verbatim — absent/empty
  incoming ⇒ NULL (⊥ merge-with-stored), and a tag change re-flags `needs_categorization=true`. Per-capture
  provenance lives in `raw_captures.raw_json`, ⊥ in the denormalized question column. T3 E2E asserts the
  clobber-on-reupload path as **expected**, ⊥ a regression to guard.

## §T — tasks

Exec order (I hand-drive — ⊥ `--next` id-order): T1 → T2 → T9 → T3 → T4 → T5 → T6 → T7 → T8.
(T9 = live grounded-tag pass over the captured question; appended id, runs after T2's capture pass and
before the T3 E2E that codifies both.) Per workflow: manual pass (find/backprop) precedes its E2E test.
`st`: `.` todo · `~` wip · `x` done.

| id | st | goal | cites |
|-----|----|------|-------|
| T1 | x | fixtures: craft uworld `CapturePayload` sample + reusable `conftest` helpers (auth header, course seed, 4-site OpenAI mock wiring, fake `renderer` returning stub page-images). ⊥ committed PDF — E2E uses fake renderer + synthetic bytes; manual T4 supplies its own local PDF | V2,V3,I |
| T2 | x | manual pass — Capture→Attempt (synthetic curl + real chrome-extension capture, qid 404824): `mise run dev`; POST a uworld capture (with + without course_slug); verify Question+Attempt+tags rows; log breakage to §B | V4,V6,I |
| T3 | . | E2E pytest — Capture→Attempt→Grounded Tag: POST capture → assert IngestResponse + Question/Attempt/media; dedup (repeat qid→same Q + new attempt); V6 absent vs unknown slug; V9 clobber-on-reupload; then mock 4 OpenAI sites, import+embed outline, `embed_pending`→`tag_pending`, assert `question_tags` (V10) via `GET /questions/by-qid` + ambiguous-skip path | V1,V2,V3,V6,V9,V10,I |
| T4 | . | manual pass — PDF→Grounded Tag (full arc): POST `/pdf/ingest` (real course+PDF, real OpenAI) → run `embed_pending` → run `tag_pending` (or scheduler `_do_run_grounded_tag`); verify pdf_sources status, atomic_facts, content_embeddings, recall candidates, persisted atomic_fact_tags(node_id, source='llm', confidence); log breakage to §B | V4,V7,V8,I |
| T5 | . | E2E pytest — PDF→Grounded Tag (full arc): mock all four OpenAI sites (vision/extract/embed/tagging+calibrator); seed+import outline; POST ingest → assert PdfIngestResponse + `/pdf-sources` + `/atomic-facts`; in-process `embed_pending`→`tag_pending`; assert persisted tag (V7) + empty-recall no-tag path | V1,V2,V3,V7,V8,I |
| T6 | . | manual pass — Outline→Mastery: POST course + `outline:import` (AAMC seed); GET outline tree; tag a question/fact to a node; GET node + course mastery; verify subtree rollup; log breakage to §B | V4,I |
| T7 | . | E2E pytest — Outline→Mastery: create course → import AAMC schema → assert tree → seed tagged attempts → assert node + course mastery rollup (set-union, V-O1) | V1,V3,I |
| T9 | . | manual pass — grounded-tag the captured question (live, real OpenAI): ensure qid 404824's course (mcat-2020) outline is imported; run `embed_pending`→`tag_pending`; verify `question_tags` persisted (node_id, source='llm', confidence) + `needs_categorization` flipped false; `GET /questions/by-qid/404824` surfaces tags; log breakage to §B | V4,V10,I |
| T8 | . | findings report: summarize manual-pass breakage + final E2E coverage; comment on Linear RCA-10; flip §T cells; ensure `mise run check` green | §G |

## §B — bug log

| id | date | cause | fix |
|-----|------|-------|-----|
| B1 | 2026-05-31 | Manual pass T2: re-capture (same `qid`) omitting/empty `uworld_aamc_tags` NULLs `questions.uworld_aamc_tags` + re-flags `needs_categorization` (`extension_capture.py:212,232,235`: incoming None != stored → `tags_changed` → clobber). Surfaced when a repeat POST without tags wiped a stored `["Biochemistry"]`. | **No code fix — confirmed intended** (capture = full snapshot; cleared tag = source no longer presents it). Per-capture provenance preserved in `raw_captures.raw_json` (verified cap5=`["Biochemistry"]`, cap7=`[]`). Documented as V9; T3 E2E asserts the clobber as expected. |
