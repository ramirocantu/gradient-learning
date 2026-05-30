# Documentation Task: gradient-server

## Role & Goal

You are documenting **gradient-server**, a **Python/FastAPI** back-end for a web-based personal learning management system (LMS). Your job is to produce complete, accurate reference documentation in **Markdown**, structured for hosting on the **GitHub wiki**. Document every server function and map it to the user-facing features it supports.

**Output location:** Write all `.md` files into the existing `docs/wiki/` folder. Inspect that folder first — if pages already exist, edit and extend them rather than restructuring; only add new pages where there's a genuine gap.

## Ground Rules

- **Source of truth is the code, not assumptions.** Read the actual implementation before documenting anything. Never invent endpoints, parameters, return shapes, or behavior. If something is ambiguous, document what the code does and flag the ambiguity in a `> **Note:**` callout.
- **Markdown only**, GitHub-flavored (GFM). Use tables, fenced code blocks with language hints, and relative wiki links (`[[Page Name]]` style or `[text](Page-Name)` — match GitHub wiki conventions).
- **No code dumps.** Show signatures and minimal illustrative snippets, not entire function bodies.
- Keep a consistent voice: precise, present-tense, second-person where addressing the API consumer.

## Phase 1 — Discover (do this first, do not skip)

1. Map the repository: directory tree, entry point(s), framework, language/runtime, build and run commands.
2. Identify the architecture layers: routing/controllers, services/business logic, data access/models, middleware, utilities, config.
3. List external dependencies and integrations (database, auth provider, file storage, email, third-party APIs).
4. Catalog **every** server function/handler. For each, note its layer, file location, and call relationships.
5. Group functions by **feature** (e.g., authentication, course/content management, progress tracking, assessments, scheduling). Infer feature groupings from routes and naming, then verify against logic.

**FastAPI specifics to capture:**
- The `FastAPI()` app instance and how it's created (factory function vs. module-level), plus `lifespan`/startup-shutdown handlers.
- All `APIRouter` instances, their `prefix` and `tags`, and where they're mounted via `include_router`. Map routers to features.
- Path operation decorators (`@router.get`, `@app.post`, etc.) — capture method, path, `status_code`, `response_model`, and `tags`.
- **Pydantic models** as the typed contract: request bodies, `response_model`s, and schemas. These ARE your Data-Models — document fields, types, validators, and `Config`.
- **Dependencies** (`Depends(...)`): auth/permission guards, DB session injection, pagination, etc. Document what each provides and which endpoints rely on it.
- Auth scheme (OAuth2/JWT/`Security(...)` scopes) and how protected routes declare requirements.
- Background tasks, `async` vs sync handlers, and any middleware registered on the app.
- The auto-generated OpenAPI docs (`/docs`, `/openapi.json`) — note them, but document from source, not just the schema.

Produce a short discovery summary before writing pages, so the structure is grounded in what actually exists.

## Phase 2 — Wiki Structure

Create these pages (one Markdown file each). Adjust to fit what the code actually contains, but keep the spirit.

- **Home.md** — Project overview, purpose, tech stack, high-level architecture diagram (Mermaid), and a linked table of contents to all pages.
- **Getting-Started.md** — Prerequisites, environment variables, install/build/run, how to run tests.
- **Architecture.md** — Layered breakdown, request lifecycle, data flow, key design decisions.
- **Data-Models.md** — Each model/entity (Pydantic schemas and ORM models): fields, types, validators, constraints, relationships. Distinguish request/response schemas from persisted DB models. Use tables.
- **One page per feature** (e.g., `Feature-Authentication.md`, `Feature-Courses.md`, etc.) — see template below.
- **API-Reference.md** — Consolidated endpoint table (method, path, auth required, brief description, link to the feature page).
- **Function-Index.md** — Alphabetical or by-module index of every documented function with a one-line description and a link to where it's fully documented.
- **Glossary.md** — Domain terms (LMS-specific vocabulary).

## Per-Function Documentation Template

For every server function/handler, document:

```
### functionName

**Location:** `path/to/file.ext:lineRange`
**Layer:** router | service | model (Pydantic/ORM) | dependency | middleware | util
**Feature(s):** which user-facing feature(s) this supports

**Purpose:** One or two sentences on what it does and why it exists.

**Signature:**
\`\`\`<lang>
functionName(param1: Type, param2: Type): ReturnType
\`\`\`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|

**Returns:** Shape and meaning of the return value / response.

**Side effects:** DB writes, external calls, emitted events, state changes.

**Errors:** Failure conditions and what is thrown/returned.

**Called by / Calls:** Upstream callers and downstream dependencies.
```

## Per-Feature Page Template

```
# Feature: <Name>

**Summary:** What the user can do and why it matters.

**User flow:** Step-by-step of the typical interaction (Mermaid sequence diagram if useful).

**Endpoints:** Table of routes powering this feature.

**Functions involved:** Linked list from controller → service → data layer.

**Data models touched:** Links to Data-Models entries.

**Edge cases & business rules:** Validation, permissions, limits.
```

## For Endpoints / Routes

Document each: HTTP method, full path (including router prefix), auth/permission requirements (via `Depends`/`Security`), path/query/body params (typed table, noting Pydantic body models), success response (`status_code` + `response_model` + example JSON), error responses (status + condition, including `HTTPException` raises and validation 422s), the `Depends(...)` it relies on, and the path operation function it maps to.

## Phase 3 — Quality Pass

- Every function appears in **Function-Index.md** and on its feature page.
- Every endpoint appears in **API-Reference.md**.
- All internal links resolve (GitHub wiki uses hyphenated page names; verify link targets).
- No orphaned pages; Home links to everything.
- Examples are valid and reflect real request/response shapes from the code.
- Flag anything undocumented-by-necessity (dead code, TODOs, unclear intent) in a final **Open-Questions** section on Home.

## Deliverable

A set of `.md` files in the **`docs/wiki/`** folder, named for GitHub wiki conventions (spaces become hyphens, e.g., `Feature-Authentication.md`). Confirm the file list at the end and note any gaps where code access or clarification is needed.