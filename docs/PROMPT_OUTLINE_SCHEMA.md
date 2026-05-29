# Outline-schema prompt template

Gradient is a **course-agnostic** study system: you add a course (biochem,
anatomy, organic chem, a board exam — anything) and study against its outline.
The outline is **a structured file you upload**, not something Gradient parses
out of your PDFs (§G). Gradient owns only *validate* + *materialize*; you bring
the schema.

This doc is the shipped prompt template. Run it against **your own sources** — a
syllabus PDF, a screenshot of a table of contents, a webpage — in your own LLM
session, then upload the JSON it produces (see [Uploading](#uploading)).

> MCAT/AAMC is just one example course (`slug: aamc`), not a special case.
> The bundled `app/seeds/aamc_outline.schema.json` is a real, valid schema you
> can read as a worked reference; re-uploading it restores the MCAT outline.

---

## The schema shape

```json
{
  "course": { "slug": "bio101", "name": "Intro Biology", "description": "optional" },
  "nodes": [
    { "path": ["Cell Biology"], "kind": "unit", "name": "Cell Biology", "position": 1 },
    { "path": ["Cell Biology", "Membranes"], "kind": "topic", "name": "Membranes", "position": 1 }
  ]
}
```

- `course.slug` — short, unique, url-safe id for the course (e.g. `bio101`).
- `course.name` — human title. `course.description` — optional.
- `nodes` — a flat list; the **tree is encoded by `path`**, not by nesting.
  - `path` — array of names from the root down to this node
    (`["Cell Biology", "Membranes"]` is the *Membranes* node under *Cell Biology*).
  - `name` — **must equal the last element of `path`**.
  - `kind` — the per-level label (`unit`, `topic`, `section`, `lecture`,
    `concept`, … — your choice; it just describes what a level *means*).
  - `position` — optional integer ordering among siblings (default `0`).
  - `depth` — optional; if you include it, it **must equal `len(path) - 1`**
    (root is depth 0). Easiest to omit and let it be inferred.

There is no depth limit — a 2-level course and a 4-level one are equally valid.

---

## Rules the output MUST satisfy

The uploader validates the **whole file atomically** — one broken node rejects
the entire upload with every error listed at once. Make the generated schema
obey all of these:

1. `course.slug` and `course.name` are present and non-empty.
2. `nodes` is non-empty.
3. Every `path` is a non-empty list of non-empty strings.
4. **No path segment may contain the reserved delimiter `" >> "`** (space-greater-greater-space).
5. Every node has a non-empty `kind` and `name`, and `name` equals the **last**
   `path` segment.
6. **No duplicate paths** (two nodes can't share the same full path).
7. **Closed parent chain**: for every non-root node, its parent path
   (`path` minus the last element) must also appear as some node in the list.
   So if `["Cell Biology", "Membranes"]` exists, `["Cell Biology"]` must too.
8. `position` (and `depth` if present) are non-negative integers.
9. **One `kind` per depth**: all nodes at the same depth should share the same
   `kind` (e.g. every depth-0 node is a `unit`); mixing kinds at a depth is
   rejected as a likely typo.

---

## The prompt

Copy everything in the block below into your LLM, then attach or paste your
source material (syllabus, outline screenshot, textbook table of contents, etc.).

````text
You convert a course outline into a strict JSON schema for the Gradient study
tool. I will give you my source material (a syllabus / table of contents /
screenshot / webpage text). Produce ONE JSON object and NOTHING else.

Output shape:
{
  "course": { "slug": "<short-url-safe-id>", "name": "<title>", "description": "<optional>" },
  "nodes": [
    { "path": ["Top", "Sub", "Leaf"], "kind": "<level-label>", "name": "Leaf", "position": <int> }
  ]
}

Hard rules (the upload is rejected if any is broken):
- "course.slug" is short, lowercase, url-safe; "course.name" is the human title.
- The tree is encoded by "path" (root-to-node name array), NOT by nesting.
- "name" MUST equal the last element of "path".
- A node's parent path (path without its last element) MUST also appear as its
  own node. Emit every intermediate level as its own node, top-down.
- No path segment may contain the substring " >> " (space, two >, space).
- No two nodes may have the same "path".
- "kind" labels what a level means (e.g. unit / topic, or section / lecture /
  concept). ALL nodes at the same depth MUST use the same "kind".
- "position" is an integer ordering siblings (start at 1); omit "depth".
- Keep names concise and human-readable; preserve the source's own wording.

Return only the JSON object — no markdown fences, no commentary.
````

---

## Worked example

A small two-level course (`unit` → `topic`). This is a valid upload:

```json
{
  "course": { "slug": "bio101", "name": "Intro Biology", "description": "Fall semester" },
  "nodes": [
    { "path": ["Cell Biology"], "kind": "unit", "name": "Cell Biology", "position": 1 },
    { "path": ["Cell Biology", "Membranes"], "kind": "topic", "name": "Membranes", "position": 1 },
    { "path": ["Cell Biology", "Organelles"], "kind": "topic", "name": "Organelles", "position": 2 },
    { "path": ["Genetics"], "kind": "unit", "name": "Genetics", "position": 2 },
    { "path": ["Genetics", "Mendelian Inheritance"], "kind": "topic", "name": "Mendelian Inheritance", "position": 1 }
  ]
}
```

For a deeper (4-level) real-world schema, see
`app/seeds/aamc_outline.schema.json` (`section → foundational_concept →
content_category → topic`).

---

## Uploading

Two calls against the backend (`/api/v1/*`). Replace `localhost:8000` with your
backend origin.

1. **Create the course** (the body's `slug` must match the schema's `course.slug`):

   ```bash
   curl -X POST http://localhost:8000/api/v1/courses \
     -H 'Content-Type: application/json' \
     -d '{ "slug": "bio101", "name": "Intro Biology" }'
   # → { "id": 7, "slug": "bio101", ... }
   ```

2. **Import the outline** into that course id (body = the full schema file):

   ```bash
   curl -X POST http://localhost:8000/api/v1/courses/7/outline:import \
     -H 'Content-Type: application/json' \
     --data-binary @bio101.schema.json
   # → { "course": {...}, "nodes_imported": 5 }
   ```

   - On any rule violation the import returns **422** with `{"errors": [...]}`
     listing every problem — fix the file and re-upload.
   - Re-importing a schema for an existing course **replaces** that course's
     outline wholesale (the old tree is wiped, the new one inserted in one
     transaction) — so editing the file and re-uploading is the update path.

Once imported, `GET /api/v1/courses/{id}/outline` returns the node tree, and the
KB pipeline (PDF ingest → atomic facts → grounded tagging) tags content against
these nodes.
