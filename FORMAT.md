# FORMAT — spec encoding

Caveman compression. All technical substance stays; fluff dies. Drop articles/filler/hedging. Fragments OK. Preserve identifiers, paths, code, errors verbatim. Symbol `⊥` = "must not / forbidden".

## Sections

`SPEC.md` carries these `##`-headed sections, in order:

- §G — goal (1 line)
- §A — architecture
- §C — constraints
- §I — interfaces (schema, api, env, jobs)
- §V — invariants (numbered `V<N>` / `V-<area><n>`; monotonic, never reused)
- §P — phases
- §O — open items
- §T — tasks (pipe table)
- §B — bug log (pipe table)

## §T — tasks

Pipe table. One row per ordered task.

```
| id | st | goal | cites |
```

- `id` — `T<n>`, monotonic, never reused.
- `st` — status cell: `.` todo · `~` in progress · `x` done. Build skill flips this cell only.
- `goal` — caveman imperative, one line.
- `cites` — comma list of §V/§I deps this task must respect (`V-O1,I.schema`). Build plan cites these.

## §B — bug log

Pipe table. Header row always present; rows appended by backprop.

```
| id | date | cause | fix |
```

- `id` — `B<n>`, monotonic.
- `date` — ISO `YYYY-MM-DD`.
- `cause` — root cause, caveman.
- `fix` — invariant that now guards it (`V<N>`) or the change made.

## Rules

- Numbering monotonic across §V/§T/§B — never reuse an id.
- spec skill is sole mutator of SPEC.md, except build flips §T `st` cells.
- Every bug → a §B row (backprop). New invariant optional but preferred.
