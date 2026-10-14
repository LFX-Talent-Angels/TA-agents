# Batch tagging (`ta-agent tag`)

`ta-agent tag` reads a client's structured document — a competency standard
(CONOCER's own shape), or a plain CSV/XLSX table for any other business —
decides which rows are worth tagging, searches every attached taxonomy suite
(ESCO, O*NET, …) for each row, asks a model which hits really match, and
writes one Excel workbook back. It is a separate pipeline from the chat
assistant: no plan cache, no user memory, no 45-second turn deadline. Business
rules (what a row's *kind* is, what it's called, which suite kinds to search)
live in a **profile** (`profile.py`), not in code — see "Profiles" below.

## What it does, in order

1. **Read** the project's input file(s) — a Neo4j `IS_PART_OF` tree export, or
   a flat CSV/XLSX table (see "Inputs").
2. **Select** which rows get tagged, and what *kind* each one is (performance,
   product, knowledge, …), from a profile file — business rules as data, not
   code.
3. **Rewrite** each row into short search phrases (one model call per group of
   sibling rows).
4. **Search** every attached taxonomy suite for each row.
5. **Choose**: a model call asks which of the search hits actually match the
   row — it can only pick from the numbered candidates, or say none fit.
6. **Verdict**: code, not the model, turns that into `accepted` / `review` /
   `adjacent` / `gap` (see below).
7. **Write** `out/<doc>-tagged.xlsx` (Summary, Review, a document sheet named
   after the profile's `document_label` — "Standard" for CONOCER, "Document"
   for the generic profile — and Gaps) and `out/<doc>-gaps.csv`.

Re-running the same command is cheap: unchanged rows are reused from a
project-local ledger, not re-searched or re-asked.

## Folder layout

```
<project>/
  inputs/            # the client's export file(s) — what gets tagged
  inputs/context/     # background material only; never tagged itself
  out/               # the workbook(s), out/<doc>-gaps.csv, out/history/
  .ta-tag/           # project memory: ledger, rewrites, reviews, gaps, run-log
  tag.yaml           # optional: overrides the built-in profile for this project
```

`<project>` can also just be a single input file, or a bare folder of input
files (no `inputs/` subfolder) — `ta-agent tag` figures out which shape it's
looking at.

## Inputs

Two readers ship today (`src/talent_angels/tagging/readers/`); the first one
that recognises a file reads it — nothing downstream of `read()` cares which
one did.

- **Neo4j tree JSON** (`neo4j_tree.py`, CONOCER's export shape): a JSON array
  of `{n, r, m}` Cypher rows (child, relationship, parent) forming one
  `IS_PART_OF` tree per file.
- **A CSV next to a JSON of the same name** (Neo4j exports both) is not tagged:
  it has no node ids, so it cannot rebuild the tree. The run checks it anyway
  and says either "same N rows as the JSON, nothing lost" or which rows or
  columns it has that the JSON lacks ("NOT used yet").
- **CSV / XLSX table** (`table.py`): a flat spreadsheet with one row per line
  to tag. Columns are matched by header name, case-insensitively, in English
  or Spanish:

  | Role | Header aliases | Required? |
  | --- | --- | --- |
  | id | `id`, `row_id`, `row id`, `code`, `#`, `no`, `número`, `numero` | no — falls back to `1`, `2`, … in row order |
  | parent | `parent`, `parent_id`, `parent id`, `padre` | no — a blank/missing parent means the row hangs off a synthetic root (the file's own name) |
  | text | `text`, `description`, `descripción`/`descripcion`, `content`/`contenido`, `statement` | **yes** — a file with no text-like column isn't recognised as a table at all |
  | type | `type`, `kind`, `tipo`, `label` | no — becomes the row's Neo4j-style `label`, for a profile to key on if it wants to |

  CSV delimiter (`,`, `;`, or tab) is sniffed automatically. For an `.xlsx`,
  the first sheet is used unless `tag.yaml` names another one. Column names
  and sheet can always be overridden explicitly in `<project>/tag.yaml`:

  ```yaml
  table:
    sheet: Data       # xlsx only; default is the first sheet
    id: Code
    parent: Parent ID
    text: Description
    type: Category
  ```

A parent id that doesn't exist among the file's rows is a hard error (naming
the offending row); a blank text cell is silently skipped (it's formatting,
not content).

## Profiles

A **profile** (`src/talent_angels/tagging/profile.py`, data in
`tagging/profiles/*.yaml`) is a business's tagging rules: what counts as a
kind heading, which kinds actually get tagged, how a row's text gets cleaned,
which suite kinds each tagged kind routes to, and what the business calls its
own document (`document_type`, `document_label` — this is the only thing that
varies the model's own system prompts; CONOCER's wording hasn't changed).

Two ship built-in:

- **`competency_standard`** (the default) — CONOCER's shape: kind headings
  (Performances, Products, Knowledge, …), row rules for "Associated
  Occupations:" lines, Spanish aliases throughout.
- **`generic`** — no business-specific structure at all: every row is tagged
  as a plain `item`, routed to `skill`/`task`/`knowledge`. Good for a flat
  table with no section headings of its own.

Which one a run uses, in order:

1. `--profile NAME` or `--profile path/to/your-profile.yaml` on the command line.
2. `base_profile: <name-or-path>` in `<project>/tag.yaml`.
3. `competency_standard`, otherwise.

Either way, `<project>/tag.yaml` is then merged on top: any top-level key you
set there (`kinds`, `cleaners`, `include_kinds`, `routing`, `depth`, …)
replaces the matching key of the base profile wholesale — see
`src/talent_angels/tagging/profiles/competency_standard.yaml` for every key
and what it does, and `profiles/generic.yaml` for the minimal shape. Writing
your own profile from scratch (a third business with its own structure) means
copying one of those two files and pointing `--profile`/`base_profile` at it.

### Use it for another business

A university wants its course catalogue tagged against ESCO. Its export is a
flat spreadsheet, not a CONOCER-style standard, so:

1. Put the file at `data/local/<project>/inputs/courses.xlsx`, with columns
   like `Code | Parent | Type | Description` (a course row, its units as
   children, each unit's capacities as grandchildren — or just one flat list,
   if that's all it has).
2. Run it with the built-in generic profile, no `tag.yaml` needed:

   ```bash
   uv run ta-agent tag data/local/<project> --profile generic --workers 4
   ```

3. Open `out/courses-tagged.xlsx` — same Summary/Review/Gaps sheets as
   CONOCER's, with the document sheet named **Document** (generic's
   `document_label`) instead of **Standard**.

If the university later wants its own kind headings (e.g. tag "Learning
Outcome" rows differently from "Assessment Criteria" rows), write a
`tag.yaml` next to `inputs/` with `base_profile: generic` plus whichever
`kinds`/`include_kinds`/`routing` keys it needs — same mechanism CONOCER's own
overrides use.

## Prerequisites

- Docker Neo4j running, loaded with the taxonomy suite(s) you want (see the
  main [README](../README.md) §§1–2).
- `.env` with `LLM_PROVIDER` and the matching API key, unless you're running
  `--no-llm`.

## Commands

```bash
# Search-only, no model calls — fast, free, good for a first look.
uv run ta-agent tag data/local/sprint7/pilot-run --no-llm

# Full run with the model, 4 concurrent `choose` calls (search itself always
# stays sequential — embedding search is not thread-safe).
uv run ta-agent tag data/local/sprint7/pilot-run --workers 4

# Just a few rows, to try changes cheaply.
uv run ta-agent tag data/local/sprint7/pilot-run --limit 20
uv run ta-agent tag data/local/sprint7/pilot-run --rows 66-91,100

# Re-run only the rows whose verdict wasn't final (review/adjacent/gap);
# human decisions are never touched.
uv run ta-agent tag data/local/sprint7/pilot-run --refresh

# Ignore the ledger entirely and re-tag everything from scratch.
uv run ta-agent tag data/local/sprint7/pilot-run --fresh
```

| Flag | What it does |
| --- | --- |
| `--no-llm` | Search only; verdict is always `review` (top hit) or `gap` (no hits) — no model spend. |
| `--limit N` | Tag at most N rows (after `--rows`, if both are given). |
| `--rows "66-91,100"` | Tag only these row ids. |
| `--suites esco,onet` | Only these suites (default: every suite the registry has attached). |
| `--depth leaves\|headings\|all` | Which rows count as taggable (overrides the profile). |
| `--profile NAME\|PATH` | Built-in profile name (e.g. `generic`) or a path to a `.yaml` profile file (see "Profiles"). |
| `--workers N` | Concurrent `choose` model calls. |
| `--refresh` | Re-run `review`/`adjacent`/`gap` rows; reuse `accepted`, every human decision, and every row the reviewer already retried. |
| `--fresh` | Ignore the ledger (still never overwrites a human decision or redoes a reviewer's retry). |

## The review loop

1. Open `out/<doc>-tagged.xlsx`, sheet **Review**. Each row needing a look has
   a **Decision ▼** dropdown and a **Hint** cell.
2. Pick one:
   - **Approve** — the suggested tag is correct. On a `gap` row it confirms
     the gap (same as Mark as gap); on an `error`/`unavailable` row there is
     nothing to approve, so it is refused with a warning.
   - **Reject** — it's wrong; the next run retries with more context (same
     as "No" on the Gaps sheet).
   - **Use alternative 2/3/4** — one of the listed alternatives is the right
     one instead.
   - **Mark as gap** — no taxonomy entry fits; record it as a confirmed gap.
   - **Retry with hint** — give a hint (e.g. "this is about software
     testing"). The hint is searched as well, and its best matches head the
     candidate list; the model is asked again with the hint and more of the
     standard's own context (parent heading, then grandparent + sibling lines
     on a second retry). Up to 2 retries per row; a 3rd becomes a confirmed
     gap automatically. The pilot's own second look doesn't count as one.
     A retry is only done once it has produced an answer: if the run is
     stopped, the model call fails, or the row isn't part of this run
     (`--rows`, `--limit`, `--suites`), the next run does it, hint included.
   - **Use my tag** — put the taxonomy's own id (e.g. `esco:skill:…`,
     `onet:task:3967`) or exact label in Hint. Pasting a whole tag cell works
     too: the id in it is used. Several ids keep exactly those tags, e.g. to
     keep only the first of two suggested tags, paste just its id.
3. On the **Gaps** sheet, one row per confirmed-or-likely gap: answer
   **Yes** ("really is a gap"; `Sí`/`si`/`y` work too) or **No** ("an existing
   skill fits" — same effect as Retry). Yes on an `adjacent` row keeps its
   partial tags and confirms only the missing part. A different decision for
   the same row on the Review sheet is a conflict: neither is applied, and the
   run says so.
4. Save the file and run the exact same `ta-agent tag` command again. Only
   the rows you decided on are touched; everything else is reused untouched.
   When decisions were applied, the previous workbook is copied to
   `out/history/` first, and a
   **Previous decision** column on the Review sheet shows what was applied
   and when.
5. Decisions only ever apply once: re-saving without changing anything is a
   0-model-call, sub-second no-op.
6. Anything that could not be applied — a misspelt decision, a Hint with no
   Decision, an alternative number that doesn't exist, a tag not found — is
   reported as a warning and stays in its cell in the new workbook to fix.

### What each verdict means

| Verdict | Meaning |
| --- | --- |
| `accepted` | The model found an exact match, and the graph itself agrees (lexical rank or search rank). |
| `review` | Something worth a human look: an exact claim the graph doesn't yet corroborate, a `--no-llm` top hit, or a retry that's still inconclusive. |
| `adjacent` | No exact match, but one or more taxonomy entries cover *part* of the line. |
| `gap` | Nothing in the taxonomy covers this line — comes with a *proposed* skill name, clearly labelled as model inference, never taxonomy fact. |
| `human_accepted` / `human_gap` | The reviewer's own decision; never overwritten automatically. |
| `unavailable` | The suite was down this run (not cached — retried every run until it's back). |
| `n/a` | This suite has no concept of this row's kind (e.g. no "occupation" node type) — not a gap. |

### Editing on a Mac without Excel (Apple Numbers)

Numbers saves your edits as a separate `.numbers` file, which this tool cannot
read. After setting decisions, choose **File > Export To > Excel** and save it
over `out/<doc>-tagged.xlsx`, replacing it. If a `.numbers` file in `out/` is
newer than the workbook, the run stops before doing anything and says so. The
exported file keeps everything the tool needs (an extra "Export Summary" sheet
that Numbers adds is ignored); the dropdown may disappear, but typing the
decision works the same. Sorting or filtering the sheets in Numbers is fine:
rows are matched by their hidden `key` column (Numbers shows it), never by
position, so leave the `key`, `suite`, `alt_ids` and `fp` columns as they are.
If the export lands in `out/` under another name, the run points it out.
Finder's `.DS_Store` and other macOS system files in `inputs/` are ignored; a
`.numbers` input must be exported to Excel or CSV first.

### Known limitations

- If the model call itself fails during a retry, that row shows `error`
  (sorted to the top of the Review sheet) until the next run repeats the
  retry with the same hint.
- Most rows of an emerging-field standard come back `adjacent`: the
  taxonomies (ESCO v1.2.1, O*NET 31.0) have no prompt-engineering entries.
  No precision figure exists until an expert has reviewed the tags.
- The ledger keys on the `ta-taxonomies` code version, not the data loaded
  in Neo4j: after reloading a taxonomy, re-run with `--refresh`.

## Cost and time

Measured on the CONOCER pilot standard (129 tagged units, ESCO + O*NET):

- Full run (`--workers 4`): **~9 minutes, ~$1.40**.
- Re-run with no changes: **< 1 second**, 0 model calls.
- Re-run after a handful of review decisions: **~15 seconds**, a few cents.
- `--no-llm`: **~36 seconds**, $0.

## Customising further

- **A new business with its own input shape or kind rules**: see "Inputs" and
  "Profiles" above — usually a `tag.yaml`, sometimes a new profile file; no
  code change needed either way.
- **A genuinely new input format** (neither a Neo4j tree nor a flat table):
  implement the `Reader` protocol (`src/talent_angels/tagging/readers/base.py`:
  `can_read(path)`, `read(path) -> Document`) and register it in
  `src/talent_angels/tagging/readers/__init__.py`. The rest of the pipeline is
  format-agnostic.
- **New taxonomy suite**: register it in `suites/registry.py` the normal way
  (see the main README) — it automatically gets searched, its own Review
  rows, and its own document-sheet columns. A suite that doesn't support a
  given row's kind just shows `n/a` for that row; it never produces a false
  gap.
