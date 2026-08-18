# Next: `ta-agent` REPL (Rich TUI + logs + open memory)

**Status:** not implemented. Written 2026-08-19 after the ESCO MVP
presentation. This is the **first** post-MVP build. Pathfind waits.

Official `cli query` stays one-shot JSON for scripts and quality cards.
The product path is a ChatGPT-style window:

```text
$ ta-agent
$ ta-agent --resume last
```

---

## 1. Rich TUI (required in the first slice)

Do **not** print a JSON blob and call it a chat. The console must look
and feel like a conversation.

**Library:** [Rich](https://github.com/Textualize/rich) for rendering
(panels, markdown, tables, theme). [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)
or Rich `Live` + a line editor for the input box. Prefer Rich-first so
we stay a library, not a full Textual app, unless the layout needs
fixed regions.

**What the screen should show**

```text
┌ ta-agent  session: 2026-08-19-dev  model: laguna-s-2.1  tokens ▓▓░░ 12k/32k ┐
│                                                                              │
│  You     developer                                                           │
│                                                                              │
│  Assistant   I found several ESCO occupations matching “developer”.          │
│              Which one did you mean?                                         │
│              1. software developer                                           │
│              2. web developer                                                │
│              …                                                               │
│              capability: locate   confidence: 0.42   warning: ambiguous      │
│                                                                              │
│  You     the first one — what essential skills?                              │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  /help  /quit  /save  /resume     last turn 1.2s  $0.000                     │
│  ▸ _                                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Surface | Job |
| --- | --- |
| Transcript | User / assistant bubbles. Render the **answer as markdown**, not a raw string dump. |
| Status bar | Session id, model, capability, warnings, last-turn latency, cost, context-budget bar. |
| Tool / graph strip | Compact: `search_nodes → 25` / `get_neighbors → 24`. Full node lists stay in the local detail file, not the chat. |
| Input | Multiline-capable prompt. Slash commands complete on `/`. |
| Theme | Dark-friendly Rich theme. No LiteLLM “Provider List” ads. |

Slash commands (first slice): `/help`, `/quit` (and `/exit`), `/save [name]`,
`/resume [name|last]`, `/clear` (new session, does not delete files).

Keep `python -m talent_angels.cli query` for CI and piping.

---

## 2. Better logging (required in the first slice)

Today: one global `runlog.jsonl` plus gitignored `data/local/query-details/<id>.md`,
and the live CLI dumps JSON on stdout. That is fine for a one-shot demo.
It is **not** enough for a session TUI.

**Split three streams. Do not mix them.**

| Stream | Where | Who reads it | What it is |
| --- | --- | --- | --- |
| **A. Conversation** | `data/local/sessions/<session_id>/transcript.jsonl` | Human + resume | User / assistant lines, timestamps, slash events. |
| **B. Turn telemetry** | `data/local/sessions/<session_id>/runlog.jsonl` **and** the existing repo `runlog.jsonl` (or `RUNLOG_PATH`) | `cli report`, evals | One `RunLogRecord` per turn (tokens, cost, tools, warnings). |
| **C. Operator log** | `data/local/sessions/<session_id>/debug.log` (stderr when `TA_LOG_LEVEL=DEBUG`) | Developers | Structured lines: prompt budget, trim events, LLM errors. **Never** the default TUI output. |

Rules:

- The TUI is the human UI. JSON does not scroll by in the chat pane.
- `/save` and auto-save write **A + B**. Resume reads **A** and the last
  typed session facts; it does not replay tools.
- Full neighbor / node dumps stay in `query-details/` (already there).
  The TUI links them (“25 nodes — details saved”) instead of printing them.
- Log **context budget** every turn: `input_tokens`, `context_limit`,
  `context_trimmed`, how many transcript lines were dropped.
- Do not log API keys. Session dirs stay under `data/` (gitignored).
- `cli report` should grow a `--session <id>` filter once sessions exist.

This is the “better logging approach”: session-scoped files + structured
telemetry + a quiet TUI. It is **not** a new cloud observability product.

---

## 3. OPEN — memory and persistence (discuss; do not implement a store)

**This is an open architecture topic.** It is not decided. Do not merge a
vector database, and do not write user chat into the ESCO Neo4j graph,
until this is discussed and recorded as an ADR.

Two different “memory” problems get mixed up. Keep them separate.

### 3.1 In-session memory (first REPL slice — decided enough to build)

While one `ta-agent` window is open (and when you `/resume` that file):

- Persist the **transcript** and **typed last-locate facts** (candidate
  labels/ids, last unique occupation) as **files** (see stream A).
- What the **LLM sees** is a **budgeted** view (last *N* lines and/or a
  summary + those typed facts). Full history can live on disk; it must
  not all go in the prompt (small free-model context windows).
- Tests: a 20-turn fake session does not grow the prompt without bound.

That is enough to do: `developer` → ambiguous list → “the first one,
what skills?”

### 3.2 Cross-day user knowledge base (NOT decided)

When a mentee or later end user comes back next week, where does *their*
knowledge live — distinct from the ESCO map?

| Option | What it stores | Attractive when | Cost / risk |
| --- | --- | --- | --- |
| **A. Files only** | Session dirs + runlog (slice 1) | Resume today; no new infra | No “that nurse we picked last week” recall across many sessions |
| **B. Vector store** | Embeddings of turns / user facts | Fuzzy “similar to what I said” | Extra service; licensing; easy to treat prose as taxonomy fact |
| **C. Knowledge graph** | Typed user nodes/edges in Neo4j **separate from ESCO** | Bind “user → chose occupation X” as data | Schema + identity; must not invent ESCO ids |
| **D. Both** | Graph for bindings, vectors for fuzzy recall | Long-lived personal KB | Two stores; easy to over-build |

**Slice 1 uses A.** B/C/D stay open.

Constraints that apply to **any** later store:

- ESCO remains the only cited taxonomy. User memory is labeled *user
  session*, never as graph fact.
- Node IDs stay suite-scoped. No silent “user node = occupation.”
- Evidence is a pointer, not a payload (licensing).
- Determinism stays in tools. The store does not walk ESCO.

**How to close the topic:** short design discussion (mentees + mentors),
then one ADR. Until then: no Chroma / Qdrant / Pinecone, no user chat
rows in the ESCO database.

---

## 4. Out of scope for this slice

- Pathfind skill
- Changing default `LLM_MODEL`
- Phoenix / LangSmith as the source of truth (JSONL stays)
- Occupation-specific `if "nurse"` hacks
