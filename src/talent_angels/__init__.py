"""LFX Talent Angels — the assistant runtime over skill/task/occupation taxonomies.

One main assistant (LangGraph loop) owns the user's goal and dispatches four
map-work capabilities implemented as skills + tools (see ARCHITECTURE.md and
TA-workspace ADR-0003):

    - skills.locate    : free text -> node candidates + confidence  (Resolve)
    - skills.connect   : neighbors/hierarchy of a resolved node     (Reveal)
    - skills.pathfind  : routes between two resolved nodes          (Compose)
    - skills.evaluate  : rank routes under a named policy           (Rank)

Supporting layers:
    - assistant : intent -> plan -> dispatch -> merge -> answer
    - contracts : typed results (Pydantic v2) crossing every boundary
    - runlog    : one structured record per turn
    - api       : thin FastAPI edge (no reasoning here)

Graph suites (ingestion, schemas, suite tools) live in the sibling repo
TA-taxonomies and are consumed as a versioned library.
"""

__version__ = "0.1.0"
