"""Pathfind fills "my job"/"my goal" from the saved profile (PF-5)."""

from __future__ import annotations

from talent_angels.memory.paths import user_md
from talent_angels.memory.profile import resolve_pathfind_subject


def _saved(*lines: str) -> None:
    user_md().parent.mkdir(parents=True, exist_ok=True)
    user_md().write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_my_job_resolves_to_this_suites_own_standing_line() -> None:
    _saved(
        "STANDING[onet]: Dancers  [onet:27-2031.00]   since: 2026-10-07",
        "STANDING[esco]: dance teacher  [esco:1eb5]   since: 2026-10-07",
        "GOAL: data analyst  [esco:d3ed]",
    )

    assert resolve_pathfind_subject("my job", suite_name="esco") == "dance teacher"
    assert resolve_pathfind_subject("my current job", suite_name="onet") == "Dancers"


def test_my_goal_resolves_regardless_of_suite() -> None:
    _saved("GOAL: data analyst  [esco:d3ed]")

    assert resolve_pathfind_subject("my goal", suite_name="esco") == "data analyst"
    assert resolve_pathfind_subject("my goal", suite_name="onet") == "data analyst"


def test_a_real_title_is_returned_unchanged() -> None:
    _saved("GOAL: data analyst  [esco:d3ed]")

    assert resolve_pathfind_subject("software developer", suite_name="esco") == "software developer"


def test_my_job_with_nothing_saved_falls_back_to_the_literal_text() -> None:
    """No STANDING for this suite: Locate reports an honest miss on "my job"
    rather than this guessing a title (ADR-0008: never invent)."""
    assert resolve_pathfind_subject("my job", suite_name="esco") == "my job"


def test_dispatch_heuristic_resolves_my_job_and_my_goal() -> None:
    from talent_angels.assistant.graph import _dispatch_heuristic
    from talent_angels.assistant.llm_plan import PlanDraft
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.suites.schema import SuiteSchema
    from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult

    _saved("STANDING[esco]: nurse  [esco:n1]   since: 2026-10-07", "GOAL: midwife  [esco:m1]")

    schema = SuiteSchema(
        skill_rel_types=("HAS_SKILL",),
        optional_rel_values=frozenset({"optional"}),
        group_rel_type=None,
        group_node_kinds=frozenset(),
    )
    seen: list[str] = []

    class _Suite:
        suite_schema = schema

        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            seen.append(text)
            node = FakeNode(
                f"esco:occupation:{text}", "Occupation", text, source="esco", source_id=text
            )
            return FakeToolResult(candidates=[FakeCandidate(node, 0.95, "exact_pref")])

        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            return FakeToolResult(warnings=["no_neighbors"])

        def enumerate_paths(self, *a: object, **k: object) -> FakeToolResult:
            return FakeToolResult(warnings=["no_path"])

    plan = build_plan_for_capability("pathfind", suites=("esco",))
    draft = PlanDraft(target="pathfind", subject="my job", secondary_subject="my goal")
    state = {"question": "path from my job to my goal", "plan": plan, "plan_draft": draft}

    out = _dispatch_heuristic(state, suite=_Suite(), suite_name="esco")

    assert seen == ["nurse", "midwife"]
    assert [n.pref_label for n in out["result"].nodes[:2]] == ["nurse", "midwife"]
