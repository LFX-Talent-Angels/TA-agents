"""Pathfind fills "my job"/"my goal" from the saved profile (PF-5)."""

from __future__ import annotations

from talent_angels.memory.paths import user_md
from talent_angels.memory.profile import (
    find_pathfind_references,
    pathfind_reference_language,
    resolve_pathfind_subject,
)


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


def test_my_job_with_nothing_saved_returns_none_not_the_literal_phrase() -> None:
    """No STANDING for this suite: returning the literal phrase would have
    Locate search for "my job" and report a misleading "no node for that
    phrase" miss — the real cause is we don't know the job yet, not a bad
    search (#94)."""
    assert resolve_pathfind_subject("my job", suite_name="esco") is None
    assert resolve_pathfind_subject("my goal", suite_name="esco") is None


def test_pathfind_reference_kind_identifies_job_and_goal_phrases() -> None:
    from talent_angels.memory.profile import pathfind_reference_kind

    assert pathfind_reference_kind("my job") == "job"
    assert pathfind_reference_kind("my current role") == "job"
    assert pathfind_reference_kind("my goal") == "goal"
    assert pathfind_reference_kind("my career goal") == "goal"
    assert pathfind_reference_kind("software developer") is None


def test_my_job_and_goal_phrases_also_match_spanish_and_swedish() -> None:
    _saved(
        "STANDING[esco]: nurse  [esco:n1]   since: 2026-10-07",
        "GOAL: midwife  [esco:m1]",
    )

    assert resolve_pathfind_subject("mi trabajo actual", suite_name="esco") == "nurse"
    assert resolve_pathfind_subject("mitt nuvarande jobb", suite_name="esco") == "nurse"
    assert resolve_pathfind_subject("mi meta", suite_name="esco") == "midwife"
    assert resolve_pathfind_subject("mitt mål", suite_name="esco") == "midwife"


def test_pathfind_reference_language_identifies_each_supported_language() -> None:
    assert pathfind_reference_language("my job") == "en"
    assert pathfind_reference_language("mi trabajo actual") == "es"
    assert pathfind_reference_language("mitt mål") == "sv"
    assert pathfind_reference_language("software developer") is None


def test_find_pathfind_references_orders_by_position_in_the_sentence() -> None:
    job, goal = find_pathfind_references("¿cómo paso de mi trabajo a mi meta en ESCO?")
    assert job is not None and goal is not None
    assert job[0] == "mi trabajo"
    assert goal[0] == "mi meta"
    assert job[1] < goal[1]

    # the reverse order: goal named first
    job2, goal2 = find_pathfind_references("mi meta es X, mi trabajo es Y")
    assert job2 is not None and goal2 is not None
    assert goal2[1] < job2[1]

    assert find_pathfind_references("software developer") == (None, None)


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


def test_dispatch_heuristic_never_searches_my_job_or_my_goal_literally() -> None:
    """Nothing saved: dispatch must report an honest "unknown" warning and
    never call search_nodes with the literal phrase "my job"/"my goal" (#94)."""
    from talent_angels.assistant.graph import _dispatch_heuristic
    from talent_angels.assistant.llm_plan import PlanDraft
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.suites.schema import SuiteSchema
    from tests.fakes.taxonomy import FakeToolResult

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
            return FakeToolResult()

        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            return FakeToolResult(warnings=["no_neighbors"])

        def enumerate_paths(self, *a: object, **k: object) -> FakeToolResult:
            return FakeToolResult(warnings=["no_path"])

    plan = build_plan_for_capability("pathfind", suites=("esco",))
    draft = PlanDraft(target="pathfind", subject="my job", secondary_subject="data analyst")
    state = {"question": "path from my job to data analyst", "plan": plan, "plan_draft": draft}

    out = _dispatch_heuristic(state, suite=_Suite(), suite_name="esco")

    assert seen == []
    assert out["result"].warnings == ["pathfind_profile_unknown:job"]
    assert out["result"].nodes == []
