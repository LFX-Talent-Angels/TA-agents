"""Roadmap under learn-order-v1: the gap per taxonomy, steps, stated order, caps, statuses."""

from __future__ import annotations

import json
from typing import Any

import pytest

from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1
from talent_angels.skills.roadmap import roadmap
from talent_angels.suites import UnknownSuiteError
from tests.fakes.order_suite import OrderSuite, edge, node, registry

# ---- ESCO: knowledge first, reuse levels, stated links, a cycle, a tie -------------------------

A = node("esco", "a", "Occupation", "analyst")
B = node("esco", "b", "Occupation", "scientist")


def _skill(key: str, label: str, skill_type: str, reuse: str) -> Any:
    return node("esco", key, "Skill", label, skill_type=skill_type, reuse_level=reuse)


SKILLS = {
    "sh": _skill("sh", "shared skill", "skill/competence", "cross-sector"),
    "k1": _skill("k1", "statistics", "knowledge", "cross-sector"),
    "k2": _skill("k2", "algebra", "knowledge", "transversal"),
    "m1": _skill("m1", "modelling", "skill/competence", "cross-sector"),
    "t1": _skill("t1", "teamwork", "skill/competence", "transversal"),
    "f1": _skill("f1", "field skill", "skill/competence", "sector-specific"),
    "j1": _skill("j1", "job skill", "skill/competence", "occupation-specific"),
    "k3": _skill("k3", "domain theory", "knowledge", "occupation-specific"),
    "c1": _skill("c1", "cycle one", "skill/competence", "cross-sector"),
    "c2": _skill("c2", "cycle two", "skill/competence", "cross-sector"),
    "tie2": _skill("tie2", "same label", "skill/competence", "transversal"),
    "tie1": _skill("tie1", "same label", "skill/competence", "transversal"),
    "o1": _skill("o1", "optional only", "skill/competence", "cross-sector"),
}


def _esco(shuffle: int = 0) -> OrderSuite:
    s = SKILLS
    needed = ["sh", "k1", "k2", "m1", "t1", "f1", "j1", "k3", "c1", "c2", "tie1", "tie2"]
    edges = [edge("HAS_SKILL", B, s[k], relation_type="essential") for k in needed]
    edges += [edge("HAS_SKILL", B, s["o1"], relation_type="optional")]
    edges += [edge("HAS_SKILL", A, s["sh"], relation_type="essential")]
    need = lambda a, b: edge("RELATED_TO", s[a], s[b], relation_type="essential")  # noqa: E731
    edges += [need("m1", "k1"), need("j1", "f1"), need("k3", "f1")]  # k3 is knowledge, f1 Field
    edges += [need("c1", "c2"), need("c2", "c1")]
    edges += [edge("RELATED_TO", s["t1"], s["k2"], relation_type="optional")]  # not a prerequisite
    return OrderSuite([A, B, *s.values()], edges, shuffle)


def _labels(data: dict[str, Any], step: int) -> list[str]:
    block = next(s for s in data["steps"] if s["step"] == step)
    return [i["label"] for i in block["items"]]


def _items(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {i["id"]: i for s in data["steps"] for i in s["items"]}


def test_esco_gap_is_the_targets_essential_skills_minus_the_starts() -> None:
    data = roadmap(registry(esco=_esco()), A.id, B.id)

    ids = set(_items(data))
    assert "esco:sh" not in ids  # the start has it
    assert "esco:o1" not in ids  # optional for the target: not in the ESCO gap
    assert ids == {f"esco:{k}" for k in SKILLS} - {"esco:sh", "esco:o1"}
    assert data["suites"]["esco"]["status"] == "ok"
    assert data["suites"]["esco"]["items"] == 11


def test_steps_follow_type_and_reuse_level_and_stated_links_come_first() -> None:
    data = roadmap(registry(esco=_esco()), A.id, B.id)

    assert [(s["step"], s["name"]) for s in data["steps"]] == [
        (1, "Foundations"),
        (2, "Core"),
        (3, "Field"),
        (4, "Job-specific"),
    ]
    assert _labels(data, 1) == ["algebra", "statistics"]
    # modelling needs statistics (stated); the tie keeps label then ID; the cycle comes last.
    assert _labels(data, 2) == [
        "modelling",
        "same label",
        "same label",
        "teamwork",
        "cycle one",
        "cycle two",
    ]
    # domain theory is knowledge, but it needs the Field skill: it moves to step 3, after it.
    assert _labels(data, 3) == ["field skill", "domain theory"]
    assert _labels(data, 4) == ["job skill"]
    by_id = _items(data)
    assert [i["id"] for i in data["steps"][1]["items"]][1:3] == ["esco:tie1", "esco:tie2"]
    assert by_id["esco:k3"]["step"] == 3
    assert "step:raised-by-stated-prerequisite" in by_id["esco:k3"]["reasons"]


def test_only_the_needed_end_of_a_stated_link_is_a_stated_prerequisite() -> None:
    by_id = _items(roadmap(registry(esco=_esco()), A.id, B.id))

    stated = {i for i, item in by_id.items() if item["stated_prerequisite"]}
    assert stated == {"esco:k1", "esco:f1", "esco:c1", "esco:c2"}
    assert by_id["esco:k1"]["reasons"] == [
        "esco:knowledge",
        "esco:reuse:cross-sector",
        "esco:essential-for:esco:m1",
    ]
    assert "esco:needs:esco:k1" in by_id["esco:m1"]["reasons"]
    assert by_id["esco:t1"]["stated_prerequisite"] is False  # an optional link states nothing


def test_a_cycle_is_broken_by_node_id_and_reported() -> None:
    data = roadmap(registry(esco=_esco()), A.id, B.id)

    assert "cycle_broken:esco:c1" in data["suites"]["esco"]["warnings"]
    assert data["warnings"].count("esco:cycle_broken:esco:c1") == 1


def test_per_step_caps_the_items_and_counts_the_rest() -> None:
    data = roadmap(registry(esco=_esco()), A.id, B.id, per_step=2)

    step2 = next(s for s in data["steps"] if s["step"] == 2)
    assert [i["label"] for i in step2["items"]] == ["modelling", "same label"]
    assert step2["more"] == 4
    assert all(s["more"] == 0 for s in data["steps"] if s["step"] != 2)
    with pytest.raises(ValueError, match="per_step"):
        roadmap(registry(esco=_esco()), A.id, B.id, per_step=0)


def test_the_result_is_byte_identical_and_independent_of_graph_order() -> None:
    def run(shuffle: int) -> str:
        return json.dumps(roadmap(registry(esco=_esco(shuffle)), A.id, B.id))

    assert run(0) == run(0) == run(1) == run(2)


def test_every_result_prints_its_meaning_and_names_the_policy() -> None:
    data = roadmap(registry(esco=_esco()), A.id, B.id)

    assert data["policy"] == "learn-order-v1"
    assert data["meaning"] == LEARN_ORDER_V1.meaning
    assert (
        "only items marked stated_prerequisite are prerequisites stated by a taxonomy"
        in (data["meaning"])
    )
    assert all(i["cite"].startswith("urn:esco:") for i in _items(data).values())


def test_the_lookup_budget_is_a_warning_not_a_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    from talent_angels.skills.roadmap import common

    monkeypatch.setattr(common, "LEARN_ORDER_V1", LEARN_ORDER_V1.__class__(lookup_budget=3))
    data = roadmap(registry(esco=_esco()), A.id, B.id)

    assert "esco:lookup_budget:3" in data["warnings"]
    assert "esco:reuse:unknown" in {r for i in _items(data).values() for r in i["reasons"]}


# ---- O*NET ---------------------------------------------------------------------------------------

OA = node("onet", "a", "Occupation", "Analysts")
OB = node("onet", "b", "Occupation", "Scientists")
ZONE = node("onet", "zone", "JobZone", "Job Zone Four")


def _onet(shuffle: int = 0) -> OrderSuite:
    sk = {
        n: node("onet", n, "Skill" if n.startswith("s") else "Knowledge", label)
        for n, label in {
            "s_prog": "Programming",
            "s_math": "Mathematics",
            "s_low": "Low importance",
            "s_small": "Small jump",
            "s_edge": "Exactly on the lines",
            "s_new": "New to the start",
            "s_sup": "Suppressed",
            "s_tie": "Programming",
            "k_math": "Mathematics",
        }.items()
    }

    def has(occ: Any, key: str, level: float, importance: float, rel: str = "HAS_SKILL", **p: Any):
        return edge(rel, occ, sk[key], level=level, importance=importance, **p)

    edges = [
        has(OA, "s_prog", 2.50, 4.0),
        has(OB, "s_prog", 3.88, 3.75),
        has(OA, "s_math", 3.88, 4.0),
        has(OB, "s_math", 5.00, 4.38),
        has(OA, "s_low", 1.0, 2.0),
        has(OB, "s_low", 4.0, 2.9),  # importance under 3.0
        has(OA, "s_small", 3.00, 4.0),
        has(OB, "s_small", 3.49, 4.0),  # jump 0.49 under 0.5
        has(OA, "s_edge", 3.00, 4.0),
        has(OB, "s_edge", 3.50, 3.0),  # jump 0.5 and importance 3.0: both lines are inclusive
        has(OB, "s_new", 2.0, 3.5),  # the start has no level for it
        has(OA, "s_sup", 1.0, 4.0),
        has(OB, "s_sup", 5.0, 4.0, recommend_suppress="Y"),
        has(OA, "s_tie", 2.5, 4.0),
        has(OB, "s_tie", 3.88, 3.75),  # same jump, importance and label as s_prog
        has(OA, "k_math", 4.67, 4.0, rel="HAS_KNOWLEDGE"),
        has(OB, "k_math", 5.68, 4.52, rel="HAS_KNOWLEDGE"),
        edge("HAS_JOB_ZONE", OA, ZONE),
        edge("HAS_JOB_ZONE", OB, ZONE),
        edge("USES_SOFTWARE", OB, node("onet", "tool", "Software", "A tool")),
    ]
    return OrderSuite([OA, OB, ZONE, *sk.values()], edges, shuffle)


def test_onet_keeps_importance_3_and_a_jump_of_half_a_level_with_both_levels() -> None:
    data = roadmap(registry(onet=_onet()), OA.id, OB.id)

    by_id = _items(data)
    assert set(by_id) == {
        "onet:s_prog",
        "onet:s_math",
        "onet:s_edge",
        "onet:s_new",
        "onet:s_tie",
        "onet:k_math",
    }
    assert by_id["onet:s_prog"]["levels"] == {
        "from": 2.5,
        "to": 3.88,
        "jump": 1.38,
        "importance": 3.75,
    }
    assert "onet:level 2.50→3.88" in by_id["onet:s_prog"]["reasons"]
    assert by_id["onet:s_new"]["levels"]["from"] is None
    assert by_id["onet:s_new"]["levels"]["jump"] == 2.0
    assert by_id["onet:k_math"]["step"] == 1  # knowledge is Foundations
    assert by_id["onet:s_math"]["step"] == 2  # O*NET skills are Core
    assert all(not i["stated_prerequisite"] for i in by_id.values())  # O*NET states no order
    assert data["warnings"] == ["onet:suppressed:1"]


def test_onet_orders_by_level_jump_then_importance_then_label_then_id() -> None:
    data = roadmap(registry(onet=_onet()), OA.id, OB.id)

    core = next(s for s in data["steps"] if s["step"] == 2)["items"]
    assert [i["id"] for i in core] == [
        "onet:s_new",  # jump 2.0
        "onet:s_prog",  # jump 1.38, importance 3.75, label Programming, id s_prog < s_tie
        "onet:s_tie",
        "onet:s_math",  # jump 1.12
        "onet:s_edge",  # jump 0.5
    ]
    assert json.dumps(data) == json.dumps(roadmap(registry(onet=_onet(5)), OA.id, OB.id))


def test_onet_reports_both_job_zones() -> None:
    zone = roadmap(registry(onet=_onet()), OA.id, OB.id)["job_zone"]

    assert zone["from"]["zone"] is None  # the fake's zone id has no number; the real one does
    assert zone["to"]["label"] == "Job Zone Four"


# ---- JobTech, BLS, SFIA -----------------------------------------------------------------

JA = node("jobtech", "a", "Occupation", "Dataanalytiker")
JB = node("jobtech", "b", "Occupation", "Data scientist")
BA = node("bls", "a", "Occupation", "Analysts")
BB = node(
    "bls",
    "b",
    "Occupation",
    "Scientists",
    entry_education="Bachelor's degree",
    work_experience="None",
    job_training="None",
)


def _jobtech() -> OrderSuite:
    s = {k: node("jobtech", k, "Skill", k) for k in ("sql", "python", "stats", "docs")}
    edges = [
        edge("HAS_SKILL", JA, s["sql"], relation_type="group"),
        edge("HAS_SKILL", JB, s["sql"], relation_type="group"),
        edge("HAS_SKILL", JB, s["stats"], relation_type="group"),
        edge("HAS_SKILL", JB, s["python"]),
        edge("HAS_SKILL", JB, s["docs"], relation_type="group"),
    ]
    return OrderSuite([JA, JB, *s.values()], edges)


def test_jobtech_gap_is_the_targets_skills_minus_the_starts_all_in_core() -> None:
    data = roadmap(registry(jobtech=_jobtech()), JA.id, JB.id)

    assert [(i["label"], i["step"]) for s in data["steps"] for i in s["items"]] == [
        ("docs", 2),
        ("python", 2),
        ("stats", 2),
    ]
    assert not any(i["stated_prerequisite"] for i in _items(data).values())


def test_bls_gives_a_credentials_block_and_no_steps() -> None:
    data = roadmap(registry(bls=OrderSuite([BA, BB], [])), BA.id, BB.id)

    assert data["steps"] == []
    assert data["credentials"] == [
        {
            "suite": "bls",
            "id": "bls:b",
            "label": "Scientists",
            "kind": "Occupation",
            "cite": "urn:bls:b",
            "entry_education": "Bachelor's degree",
            "work_experience": "None",
            "job_training": "None",
        }
    ]


def test_a_taxonomy_without_order_data_says_so_and_is_not_opened() -> None:
    sfia = OrderSuite([], [])
    data = roadmap(registry(sfia=sfia, esco=_esco()), A.id, B.id)

    assert data["suites"]["sfia"] == {
        "status": "no_order_data",
        "start": None,
        "target": None,
        "candidates": [],
        "items": 0,
        "warnings": [],
    }
    assert sfia.calls == []


# ---- statuses, merging, errors -----------------------------------------------------------------


def test_an_unclear_name_returns_candidates_and_a_missing_one_is_not_found() -> None:
    reg = registry(esco=_esco(), onet=_onet())
    data = roadmap(reg, "a", B.id, suites=["esco"])  # "a" is inside every label

    esco = data["suites"]["esco"]
    assert esco["status"] == "ambiguous"
    assert {c["side"] for c in esco["candidates"]} == {"from"}
    assert len(esco["candidates"]) <= 3
    assert data["steps"] == []

    missing = roadmap(reg, "nothing like it", B.id, suites=["esco"])
    assert missing["suites"]["esco"]["status"] == "not_found"


def test_an_id_of_one_taxonomy_is_never_searched_in_another() -> None:
    onet = _onet()
    data = roadmap(registry(esco=_esco(), onet=onet), A.id, B.id)

    assert data["suites"]["onet"]["status"] == "not_found"
    assert onet.calls == []
    assert data["suites"]["esco"]["status"] == "ok"


def test_a_skill_is_not_an_occupation_and_the_same_end_twice_is_flagged() -> None:
    reg = registry(esco=_esco())

    skill = roadmap(reg, "esco:k1", B.id)
    assert skill["suites"]["esco"]["status"] == "no_order_data"
    assert skill["suites"]["esco"]["warnings"] == ["not_an_occupation:Skill/Occupation"]

    same = roadmap(reg, A.id, A.id)
    assert same["suites"]["esco"]["status"] == "ok"
    assert same["suites"]["esco"]["warnings"] == ["same_start_and_target"]
    assert same["steps"] == []


def test_a_broken_taxonomy_does_not_hide_the_others_and_unknown_suites_are_refused() -> None:
    data = roadmap(registry(esco=_esco(), onet=ConnectionRefusedError("down")), A.id, B.id)

    assert data["suites"]["onet"]["status"] == "error:ConnectionRefusedError"
    assert data["suites"]["esco"]["status"] == "ok"
    with pytest.raises(UnknownSuiteError):
        roadmap(registry(esco=_esco()), A.id, B.id, suites=["nope"])


def test_each_taxonomy_fills_only_its_own_items_and_unclear_names_stay_unguessed() -> None:
    reg = registry(esco=_esco(), onet=_onet())
    data = roadmap(reg, "esco:a", "esco:b")
    onet = roadmap(reg, "onet:a", "onet:b")

    assert {i["suite"] for s in data["steps"] for i in s["items"]} == {"esco"}
    assert {i["suite"] for s in onet["steps"] for i in s["items"]} == {"onet"}
    both = roadmap(reg, "a", "b")  # both names are ambiguous in both fakes: no items, no guess
    assert [s["status"] for s in both["suites"].values()] == ["ambiguous", "ambiguous"]
