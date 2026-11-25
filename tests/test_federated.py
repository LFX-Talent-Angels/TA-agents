"""All-taxonomy map: merge, subset, per-suite status, crosswalk links and seeding, route mode."""

from __future__ import annotations

import pytest

from talent_angels.skills.connect.federated import map_all, map_all_route
from talent_angels.suites import SuiteRegistry, UnknownSuiteError
from tests.fakes.suite import (
    DEV,
    HAS_PYTHON,
    PYTHON,
    FakeSuite,
    fake_registry,
    suite_factory,
)
from tests.fakes.taxonomy import FakeNode, FakeToolResult
from tests.test_graph_mapper import GraphSuite, _edge


def _occ(node_id: str, label: str, kind: str = "Occupation", **props: object) -> FakeNode:
    return FakeNode(
        id=node_id,
        kind=kind,
        label=label,
        source=node_id.split(":")[0],
        source_id=f"https://example.test/{node_id}",
        properties=props,
    )


def _skill(node_id: str, label: str) -> FakeNode:
    return _occ(node_id, label, "Skill")


def _registry(*, bls_props: dict[str, object] | None = None) -> SuiteRegistry:
    bls = GraphSuite(
        [
            _occ("bls:occupation:15-1252", "Software Developers", **(bls_props or {})),
            _skill("bls:skill:adapt", "Adaptability"),
        ],
        [
            _edge(
                "HAS_SKILL", "bls:occupation:15-1252", "bls:skill:adapt", relation_type="essential"
            )
        ],
    )
    onet = GraphSuite(
        [
            _occ("onet:occupation:15-1252.00", "Software Developers"),
            _skill("onet:skill:py", "Python"),
        ],
        [
            _edge(
                "HAS_SKILL",
                "onet:occupation:15-1252.00",
                "onet:skill:py",
                relation_type="essential",
            )
        ],
    )
    lone = GraphSuite([_occ("sfia:skill:x", "Other")], [])
    return SuiteRegistry(
        {
            "bls": suite_factory("bls", bls),
            "onet": suite_factory("onet", onet),
            "sfia": suite_factory("sfia", lone),
        },
        default="bls",
    )


def test_all_suites_merge_with_cites_suite_and_a_crosswalk_link() -> None:
    data = map_all(_registry(), "Software Developers", breadth=3).to_dict()

    assert data["policy"] == "map-rank-v1"
    assert [(n["id"], n["suite"]) for n in data["nodes"]][:2] == [  # type: ignore[index]
        ("bls:occupation:15-1252", "bls"),
        ("bls:skill:adapt", "bls"),
    ]
    assert all(n["cite"].startswith("https://") for n in data["nodes"])  # type: ignore[attr-defined]
    link = next(e for e in data["edges"] if e["graph_edge"] == "CROSSWALK")  # type: ignore[attr-defined]
    assert (link["from_id"], link["to_id"], link["strength"]) == (
        "bls:occupation:15-1252",
        "onet:occupation:15-1252.00",
        "soc",
    )
    confirmed = {n["id"]: n["confirmed_by"] for n in data["nodes"] if "confirmed_by" in n}  # type: ignore[attr-defined]
    assert confirmed == {
        "bls:occupation:15-1252": ["onet"],
        "onet:occupation:15-1252.00": ["bls"],
    }
    assert data["suites"]["sfia"]["status"] == "not_found"  # type: ignore[index]
    summary = data["summary"]
    assert summary == {  # type: ignore[comparison-overlap]
        "suites_mapped": 2,
        "nodes": 4,
        "edges": 3,
        "crosswalk_links": 1,
        "confirmed_nodes": 2,
    }
    ids = {n["id"] for n in data["nodes"]}  # type: ignore[attr-defined]
    assert all(e["from_id"] in ids and e["to_id"] in ids for e in data["edges"])  # type: ignore[attr-defined]


def test_same_input_gives_identical_json() -> None:
    assert (
        map_all(_registry(), "Software Developers").to_dict()
        == map_all(_registry(), "Software Developers").to_dict()
    )


def test_a_subset_maps_only_those_suites() -> None:
    data = map_all(_registry(), "Software Developers", suites=["onet"]).to_dict()
    assert list(data["suites"]) == ["onet"]  # type: ignore[call-overload]
    assert data["summary"]["crosswalk_links"] == 0  # type: ignore[index]


def test_unknown_suite_is_refused() -> None:
    with pytest.raises(UnknownSuiteError):
        map_all(_registry(), "x", suites=["nope"])


def test_ambiguous_in_one_suite_while_another_maps() -> None:
    reg = _registry()
    twin = GraphSuite(
        [_occ("sfia:skill:a", "Developer A"), _occ("sfia:skill:b", "Developer B")], []
    )
    both = SuiteRegistry(
        {"bls": lambda: reg.open("bls"), "sfia": suite_factory("sfia", twin)},
        default="bls",
    )
    data = map_all(both, "Developer", breadth=2).to_dict()
    suites = data["suites"]
    assert suites["sfia"]["status"] == "ambiguous"  # type: ignore[index]
    assert [c["id"] for c in suites["sfia"]["candidates"]] == ["sfia:skill:a", "sfia:skill:b"]  # type: ignore[index]
    assert suites["bls"]["status"] == "mapped"  # type: ignore[index]
    assert data["warnings"] == []


def test_crosswalk_seeds_a_suite_that_cannot_resolve_the_text() -> None:
    reg = _registry(bls_props={"onet_soc_codes": ["15-1252.00"]})
    onet = GraphSuite(
        [_occ("onet:occupation:15-1252.00", "Totally different title")],
        [],
    )
    seeded = SuiteRegistry(
        {"bls": lambda: reg.open("bls"), "onet": suite_factory("onet", onet)}, default="bls"
    )
    data = map_all(seeded, "Software Developers").to_dict()
    entry = data["suites"]["onet"]  # type: ignore[index]
    assert (entry["status"], entry["via"]) == ("via_crosswalk", "crosswalk:bls-onet-soc")
    assert entry["start"]["id"] == "onet:occupation:15-1252.00"
    assert data["summary"]["crosswalk_links"] == 1  # type: ignore[index]


def test_a_failing_suite_is_an_error_entry_and_the_rest_answer() -> None:
    data = map_all(fake_registry(), "software developer").to_dict()
    statuses = {k: v["status"] for k, v in data["suites"].items()}  # type: ignore[attr-defined]
    assert statuses == {
        "fake": "mapped",
        "locate_only": "not_found",
        "broken": "error",
        "down": "error",
    }
    assert data["suites"]["broken"]["warnings"] == ["error:RuntimeError"]  # type: ignore[index]


class _RouteSuite(FakeSuite):
    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        if text == "python":
            from tests.fakes.taxonomy import FakeCandidate

            return FakeToolResult(
                candidates=[FakeCandidate(node=PYTHON, confidence=0.95, method="exact_pref")],
                nodes=[PYTHON],
            )
        return super().search_nodes(text, kind)

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(nodes=[DEV, PYTHON], edges=[HAS_PYTHON])


def test_route_mode_maps_the_top_route_per_suite_with_score_and_reasons() -> None:
    reg = SuiteRegistry(
        {"fake": suite_factory("fake", _RouteSuite()), "bls": suite_factory("bls", FakeSuite())},
        default="fake",
    )
    data = map_all_route(reg, "software developer", "python").to_dict()
    entry = data["suites"]["fake"]  # type: ignore[index]
    assert entry["status"] == "mapped"
    assert entry["route"]["node_ids"] == [DEV.id, PYTHON.id]
    assert isinstance(entry["route"]["score"], float) and entry["route"]["reasons"]
    assert data["suites"]["bls"]["status"] == "not_found"  # type: ignore[index]
    assert [n["on_route"] for n in data["nodes"]] == [True, True]  # type: ignore[attr-defined]
    assert data["query"] == "software developer -> python"


def test_an_id_is_walked_only_in_its_own_taxonomy() -> None:
    # Live: an ESCO skill id with no suite was searched in O*NET as text ("dESCO ESC").
    data = map_all(_registry(), "bls:occupation:15-1252").to_dict()
    suites = data["suites"]
    assert suites["bls"]["status"] == "mapped"  # type: ignore[index]
    assert suites["sfia"]["status"] == "not_searched"  # type: ignore[index]
    # O*NET is not searched with the text, but may start from the official SOC crosswalk.
    assert suites["onet"]["status"] == "via_crosswalk"  # type: ignore[index]


def test_max_nodes_is_a_hard_cap_across_taxonomies() -> None:
    data = map_all(_registry(), "Software Developers", max_nodes=2).to_dict()
    assert len(data["nodes"]) <= 2  # type: ignore[arg-type]
