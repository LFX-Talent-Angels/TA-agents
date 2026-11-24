"""`ta-agent map` prints the Graph-Mapper JSON and signals an unresolved start by exit code."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from talent_angels.cli import main
from talent_angels.suites import SuiteRegistry, SuiteRuntime
from tests.test_graph_mapper import GraphSuite, _edge, _node


def _registry() -> SuiteRegistry:
    suite = GraphSuite(
        [
            _node("test:py", label="Python"),
            _node("test:py2", label="Python snake"),
            _node("test:rel", label="scripting"),
        ],
        [_edge("RELATED_TO", "test:py", "test:rel", relation_type="optional")],
    )

    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=suite, health_check=lambda: True)  # type: ignore[arg-type]

    return SuiteRegistry({"test": factory}, default="test")


def test_map_prints_json_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["map", "Python", "--suite", "test", "--breadth", "3"], registry=_registry())

    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["start"] == {"id": "test:py", "label": "Python"}
    assert [n["label"] for n in data["nodes"]] == ["Python", "scripting"]


@pytest.mark.parametrize(
    ("start", "warning"), [("pyth", "ambiguous_start"), ("zzz", "start_not_found:zzz")]
)
def test_unresolved_start_exits_two_with_json(
    capsys: pytest.CaptureFixture[str], start: str, warning: str
) -> None:
    code = main(["map", start, "--suite", "test"], registry=_registry())

    assert code == 2
    assert json.loads(capsys.readouterr().out)["warnings"] == [warning]


def test_bad_depth_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit):
        main(["map", "Python", "--depth", "9"], registry=_registry())


def test_ta_agent_routes_map_to_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from talent_angels.tui.app import main as tui_main

    assert tui_main(["map", "Python", "--suite", "test"], registry=_registry()) == 0
    assert json.loads(capsys.readouterr().out)["suite"] == "test"


def test_route_with_an_unknown_end_exits_two_with_no_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.fakes.suite import fake_registry

    code = main(
        ["map", "--route", "software developer", "stonemason", "--suite", "fake"],
        registry=fake_registry(),
    )

    data = json.loads(capsys.readouterr().out)
    assert code == 2
    assert data["warnings"][0] == "no_route"
    assert data["nodes"] == []


def test_map_needs_start_or_route(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["map"], registry=_registry()) == 2


def _two_suites(onet_id: str = "onet:occupation:15-1252.00") -> SuiteRegistry:
    onet = GraphSuite(
        [
            _node(onet_id, kind="Occupation", label="Software Developers"),
            _node("onet:skill:py", label="Python"),
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
    bls = GraphSuite(
        [
            _node("bls:occupation:15-1252", kind="Occupation", label="Software Developers"),
            _node("bls:skill:a", label="Adaptability"),
            _node("bls:occupation:29-1", kind="Occupation", label="Software nurse A"),
            _node("bls:occupation:29-2", kind="Occupation", label="Software nurse B"),
        ],
        [_edge("HAS_SKILL", "bls:occupation:15-1252", "bls:skill:a", relation_type="essential")],
    )
    from tests.fakes.suite import suite_factory

    return SuiteRegistry(
        {"onet": suite_factory("onet", onet), "bls": suite_factory("bls", bls)}, default="onet"
    )


def test_map_without_a_suite_merges_every_taxonomy_with_a_crosswalk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["map", "Software Developers"], registry=_two_suites())

    out = capsys.readouterr()
    data = json.loads(out.out)
    assert code == 0 and out.err == ""
    assert data["summary"]["suites_mapped"] == 2 and data["summary"]["crosswalk_links"] == 1


def test_suites_picks_a_subset(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["map", "Software Developers", "--suites", "bls"], registry=_two_suites())

    assert code == 0
    assert list(json.loads(capsys.readouterr().out)["suites"]) == ["bls"]


def test_ambiguous_suite_hints_on_stderr_while_json_stays_clean(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["map", "Software"], registry=_two_suites("onet:occupation:99-9999.00"))

    out = capsys.readouterr()
    assert code == 0
    assert json.loads(out.out)["suites"]["bls"]["status"] == "ambiguous"
    assert out.err.startswith('bls: "Software" is ambiguous')
    assert 'ta-agent map "bls:occupation:15-1252" --suite bls' in out.err


def test_exit_two_when_no_taxonomy_maps(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["map", "zzz"], registry=_two_suites()) == 2
    assert json.loads(capsys.readouterr().out)["warnings"] == ["no_suite_mapped"]


def test_unknown_suites_value_is_exit_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["map", "x", "--suites", "nope"], registry=_two_suites()) == 2
    assert "unknown suite" in capsys.readouterr().err
