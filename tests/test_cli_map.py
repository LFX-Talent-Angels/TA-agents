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
    code = main(["map", "Python", "--breadth", "3"], registry=_registry())

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
    code = main(["map", start], registry=_registry())

    assert code == 2
    assert json.loads(capsys.readouterr().out)["warnings"] == [warning]


def test_bad_depth_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit):
        main(["map", "Python", "--depth", "9"], registry=_registry())


def test_ta_agent_routes_map_to_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from talent_angels.tui.app import main as tui_main

    assert tui_main(["map", "Python"], registry=_registry()) == 0
    assert json.loads(capsys.readouterr().out)["suite"] == "test"


def test_route_with_an_unknown_end_exits_two_with_no_route(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.fakes.suite import fake_registry

    code = main(["map", "--route", "software developer", "stonemason"], registry=fake_registry())

    data = json.loads(capsys.readouterr().out)
    assert code == 2
    assert data["warnings"][0] == "no_route"
    assert data["nodes"] == []


def test_map_needs_start_or_route(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["map"], registry=_registry()) == 2
