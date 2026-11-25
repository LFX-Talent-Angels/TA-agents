"""`ta-agent roadmap` and `ta-agent successors`: JSON out, exit codes, validated arguments."""

from __future__ import annotations

import json

import pytest

from talent_angels.cli import main
from tests.fakes.order_suite import OrderSuite, registry
from tests.test_roadmap import A, B, _esco
from tests.test_successors import X
from tests.test_successors import _esco as _skills


def _run(capsys: pytest.CaptureFixture[str], argv: list[str], reg: object) -> tuple[int, dict]:
    code = main(argv, registry=reg)  # type: ignore[arg-type]
    return code, json.loads(capsys.readouterr().out)


def test_roadmap_prints_json_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code, data = _run(capsys, ["roadmap", A.id, B.id, "--per-step", "2"], registry(esco=_esco()))

    assert code == 0
    assert data["policy"] == "learn-order-v1"
    assert data["per_step"] == 2
    assert data["steps"][0]["items"][0]["label"] == "algebra"


def test_roadmap_exits_two_when_no_taxonomy_resolved_both_ends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, data = _run(capsys, ["roadmap", "zzz", B.id, "--suites", "esco"], registry(esco=_esco()))

    assert code == 2
    assert data["suites"]["esco"]["status"] == "not_found"


def test_an_ambiguous_end_prints_a_pick_list_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["roadmap", "a", B.id], registry=registry(esco=_esco()))

    assert code == 2
    assert "esco: ambiguous; rerun with an id" in capsys.readouterr().err


def test_successors_prints_json_and_unknown_suites_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reg = registry(esco=_skills())
    code, data = _run(capsys, ["successors", X.id, "--depth", "1", "--breadth", "2"], reg)

    assert code == 0
    assert [n["id"] for n in data["suites"]["esco"]["nodes"]] == ["esco:x", "esco:sb", "esco:sa"]

    assert main(["successors", X.id, "--suites", "nope"], registry=reg) == 2
    assert "unknown suite" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["successors", "x", "--depth", "5"],
        ["successors", "x", "--depth", "0"],
        ["successors", "x", "--breadth", "11"],
        ["successors", "x", "--max-nodes", "501"],
        ["roadmap", "a", "b", "--per-step", "0"],
        ["roadmap", "a", "b", "--per-step", "51"],
        ["roadmap", "a"],
    ],
)
def test_bad_arguments_are_argparse_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(argv, registry=registry(esco=OrderSuite([], [])))


def test_ta_agent_routes_both_commands_to_the_cli_one_shot(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from talent_angels.tui.app import main as tui_main

    assert tui_main(["roadmap", A.id, B.id], registry=registry(esco=_esco())) == 0
    assert json.loads(capsys.readouterr().out)["policy"] == "learn-order-v1"
    assert tui_main(["successors", X.id], registry=registry(esco=_skills())) == 0
    assert json.loads(capsys.readouterr().out)["x"] == X.id
