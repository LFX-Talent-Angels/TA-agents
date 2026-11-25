"""`ta-agent capabilities` and `ta-agent run`: the manifest and the one way in, with exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_angels.cli import main
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_roadmap import A, B, _esco


def test_capabilities_lists_one_line_each_and_json_has_the_schemas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["capabilities"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in lines] == [
        "locate", "connect", "pathfind", "map", "discover", "roadmap", "successors", "tag",
    ]  # fmt: skip
    assert (
        "[cli: roadmap, slash: roadmap, mcp: roadmap, http: /v1/capabilities/roadmap]" in lines[5]
    )

    assert main(["capabilities", "--json"]) == 0
    full = json.loads(capsys.readouterr().out)
    assert len(full["capabilities"]) == 8 and full["capabilities"][0]["input_schema"]


def test_run_prints_the_result_and_its_output_is_the_roadmap_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reg = registry(esco=_esco())
    body = json.dumps({"from": A.id, "to": B.id, "per_step": 2})
    assert main(["run", "roadmap", body], registry=reg) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok" and result["call"]["capability"] == "roadmap"

    assert main(["roadmap", A.id, B.id, "--per-step", "2"], registry=reg) == 0
    assert result["output"] == json.loads(capsys.readouterr().out)  # same as `ta-agent roadmap`


def test_run_reads_input_from_a_file_and_takes_an_alias(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    spec = tmp_path / "in.json"
    spec.write_text(json.dumps({"start": "software developer", "suites": ["fake"]}))
    assert main(["run", "map_neighbors", f"@{spec}"], registry=fake_registry()) == 0
    assert json.loads(capsys.readouterr().out)["call"]["capability"] == "map"


@pytest.mark.parametrize(
    ("argv", "code", "status"),
    [
        (["run", "roadmap", '{"from": "a"}'], 2, "invalid_input"),
        (["run", "roadmap", '{"from": "a", "to": "b", "x": 1}'], 2, "invalid_input"),
        (["run", "shell", "{}"], 2, "unknown_capability"),
        (["run", "roadmap", '{"from": "a", "to": "b", "suites": ["nope"]}'], 2, "unknown_suite"),
        (["run", "roadmap", '{"from": "zzz", "to": "b", "suites": ["fake"]}'], 2, "no_result"),
    ],
)
def test_refusals_exit_two_with_the_status(
    capsys: pytest.CaptureFixture[str], argv: list[str], code: int, status: str
) -> None:
    assert main(argv, registry=fake_registry()) == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_unreadable_input_exits_two_and_never_echoes_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["run", "map", "{not json sk-SECRET"], registry=fake_registry()) == 2
    assert main(["run", "map", f"@{tmp_path}/missing-SECRET.json"], registry=fake_registry()) == 2
    assert main(["run", "map", "[1]"], registry=fake_registry()) == 2
    err = capsys.readouterr().err
    assert "SECRET" not in err and "JSONDecodeError" in err


def test_an_unexpected_error_exits_one_with_the_type_only(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import talent_angels.skills.roadmap as skill

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("sk-SECRET")

    monkeypatch.setattr(skill, "roadmap", boom)
    assert main(["run", "roadmap", '{"from": "a", "to": "b"}'], registry=fake_registry()) == 1
    out = capsys.readouterr().out
    assert json.loads(out)["error"] == "RuntimeError" and "SECRET" not in out


def test_the_ta_agent_entry_point_dispatches_both_commands_to_the_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from talent_angels.tui.app import main as tui_main

    assert tui_main(["capabilities"]) == 0
    assert capsys.readouterr().out.startswith("locate")
    assert tui_main(["run", "map", '{"start": "software developer", "suites": ["fake"]}'],
                    registry=fake_registry()) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_run_tag_runs_in_the_foreground_without_a_root(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import shutil

    from tests.test_tagging_run import FIXTURE
    from tests.test_tagging_run import _registry as tag_registry

    (tmp_path / "inputs").mkdir()
    shutil.copy(FIXTURE, tmp_path / "inputs" / "standard.json")
    body = json.dumps({"path": str(tmp_path), "limit": 2})
    assert main(["run", "tag", body], registry=tag_registry()) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok" and result["output"]["units"] > 0
