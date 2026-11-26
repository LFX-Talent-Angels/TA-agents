"""Chat slash commands for the batch capabilities: /map /discover /roadmap /successors /tag.

Deterministic wrappers around the same core functions the CLI calls: no model call, no new
logic. Each returns a short summary and writes the full result under ``<TA home>/outputs/``.
Syntax: a route is ``A | B`` (``/map --route A | B``, ``/roadmap A | B``); flags mirror the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from talent_angels.engine import CAPABILITIES, Call, run
from talent_angels.llm.protocol import check_cancelled
from talent_angels.memory.paths import outputs_dir
from talent_angels.report.pages import map_page, plain, roadmap_page, successors_page
from talent_angels.session.commands import UnknownCommand, parse_command
from talent_angels.skills.connect.discover import (
    DEFAULT_DISCOVER_MAX_NODES,
    MAX_DISCOVER_NODES,
    discover,
)
from talent_angels.skills.connect.discover_io import read_items
from talent_angels.skills.connect.discover_report import discover_summary, write_outputs
from talent_angels.skills.connect.neighbourhood import (
    DEFAULT_MAX_NODES,
    MAX_BREADTH,
    MAX_DEPTH,
)
from talent_angels.skills.roadmap import DEFAULT_MAX_NODES as SUCCESSORS_MAX_NODES
from talent_angels.skills.roadmap import MAX_PER_STEP
from talent_angels.suites import SuiteRegistry, UnknownSuiteError

# From the engine registry: a capability with a "slash" alias is a command.
SLASH_COMMANDS = tuple(c.aliases["slash"] for c in CAPABILITIES.values() if "slash" in c.aliases)
TOP = 5  # items named in a chat summary; the rest is in the files

_log = logging.getLogger(__name__)


class _Bad(Exception):
    """Bad input: shown as one line."""


class _Failed(Exception):
    """An unexpected engine error: the chat gets the type only (the details are in the log)."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _Bad(message)


class _TagParser(_Parser):
    """The tag runner's parser with no ``-h``: help would print on the terminal and exit."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog=prog, add_help=False)


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _within(low: int, high: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        if not text.lstrip("-").isdigit() or not low <= int(text) <= high:
            raise argparse.ArgumentTypeError(f"must be {low}..{high}, got {text}")
        return int(text)

    return parse


def _parser(name: str, *, words: bool) -> _Parser:
    p = _Parser(prog=f"/{name}", add_help=False)
    if words:
        p.add_argument("words", nargs="*")
    return p


def _walk_flags(p: _Parser, *, depth: int, breadth: int, max_nodes: int, cap: int) -> None:
    p.add_argument("--suites", type=_csv, default=None)
    p.add_argument("--depth", type=_within(1, MAX_DEPTH), default=depth)
    p.add_argument("--breadth", type=_within(1, MAX_BREADTH), default=breadth)
    p.add_argument("--max-nodes", type=_within(1, cap), default=max_nodes,
                   metavar=f"1..{cap}")  # fmt: skip


def _tokens(argument: str | None) -> list[str]:
    try:
        return shlex.split(argument or "")
    except ValueError:  # an apostrophe in a name, not a quote
        return (argument or "").split()


def _out(command: str, argument: str | None, label: str | None = None) -> Path:
    """A folder named from the command line: readable slug, hash of the whole argument."""
    key = argument or ""
    slug = re.sub(r"[^a-z0-9]+", "-", (label or key).casefold()).strip("-")[:40] or "x"
    return outputs_dir() / f"{command}-{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def _save(folder: Path, stem: str, data: dict[str, Any], page: str) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.html``: after one last look at Esc, never before it."""
    check_cancelled()  # Esc while the result was being computed: write nothing
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html = folder / f"{stem}.html"
    html.write_text(page, encoding="utf-8")
    return path, html


def _engine_output(name: str, payload: dict[str, Any], registry: SuiteRegistry) -> dict[str, Any]:
    """The capability's own dict via `engine.run` (the same bytes the CLI and HTTP give)."""
    result = run(name, payload, registry=registry)
    if result.status == "error":
        raise _Failed(result.error)
    if result.output is None:  # invalid_input, unknown_suite: one line, never a traceback
        raise _Bad(result.error or result.status)
    return result.output


def _names(words: list[str], what: str, *, pair: bool) -> list[str]:
    parts = [p.strip() for p in " ".join(words).split("|")]
    if len(parts) != (2 if pair else 1) or not all(parts):
        raise _Bad(f"give {what}" + (" as: FROM | TO" if pair else ""))
    return parts


def _pick_hints(suites: dict[str, Any], limit: int = 3) -> list[str]:
    return [
        f"{name}: ambiguous, rerun with an id: "
        + "; ".join(f"{c['label']} = {c['id']}" for c in part["candidates"][:limit])
        for name, part in suites.items()
        if part.get("status") == "ambiguous" and part.get("candidates")
    ]


def _shown(labels: list[str]) -> str:
    more = f" (+{len(labels) - TOP} more)" if len(labels) > TOP else ""
    return ", ".join(labels[:TOP]) + more


def _show_map(data: dict[str, Any], folder: Path) -> list[str]:
    path, html = _save(folder, "map", data, map_page(data))
    s = data["summary"]
    lines = [
        f'Map for "{data["query"]}": {s["nodes"]} nodes, {s["edges"]} edges in '
        f"{s['suites_mapped']} taxonomies, {s['crosswalk_links']} crosswalk links.",
        "Per taxonomy: "
        + ", ".join(f"{n} {plain(p['status'], n)}" for n, p in data["suites"].items()),
    ]
    nodes = [str(n["label"]) for n in data["nodes"]]
    if nodes:
        lines.append("Top: " + _shown(nodes))
    return [*lines, *_pick_hints(data["suites"]), f"json: {path}", f"html: {html}"]


def _map(argument: str | None, registry: SuiteRegistry) -> list[str]:
    p = _parser("map", words=True)
    p.add_argument("--route", action="store_true")
    p.add_argument("--kinds", type=_csv, default=None)
    _walk_flags(p, depth=1, breadth=0, max_nodes=DEFAULT_MAX_NODES, cap=DEFAULT_MAX_NODES)
    a = p.parse_intermixed_args(_tokens(argument))
    names = _names(a.words, "a name or id" if not a.route else "two names", pair=a.route)
    walk = {
        "suites": a.suites,
        "depth": a.depth,
        "breadth": a.breadth or None,  # the engine's default: 4, or 3 on a route
        "kinds": a.kinds,
        "max_nodes": a.max_nodes,
    }
    ends = {"route_from": names[0], "route_to": names[1]} if a.route else {"start": names[0]}
    data = _engine_output("map", {**ends, **walk}, registry)
    return _show_map(data, _out("map", argument))


def _show_discover(data: dict[str, Any], folder: Path) -> list[str]:
    check_cancelled()  # Esc during the last step still writes nothing
    paths = write_outputs(data, folder)
    down = [f"{n}: {st['status']}" for n, st in data["suites"].items() if st["status"] != "ok"]
    return [*discover_summary(data).split("\n"), *down, *(f"{k}: {v}" for k, v in paths.items())]


def _discover(argument: str | None, registry: SuiteRegistry) -> list[str]:
    p = _parser("discover", words=False)
    p.add_argument("file")
    p.add_argument("--kinds", type=_csv, default=None)
    p.add_argument("--nodes", default=None)
    _walk_flags(p, depth=2, breadth=4, max_nodes=DEFAULT_DISCOVER_MAX_NODES, cap=MAX_DISCOVER_NODES)
    a = p.parse_args(_tokens(argument))
    items, notes = read_items(Path(a.file).expanduser())
    recorded = None
    if a.nodes:
        loaded = json.loads(Path(a.nodes).expanduser().read_text(encoding="utf-8"))
        recorded = loaded.get("resolved") if isinstance(loaded, dict) else loaded
        if not isinstance(recorded, list):
            raise _Bad("--nodes needs a discovery.json or a list of resolved entries")
    data = discover(
        registry, items, suites=a.suites, depth=a.depth, breadth=a.breadth, kinds=a.kinds,
        max_nodes=a.max_nodes, recorded=recorded, notes=notes,
    )  # fmt: skip
    return _show_discover(data, _out("discover", argument, Path(a.file).stem))


def _show_roadmap(data: dict[str, Any], folder: Path, start: str, target: str) -> list[str]:
    path, html = _save(folder, "roadmap", data, roadmap_page(data))
    lines = [
        f'Learning order from "{start}" to "{target}" ({data["policy"]}): '
        + ", ".join(f"{n} {plain(s['status'], n)}" for n, s in data["suites"].items())
    ]
    for step in data["steps"]:
        labels = [i["label"] for i in step["items"]]
        lines.append(f"{step['step']}. {step['name']}: {_shown(labels)}")
    return [*lines, *_pick_hints(data["suites"]), f"json: {path}", f"html: {html}"]


def _roadmap(argument: str | None, registry: SuiteRegistry) -> list[str]:
    p = _parser("roadmap", words=True)
    p.add_argument("--suites", type=_csv, default=None)
    p.add_argument("--per-step", type=_within(1, MAX_PER_STEP), default=10,
                   metavar=f"1..{MAX_PER_STEP}")  # fmt: skip
    a = p.parse_intermixed_args(_tokens(argument))
    start, target = _names(a.words, "two names", pair=True)
    data = _engine_output(
        "roadmap",
        {"from": start, "to": target, "suites": a.suites, "per_step": a.per_step},
        registry,
    )
    return _show_roadmap(data, _out("roadmap", argument), start, target)


def _show_successors(data: dict[str, Any], folder: Path, start: str) -> list[str]:
    path, html = _save(folder, "successors", data, successors_page(data))
    lines = [f'What comes after "{start}" ({data["policy"]}):']
    for name, part in data["suites"].items():
        if part.get("status") == "ambiguous" and part.get("candidates"):
            continue  # listed once below, with the ids to rerun with
        labels = [n["label"] for n in part.get("nodes", [])[1:]]
        lines.append(
            f"{name}: {plain(part['status'], name)}" + (f", {_shown(labels)}" if labels else "")
        )
    return [*lines, *_pick_hints(data["suites"]), f"json: {path}", f"html: {html}"]


def _successors(argument: str | None, registry: SuiteRegistry) -> list[str]:
    p = _parser("successors", words=True)
    _walk_flags(p, depth=2, breadth=4, max_nodes=SUCCESSORS_MAX_NODES, cap=DEFAULT_MAX_NODES)
    a = p.parse_intermixed_args(_tokens(argument))
    (start,) = _names(a.words, "a skill name or id", pair=False)
    walk = {"suites": a.suites, "depth": a.depth, "breadth": a.breadth, "max_nodes": a.max_nodes}
    data = _engine_output("successors", {"skill": start, **walk}, registry)
    return _show_successors(data, _out("successors", argument), start)


def _tag(argument: str | None, registry: SuiteRegistry) -> list[str]:
    from talent_angels.tagging.cli import main as tag_main

    out: list[str] = []
    err: list[str] = []
    # The runner reports through these callbacks: no swap of the process-wide stdout, which
    # the terminal keeps drawing on while this runs on a worker thread.
    code = tag_main(
        _tokens(argument), registry=registry, out=out.append, err=err.append,
        parser_class=_TagParser,
    )  # fmt: skip
    if code:
        lines = [ln for ln in err if ln.strip()]
        raise _Bad((lines[-1] if lines else f"tag exited with {code}").strip())
    return [ln for ln in out if ln.strip()]


_RUNNERS: dict[str, Callable[[str | None, SuiteRegistry], list[str]]] = {
    "map": _map,
    "discover": _discover,
    "roadmap": _roadmap,
    "successors": _successors,
    "tag": _tag,
}

_PRESENT: dict[str, Callable[[dict[str, Any], Path, dict[str, Any]], list[str]]] = {
    "map": lambda d, f, _i: _show_map(d, f),
    "roadmap": lambda d, f, i: _show_roadmap(d, f, str(i["from"]), str(i["to"])),
    "successors": lambda d, f, i: _show_successors(d, f, str(i["skill"])),
    "discover": lambda d, f, _i: _show_discover(d, f),
}


def _write_json(folder: Path, filename: str, payload: dict[str, Any]) -> Path:
    check_cancelled()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _argument(call: Call, file: str | None) -> str:
    inp = call.input
    suites = inp.get("suites")
    if file is not None:
        parts = [file, *(["--suites", ",".join(suites)] if suites else [])]
        return shlex.join(parts)
    parts = [str(inp["path"])]
    if inp.get("no_llm"):
        parts.append("--no-llm")
    if inp.get("limit"):
        parts.extend(["--limit", str(inp["limit"])])
    if suites:
        parts.extend(["--suites", ",".join(suites)])
    return shlex.join(parts)


def _label(inp: dict[str, Any]) -> str:
    for key in ("start", "route_from", "from", "skill"):
        if inp.get(key):
            return str(inp[key])
    items = inp.get("items") or []
    if items and isinstance(items[0], dict) and items[0].get("item"):
        return str(items[0]["item"])
    return "x"


def echo(call: Call, *, by: str, file: str | None = None) -> str:
    """Code-written echo line telling the user exactly what capability runs and with what args."""
    spec = CAPABILITIES[call.capability]
    inp = call.input
    if file is not None:
        suites = inp.get("suites")
        arg_text = f"on file={file}" + (f", suites={','.join(suites)}" if suites else "")
    else:
        shown: list[str] = []
        for field_name, info in spec.input_model.model_fields.items():
            alias = info.alias or field_name
            val = inp.get(alias)
            if val is None and not info.is_required():
                continue
            if not info.is_required() and val == info.get_default(call_default_factory=True):
                continue
            if call.capability == "tag" and alias == "no_llm":
                continue
            if alias == "items" and isinstance(val, list):
                labels = [
                    str(it["item"]) for it in val[:3] if isinstance(it, dict) and "item" in it
                ]
                more = f" (+{len(val) - 3} more)" if len(val) > 3 else ""
                shown.append(f"items={', '.join(labels)}{more}")
            elif isinstance(val, list):
                shown.append(f"{alias}={','.join(str(x) for x in val)}")
            else:
                shown.append(f"{alias}={val}")
        arg_text = f"with {', '.join(shown)}" if shown else ""

    if call.capability == "tag" and inp.get("no_llm") is False:
        tail = "(uses your model; the cost is reported when it ends)."
    elif spec.policy:
        tail = f"({spec.policy}; deterministic, no model call)."
    else:
        tail = "(deterministic, no model call)."

    line = f"Running **{call.capability}**" + (f" {arg_text}" if arg_text else "") + f" {tail}"
    if by == "model":
        line += " Your model picked the capability; it wrote none of the result."
    return line


def _guarded(name: str, fn: Callable[[], list[str]]) -> str:
    try:
        return "\n".join(f"- {ln}" if i else ln for i, ln in enumerate(fn()))
    except _Bad as exc:
        return f"/{name}: {exc}"
    except _Failed as exc:
        return f"/{name} failed ({exc}); the details are in the log."
    except (OSError, ValueError, UnknownSuiteError) as exc:
        return f"/{name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - a long batch must report, not traceback
        # The message may carry a provider reply or a key: the chat gets the type only.
        _log.warning("/%s failed", name, exc_info=True)
        return f"/{name} failed ({type(exc).__name__}); the details are in the log."


def run_call(
    call: Call, registry: SuiteRegistry, *, by: str = "keyword", file: str | None = None
) -> str:
    """Say what runs, run it deterministically, summarise in code, save call.json."""
    head = echo(call, by=by, file=file)
    return head + "\n" + _guarded(call.capability, lambda: _run_call(call, registry, by, file))


def _run_call(call: Call, registry: SuiteRegistry, by: str, file: str | None) -> list[str]:
    name, inp = call.capability, call.input
    if file is not None:
        # Save the items that were read, so call.json alone replays the run.
        try:
            items, _ = read_items(Path(file).expanduser())
        except (OSError, ValueError):
            pass  # the runner below reports the bad file in one line
        else:
            replay = [
                {"item": i.text, "kind": i.kind, "related_to": list(i.related_to)} for i in items
            ]
            call = call.model_copy(update={"input": {**inp, "items": replay}})
    if file is not None or name == "tag":
        argument = _argument(call, file)
        lines = _RUNNERS[name](argument, registry)
        folder = _out(name, argument, Path(file).stem if file else None)
    else:
        result = run(name, inp, registry=registry)
        if result.output is None:
            raise _Bad(result.error or result.status)
        folder = _out(name, json.dumps(inp, sort_keys=True), _label(inp))
        lines = _PRESENT[name](result.output, folder, inp)
    _write_json(
        folder,
        "call.json",
        {
            **call.model_dump(mode="json"),
            "routed_by": by,
            **({"file": file} if file else {}),
        },
    )
    return lines


def capability_command(text: str) -> tuple[str, str | None] | None:
    """``(name, argument)`` when the line is one of the batch commands, else None."""
    try:
        parsed = parse_command(text)
    except UnknownCommand:
        return None
    return (parsed.name, parsed.argument) if parsed and parsed.name in SLASH_COMMANDS else None


def run_slash(name: str, argument: str | None, registry: SuiteRegistry) -> str:
    """Run one capability command. Bad input is one line, never a traceback."""
    return _guarded(name, lambda: _RUNNERS[name](argument, registry))
