"""Interactive `ta-agent` command — Rich loop around the session kernel."""

from __future__ import annotations

import contextvars
import os
import sys
import threading
from collections.abc import Sequence
from functools import partial

from rich.console import Console

from talent_angels.assistant import TurnOutcome, run_turn
from talent_angels.assistant.llm_call import PHRASING_STAGES, collect_stages
from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.turn import persist_turn_record, with_extra_stages
from talent_angels.env import load_local_dotenv
from talent_angels.llm import LLMClient
from talent_angels.llm.factory import get_llm_client
from talent_angels.llm.protocol import bind_cancel
from talent_angels.query_details import write_query_details
from talent_angels.session.catalog import render_catalogue
from talent_angels.session.copy import NO_MODEL_NOTICE, WELCOME
from talent_angels.session.kernel import handle_line
from talent_angels.session.models import SessionState
from talent_angels.session.route import Routed
from talent_angels.session.route import route as route_plain
from talent_angels.session.router import route_line
from talent_angels.session.slash import capability_command
from talent_angels.session.store import load_last, new_session, save_session, sessions_dir
from talent_angels.session.switch import SwitchError, apply, current_choice, load_catalogue, resolve
from talent_angels.suites import SuiteRegistry, default_suite_registry
from talent_angels.tui.connect import ask_and_save
from talent_angels.tui.picker_ui import Option, is_interactive, select
from talent_angels.tui.render import render_assistant, render_status, render_welcome
from talent_angels.tui.working import Cancelled, run_with_status


def _writes_files(batch: bool, running: threading.Event) -> bool:
    return batch or running.is_set()


def _start_client(console: Console, *, first_run: bool = False) -> LLMClient:
    """Build the configured client, asking once on first interactive start if unset."""
    if first_run and "LLM_PROVIDER" not in os.environ and is_interactive():
        console.print(ask_and_save(console, first_run=True))
    try:
        return get_llm_client()
    except ValueError as exc:
        console.print(f"[yellow]Starting without a model:[/] {exc}")
        console.print(f"[dim]{NO_MODEL_NOTICE}[/]")
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["LLM_MODEL"] = "stub"
        return get_llm_client()


def _pick_model(console: Console) -> str:
    """Run the picker over the free catalogue. Returns a message to show."""
    catalogue, note = load_catalogue()
    if note:
        return note
    if not catalogue:
        return "No free models are listed right now."
    current = current_choice().model
    options = [
        Option(key=m.id, label=f"{m.id}{f'  {m.context // 1000}k' if m.context else ''}")
        for m in catalogue
    ]
    options.append(Option(key="none stub deterministic", label="none — deterministic stub"))
    if not is_interactive():
        return render_catalogue(catalogue, current=current)
    chosen = select(console, options)
    if chosen is None:
        return "Kept the current model."
    argument = "none" if chosen.key.startswith("none ") else chosen.key
    try:
        choice = resolve(argument, catalogue)
        client = apply(choice)
    except SwitchError as exc:
        return f"Did not switch: {exc}"
    _pick_model.client = client  # type: ignore[attr-defined]
    return f"Now using {choice.label()}."


def _read_line(console: Console) -> str:
    """Single prompt. Rich Prompt.ask would add a second ': ' suffix."""
    parts: list[str] = []
    while True:
        chunk = console.input("[bold cyan]›[/] ")
        if chunk.endswith("\\"):
            parts.append(chunk[:-1])
            continue
        parts.append(chunk)
        break
    return "\n".join(parts)


def _sync_runlog_path(state: SessionState, previous_key: str) -> str:
    """Keep ``RUNLOG_PATH`` pointing at the session that is actually live.

    ``/reset`` and ``/reset-all`` delete the old session directory and hand back
    a fresh ``session_id``. ``RUNLOG_PATH`` is read from the environment on every
    write, so without this the next turn would recreate the directory just erased
    and log a conversation the user discarded — the erase would undo itself.

    Returns the key to pass in next time.
    """
    current_key = state.name or state.session_id
    if current_key != previous_key:
        os.environ["RUNLOG_PATH"] = str(sessions_dir() / current_key / "runlog.jsonl")
    return current_key


def main(argv: Sequence[str] | None = None, *, registry: SuiteRegistry | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["setup"]:
        from talent_angels.tui.connect import setup_main

        return setup_main(args[1:])

    if args[:1] == ["tag"]:
        # Batch tagging is a one-shot command, not a chat turn: the chat's 45s
        # turn deadline would kill it mid-run, so it never enters the loop
        # below.
        from talent_angels.tagging.cli import main as tag_main

        return tag_main(args[1:], registry=registry)

    if args[:1] in (
        ["map"],
        ["discover"],
        ["roadmap"],
        ["successors"],
        ["capabilities"],
        ["run"],
    ):
        from talent_angels.cli import main as cli_main

        return cli_main(args, registry=registry)

    load_local_dotenv()
    selected = registry or default_suite_registry()
    console = Console()

    llm_client = _start_client(console, first_run=True)
    try:
        state = load_last()
    except Exception:
        state = new_session()
    os.environ["RUNLOG_PATH"] = str(sessions_dir() / state.session_id / "runlog.jsonl")
    runlog_key = state.name or state.session_id
    attached = " · ".join(suite_heading(name) for name in selected.available)

    def make_runner(outcomes: list[tuple[TurnOutcome, str]]):
        def runner(
            question,
            *,
            bound_node=None,
            bound_nodes=None,
            force_capability=None,
            area=None,
            suite=None,
        ):
            # Structured facts from the assistant; the TUI phrases them (see session.phrase).
            outcome = run_turn(
                registry=selected,
                llm_client=llm_client,
                question=question,
                answer_mode="structured",
                bound_node=bound_node,
                bound_nodes=bound_nodes,
                force_capability=force_capability,
                persist=False,
                area=area,
                suite_override=suite,
            )
            outcomes.append((outcome, question))
            return outcome

        return runner

    render_welcome(console, WELCOME)
    render_status(
        console,
        suite=attached,
        bound_label=None,
        session_name=state.session_id,
        model_label=current_choice().label(),
    )

    while True:
        try:
            line = _read_line(console)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not line.strip():
            continue
        kind = route_line(line).kind
        turn_outcomes: list[tuple[TurnOutcome, str]] = []
        turn_state = state.model_copy(deep=True)
        # Every metered LLM call made while handling this line, including the
        # phrasing that happens after run_turn built its record. The context is
        # copied into the worker thread so the collector sees its calls.
        stage_scope = collect_stages()
        line_stages = stage_scope.__enter__()
        line_context = contextvars.copy_context()
        # Esc sets this; the abandoned turn sees it at its next model call.
        cancel = threading.Event()
        line_context.run(bind_cancel, cancel)
        batch = capability_command(line) is not None
        # Set once a plain-language line starts a capability (it can write files).
        running = threading.Event()
        if kind == "command" and not batch:
            # Commands can change session state, the active client, or the
            # process environment. Keep those effects on this thread so an
            # Esc cancellation cannot leave the UI and active client apart.
            reply = handle_line(
                turn_state,
                line,
                runner=make_runner(turn_outcomes),
                llm_client=llm_client,
                registry=selected,
            )
        else:
            peek = route_plain(line, available=selected.available) if kind == "map" else None
            if isinstance(peek, Routed):
                label = f"Running {peek.call.capability}…"
            else:
                label = "Looking that up…" if kind == "map" else "Thinking…"
            if batch:
                # /map /discover /roadmap /successors /tag change nothing but the copy of the
                # state and their own files, so they run behind the same Esc watcher; they stop
                # at the next item and write nothing. Esc waits for them to stop (they are not
                # abandoned), so "Stopped" is never printed while a file can still be written.
                label = "Running"
                console.print("[dim]Running… this can take a while; esc stops it.[/]")
            try:
                reply = run_with_status(
                    console,
                    label,
                    partial(
                        line_context.run,
                        handle_line,
                        turn_state,
                        line,
                        runner=make_runner(turn_outcomes),
                        llm_client=llm_client,
                        registry=selected,
                        on_capability=running.set,
                    ),
                    cancel=cancel,
                    wait_on_cancel=partial(_writes_files, batch, running),
                )
            except Cancelled:
                stage_scope.__exit__(None, None, None)
                console.print(
                    "[dim]Stopped waiting. The in-flight request may still finish and count "
                    "toward usage.[/]"
                )
                continue
        state = turn_state
        save_session(state)
        runlog_key = _sync_runlog_path(state, runlog_key)
        stage_scope.__exit__(None, None, None)
        phrasing = [stage for stage in line_stages if stage.stage in PHRASING_STAGES]
        for index, (outcome, question) in enumerate(turn_outcomes):
            # run_turn(persist=False) above skips its own persistence so a
            # cancelled (Esc'd) turn is never written; a completed one is
            # persisted here through the same guarded path the API uses
            # (run-log, episode, vector recall index). Phrasing calls are billed
            # to the line's last turn so the run-log totals are complete.
            if index == len(turn_outcomes) - 1:
                outcome.record = with_extra_stages(outcome.record, phrasing)
            persist_turn_record(outcome.record)
            write_query_details(outcome, question=question)
        if reply.new_llm_client is not None:
            llm_client = reply.new_llm_client
        if reply.request_model_pick:
            console.print(_pick_model(console))
            picked = getattr(_pick_model, "client", None)
            if picked is not None:
                llm_client = picked
                del _pick_model.client  # type: ignore[attr-defined]
            render_status(
                console,
                suite=attached,
                bound_label=state.binding.node.pref_label if state.binding else None,
                session_name=state.name or state.session_id,
                model_label=current_choice().label(),
            )
            continue
        if reply.request_key:
            console.print(ask_and_save(console))
            llm_client = _start_client(console)
            render_status(
                console,
                suite=attached,
                bound_label=state.binding.node.pref_label if state.binding else None,
                session_name=state.name or state.session_id,
                model_label=current_choice().label(),
            )
            continue
        if reply.quit:
            return 0
        render_assistant(console, reply)
        render_status(
            console,
            suite=attached,
            bound_label=reply.bound_label,
            session_name=state.name or state.session_id,
            model_label=current_choice().label(),
        )


if __name__ == "__main__":
    sys.exit(main())
