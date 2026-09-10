"""Interactive `ta-agent` command — Rich loop around the session kernel."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from functools import partial

from rich.console import Console

from talent_angels.assistant import TurnOutcome, run_turn
from talent_angels.env import load_local_dotenv
from talent_angels.llm import LLMClient
from talent_angels.llm.factory import get_llm_client
from talent_angels.query_details import write_query_details
from talent_angels.runlog import append_record
from talent_angels.session.catalog import render_catalogue
from talent_angels.session.copy import WELCOME
from talent_angels.session.credentials import (
    CredentialError,
    looks_like_openrouter_key,
    store,
    verify,
)
from talent_angels.session.kernel import handle_line
from talent_angels.session.router import route_line
from talent_angels.session.store import new_session, sessions_dir
from talent_angels.session.switch import SwitchError, apply, current_choice, load_catalogue, resolve
from talent_angels.suites import SuiteRegistry, default_suite_registry
from talent_angels.tui.picker_ui import Option, is_interactive, select
from talent_angels.tui.render import render_assistant, render_status, render_welcome
from talent_angels.tui.working import Cancelled, run_with_status


def _collect_key(console: Console) -> str:
    """Read the key with echo disabled, verify it, store it. Returns a message.

    The value is never returned to the caller and never enters the transcript
    or the run log, so it cannot be replayed to the model on a later turn.
    """
    console.print(
        "[dim]Paste an OpenRouter key (input hidden). "
        "Get one at https://openrouter.ai/keys — free models need no balance.[/]"
    )
    try:
        key = console.input("[bold cyan]key ›[/] ", password=True).strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return "Cancelled. Nothing was stored."
    if not key:
        return "Nothing entered. Nothing was stored."
    if not looks_like_openrouter_key(key):
        return (
            "That does not look like an OpenRouter key (they start with sk-or-). "
            "Nothing was stored."
        )
    try:
        summary = verify(key)
        path = store(key)
    except CredentialError as exc:
        return f"Not stored: {exc}"
    return f"{summary}. Saved to {path}. Use /model to pick one."


def _start_client(console: Console) -> LLMClient:
    """Build the configured client, or fall back to the stub and say why.

    A missing key raised out of main() before the welcome screen, so the first
    run of a fresh checkout was a traceback with no way forward — and /login,
    the command that fixes it, could never be reached. Start on the stub
    instead: it answers every Locate and Connect question deterministically,
    which is a working product, not an error state.
    """
    try:
        return get_llm_client()
    except ValueError as exc:
        console.print(f"[yellow]Starting without a model:[/] {exc}")
        console.print("[dim]Run /login to store an OpenRouter key, then /model to pick one.[/]")
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


def main(argv: Sequence[str] | None = None, *, registry: SuiteRegistry | None = None) -> int:
    del argv
    load_local_dotenv()
    selected = registry or default_suite_registry()
    console = Console()

    llm_client = _start_client(console)
    state = new_session()
    os.environ["RUNLOG_PATH"] = str(sessions_dir() / state.session_id / "runlog.jsonl")
    attached = " · ".join(
        "O*NET" if name == "onet" else name.upper() for name in selected.available
    )

    def make_runner(outcomes: list[tuple[TurnOutcome, str]]):
        def runner(question, *, bound_node=None, force_capability=None):
            # Structured facts from the assistant; the TUI phrases them (see session.phrase).
            outcome = run_turn(
                registry=selected,
                llm_client=llm_client,
                question=question,
                answer_mode="structured",
                bound_node=bound_node,
                force_capability=force_capability,
                persist=False,
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
        if kind == "command":
            # Commands can change session state, the active client, or the
            # process environment. Keep those effects on this thread so an
            # Esc cancellation cannot leave the UI and active client apart.
            reply = handle_line(
                turn_state,
                line,
                runner=make_runner(turn_outcomes),
                llm_client=llm_client,
            )
        else:
            label = "Looking that up…" if kind == "map" else "Thinking…"
            try:
                reply = run_with_status(
                    console,
                    label,
                    partial(
                        handle_line,
                        turn_state,
                        line,
                        runner=make_runner(turn_outcomes),
                        llm_client=llm_client,
                    ),
                )
            except Cancelled:
                console.print(
                    "[dim]Stopped waiting. The in-flight request may still finish and count "
                    "toward usage.[/]"
                )
                continue
        state = turn_state
        for outcome, question in turn_outcomes:
            append_record(outcome.record)
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
            console.print(_collect_key(console))
            llm_client = get_llm_client()
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
