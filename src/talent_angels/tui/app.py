"""Interactive `ta-agent` command — Rich loop around the session kernel."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from rich.console import Console

from talent_angels.assistant import run_turn
from talent_angels.env import load_local_dotenv
from talent_angels.llm.factory import get_llm_client
from talent_angels.query_details import write_query_details
from talent_angels.session.copy import WELCOME
from talent_angels.session.kernel import handle_line
from talent_angels.session.router import route_line
from talent_angels.session.store import new_session, sessions_dir
from talent_angels.suites import SuiteRegistry, default_suite_registry
from talent_angels.tui.render import render_assistant, render_status, render_welcome


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

    with selected.open() as runtime:
        llm_client = get_llm_client()
        state = new_session()
        os.environ["RUNLOG_PATH"] = str(sessions_dir() / state.session_id / "runlog.jsonl")

        def runner(question, *, bound_node=None, force_capability=None):
            # Structured facts from the assistant; the TUI phrases them (see session.phrase).
            outcome = run_turn(
                suite=runtime.suite,
                suite_name=runtime.name,
                llm_client=llm_client,
                question=question,
                answer_mode="structured",
                bound_node=bound_node,
                force_capability=force_capability,
            )
            write_query_details(outcome, question=question)
            return outcome

        render_welcome(console, WELCOME)
        render_status(
            console,
            suite=runtime.name,
            bound_label=None,
            session_name=state.session_id,
        )

        while True:
            try:
                line = _read_line(console)
            except (EOFError, KeyboardInterrupt):
                console.print()
                return 0
            if not line.strip():
                continue
            if route_line(line).kind == "map":
                with console.status("[cyan]Looking that up…[/]"):
                    reply = handle_line(state, line, runner=runner, llm_client=llm_client)
            else:
                reply = handle_line(state, line, runner=runner, llm_client=llm_client)
            if reply.quit:
                return 0
            render_assistant(console, reply)
            render_status(
                console,
                suite=runtime.name,
                bound_label=reply.bound_label,
                session_name=state.name or state.session_id,
            )


if __name__ == "__main__":
    sys.exit(main())
