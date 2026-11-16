"""`ta-agent tag <folder|file>` — the batch tagging entry point.

Two callers reach `main` (PILOT-PLAN.md "Small changes to existing code"):
`tui/app.py` dispatches `["tag", ...]` here instead of opening the chat loop
(a 45s turn deadline would kill a batch run), and the console script's own
`cli.py` does the same for `python -m talent_angels.cli tag ...`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from talent_angels.env import load_local_dotenv
from talent_angels.llm.factory import get_llm_client
from talent_angels.llm.protocol import LLMClient
from talent_angels.suites import SuiteRegistry, UnknownSuiteError, default_suite_registry
from talent_angels.tagging.models import RunOptions, RunSummary
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.runner import WorkbookOpenError, run

#: Batch runs set no `turn_deadline` (PILOT-PLAN.md); a 20s single-call default
#: would strand a slow model mid-document. 60s balances that against noticing
#: a truly stuck provider.
_DEFAULT_TAG_LLM_TIMEOUT_SECONDS = "60"


def _parse_rows(spec: str) -> set[str]:
    """`"66-91,100"` -> `{"66", "67", ..., "91", "100"}`; "D-PR.1" is one id, not a range."""
    rows: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start_text, dash, end_text = part.partition("-")
        if dash and start_text.strip().isdigit() and end_text.strip().isdigit():
            start, end = int(start_text), int(end_text)
            rows.update(str(i) for i in range(start, end + 1))
        else:
            rows.add(part)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ta-agent tag")
    parser.add_argument("path", help="A project folder (with inputs/) or a single input file.")
    parser.add_argument(
        "--no-llm", action="store_true", help="Search only; no model calls (review the top hit)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Tag at most N units.")
    parser.add_argument("--rows", default=None, help='Row ids to tag, e.g. "66-91,100".')
    parser.add_argument(
        "--suites", default=None, help='Comma-separated suite names, e.g. "esco,onet".'
    )
    parser.add_argument("--depth", choices=["leaves", "headings", "all"], default=None)
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Built-in profile name (e.g. 'generic') or a path to a .yaml profile file; "
            "default: tag.yaml's base_profile, else competency_standard."
        ),
    )
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--refresh", action="store_true", help="Re-run non-final verdicts; keep human decisions."
    )
    refresh_group.add_argument("--fresh", action="store_true", help="Ignore the ledger entirely.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent `choose` model calls (search stays sequential).",
    )
    return parser


def _print_summary(summary: RunSummary) -> None:
    if summary.profile:
        print(f"profile={summary.profile}")
    for suite in sorted(summary.counts):
        counts = ", ".join(f"{verdict}={n}" for verdict, n in sorted(summary.counts[suite].items()))
        print(f"{suite}: {counts}")
    print(
        f"units={summary.units} reused={summary.reused} searched={summary.searched} "
        f"seconds={summary.seconds:.1f}"
    )
    if summary.model_calls:
        cost_text = f"${summary.cost_usd:.4f}" if summary.cost_usd is not None else "unknown"
        print(f"model_calls={summary.model_calls} cost_usd={cost_text}")
    if summary.reviews_applied or summary.retried:
        print(f"reviews_applied={summary.reviews_applied} retried={summary.retried}")
    for warning in summary.warnings:
        print(f"warning: {warning}")
    for output in summary.outputs:
        print(output)


class _ProgressView:
    """A live spinner + bar + elapsed time in a terminal; plain lines otherwise.

    Model calls take seconds each, so a step can sit on one number for a while:
    the spinner and the clock keep moving to show the run is alive.
    """

    def __init__(self, *, live: bool) -> None:
        self.live = live
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def __enter__(self) -> _ProgressView:
        if self.live:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=Console(stderr=True),
                # The bar is only for while it runs; the printed lines and the
                # summary stay. A finished run must not leave a stale bar behind.
                transient=True,
            )
            self._progress.start()
            self._task = self._progress.add_task("starting…", total=None)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def message(self, text: str) -> None:
        if self._progress is None or self._task is None:
            print(text, file=sys.stderr)
            return
        self._progress.console.print(text)
        _, _, step = text.partition(": ")
        if step.startswith("step") and text.endswith("…"):
            # A new step without a count yet: a fresh pulsing bar. (A rich task
            # can't go back to "no total", so the last step's would linger.)
            self._progress.remove_task(self._task)
            self._task = self._progress.add_task(step.rstrip("…"), total=None)

    def step(self, label: str, done: int, total: int) -> None:
        if self._progress is None or self._task is None:
            return
        _, _, short = label.partition(": ")
        self._progress.update(self._task, description=short or label, completed=done, total=total)


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: SuiteRegistry | None = None,
    llm_client: LLMClient | None = None,
) -> int:
    load_local_dotenv()
    args = _build_parser().parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"no such path: {path}", file=sys.stderr)
        return 2

    # No `turn_deadline` governs a batch run (PILOT-PLAN.md); without this the
    # provider's own 20s single-call default can cut a slow completion short.
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", _DEFAULT_TAG_LLM_TIMEOUT_SECONDS)

    client = llm_client
    if client is None and not args.no_llm:
        try:
            # Reasoning off: `max_tokens` is fixed at 1024 and reasoning
            # tokens can eat the whole budget, truncating the JSON reply.
            client = get_llm_client(reasoning_enabled=False)
        except Exception as exc:  # noqa: BLE001 - a bad client config must not traceback
            print(f"could not start the model client: {exc} (try --no-llm)", file=sys.stderr)
            return 2

    no_llm = args.no_llm or client is None or getattr(client, "provider", "none") == "none"

    opts = RunOptions(
        no_llm=no_llm,
        limit=args.limit,
        rows=_parse_rows(args.rows) if args.rows else None,
        suites=[name.strip() for name in args.suites.split(",") if name.strip()]
        if args.suites
        else None,
        depth=args.depth,
        refresh=args.refresh,
        fresh=args.fresh,
        workers=args.workers,
        profile=args.profile,
    )

    selected_registry = (
        registry if registry is not None else default_suite_registry(neighbor_cache=False)
    )

    try:
        with _ProgressView(live=sys.stderr.isatty()) as view:
            summary = run(
                path,
                registry=selected_registry,
                opts=opts,
                llm_client=None if no_llm else client,
                progress=view.message,
                step_progress=view.step if view.live else None,
            )
    except (ReaderError, WorkbookOpenError, UnknownSuiteError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a batch run must report, not traceback
        print(f"tag failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
