"""`ta-agent tag <folder|file>` — the batch tagging entry point.

Two callers reach `main` (PILOT-PLAN.md "Small changes to existing code"):
`tui/app.py` dispatches `["tag", ...]` here instead of opening the chat loop
(a 45s turn deadline would kill a batch run), and the console script's own
`cli.py` does the same for `python -m talent_angels.cli tag ...`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from talent_angels.env import load_local_dotenv
from talent_angels.suites import SuiteRegistry, UnknownSuiteError, default_suite_registry
from talent_angels.tagging.models import RunOptions, RunSummary
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.runner import WorkbookOpenError, run


def _parse_rows(spec: str) -> set[str]:
    """`"66-91,100"` -> `{"66", "67", ..., "91", "100"}`."""
    rows: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, _, end_text = part.partition("-")
            start, end = int(start_text), int(end_text)
            rows.update(str(i) for i in range(start, end + 1))
        else:
            rows.add(part)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ta-agent tag")
    parser.add_argument("path", help="A project folder (with inputs/) or a single input file.")
    parser.add_argument(
        "--no-llm", action="store_true", help="Search only; no model calls (the only mode in PR2)."
    )
    parser.add_argument("--limit", type=int, default=None, help="Tag at most N units.")
    parser.add_argument("--rows", default=None, help='Row ids to tag, e.g. "66-91,100".')
    parser.add_argument(
        "--suites", default=None, help='Comma-separated suite names, e.g. "esco,onet".'
    )
    parser.add_argument("--depth", choices=["leaves", "headings", "all"], default=None)
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--refresh", action="store_true", help="Re-run non-final verdicts; keep human decisions."
    )
    refresh_group.add_argument("--fresh", action="store_true", help="Ignore the ledger entirely.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Accepted; model steps use it in PR3."
    )
    return parser


def _print_summary(summary: RunSummary) -> None:
    for suite in sorted(summary.counts):
        counts = ", ".join(f"{verdict}={n}" for verdict, n in sorted(summary.counts[suite].items()))
        print(f"{suite}: {counts}")
    print(
        f"units={summary.units} reused={summary.reused} searched={summary.searched} "
        f"seconds={summary.seconds:.1f}"
    )
    for warning in summary.warnings:
        print(f"warning: {warning}")
    for output in summary.outputs:
        print(output)


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: SuiteRegistry | None = None,
    llm_client: object | None = None,
) -> int:
    load_local_dotenv()
    args = _build_parser().parse_args(argv)

    if not args.no_llm:
        print("model steps are not available yet; running search only (--no-llm)")

    path = Path(args.path)
    if not path.exists():
        print(f"no such path: {path}", file=sys.stderr)
        return 2

    opts = RunOptions(
        no_llm=True,
        limit=args.limit,
        rows=_parse_rows(args.rows) if args.rows else None,
        suites=[name.strip() for name in args.suites.split(",") if name.strip()]
        if args.suites
        else None,
        depth=args.depth,
        refresh=args.refresh,
        fresh=args.fresh,
        workers=args.workers,
    )

    selected_registry = (
        registry if registry is not None else default_suite_registry(neighbor_cache=False)
    )

    def progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        summary = run(
            path, registry=selected_registry, opts=opts, llm_client=llm_client, progress=progress
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
