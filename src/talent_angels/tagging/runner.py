"""Orchestrates one `ta-agent tag` run: read -> select -> search -> verdict -> write.

Never calls `run_turn`/`interpret_question` (PILOT-PLAN.md "Architecture") —
no plan cache, no episodes, no profile writes, and project memory lives under
`<project>/.ta-tag/`, never `TA_AGENTS_HOME`.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from talent_angels.runlog import RunLogRecord, ToolCall, append_record
from talent_angels.suites import SuiteRegistry, UnknownSuiteError
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.tagging.keys import fingerprint, versions
from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import (
    Document,
    RunOptions,
    RunSummary,
    SearchOutcome,
    TagRecord,
    TagUnit,
    Verdict,
)
from talent_angels.tagging.profile import Profile, load_profile
from talent_angels.tagging.readers import READERS, read_document
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.search import Router, queries_for, search_unit
from talent_angels.tagging.select import document_context, select_units
from talent_angels.tagging.verdict import decide
from talent_angels.tagging.writers.xlsx import write_workbook

#: Not cached — rules 1/3 of tag-v1 (verdict.py) — so a transient outage or a
#: botched model call is retried next run instead of frozen into the ledger.
_NOT_CACHED = (Verdict.ERROR, Verdict.UNAVAILABLE)
_RETRY_ON_REFRESH = (Verdict.REVIEW, Verdict.ADJACENT, Verdict.GAP)


class WorkbookOpenError(RuntimeError):
    """A previous `out/<doc>-tagged.xlsx` is still open (Excel's `~$` lock file)."""


def _scan(folder: Path) -> tuple[list[Path], list[str]]:
    """Readable files directly under `folder` (not recursive): candidates + warnings."""
    files: list[Path] = []
    warnings: list[str] = []
    if not folder.is_dir():
        return files, warnings
    for entry in sorted(p for p in folder.iterdir() if p.is_file()):
        if entry.suffix.lower() == ".csv" and entry.with_suffix(".json").exists():
            warnings.append(f"{entry.name}: CSV twin has no node ids; using the JSON")
            continue
        if not any(reader.can_read(entry) for reader in READERS):
            warnings.append(f"{entry.name}: no reader recognizes this file")
            continue
        files.append(entry)
    return files, warnings


def _search_routed(
    search: Callable[[str, str | None], object],
    suite_name: str,
    unit: TagUnit,
    queries: list[str],
    routed_kinds: list[str],
) -> SearchOutcome:
    """No routable kind means the suite has no such concept (`n/a`), not a gap."""
    if not routed_kinds:
        return SearchOutcome(suite=suite_name, status="n/a", routed_kind=None)
    return search_unit(search, suite_name, unit, queries, routed_kinds)


def _document_title(doc: Document, profile: Profile) -> str:
    """The standard's own title for the Summary sheet; the file key as a fallback."""
    for text in document_context(doc, profile):
        label, sep, value = text.partition(":")
        if sep and label.strip().casefold() == "title" and value.strip():
            return value.strip()
    return doc.key


def discover(path: Path) -> tuple[Path, list[Path]]:
    """(project_dir, input files): `inputs/` folder, bare folder, or a single file."""
    if path.is_file():
        return path.parent, [path]
    inputs_dir = path / "inputs"
    if inputs_dir.is_dir():
        files, _ = _scan(inputs_dir)
        return path, files
    files, _ = _scan(path)
    return path, files


def _input_warnings(path: Path) -> list[str]:
    if path.is_file():
        return []
    inputs_dir = path / "inputs"
    _, warnings = _scan(inputs_dir if inputs_dir.is_dir() else path)
    return warnings


def run(
    path: Path,
    *,
    registry: SuiteRegistry,
    opts: RunOptions,
    llm_client: object | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    del llm_client  # PR3 parameter: model steps are not implemented yet
    start = time.perf_counter()
    run_id = str(uuid.uuid4())

    project, input_files = discover(path)
    if not input_files:
        raise ReaderError(f"no readable input files under {path}")
    run_warnings = _input_warnings(path)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    profile = load_profile(project)
    if opts.depth is not None:
        profile = profile.model_copy(update={"depth": opts.depth})

    model = "none"  # no-LLM mode only, in PR2
    pinned_versions = versions(model)

    suite_names = list(opts.suites) if opts.suites is not None else list(registry.available)
    unknown = [name for name in suite_names if name not in registry.available]
    if unknown:
        available = ", ".join(registry.available)
        raise UnknownSuiteError(f"unknown suite(s) {unknown}; available: {available}")

    ledger = Ledger(project / ".ta-tag")
    documents: list[str] = []
    outputs: list[str] = []
    overall_counts: dict[str, dict[str, int]] = {}
    total_units = 0
    total_reused = 0
    total_searched = 0

    seen_docs: dict[str, str] = {}
    for input_path in input_files:
        doc = read_document(input_path)
        if doc.key in seen_docs:
            # Same document twice (e.g. a renamed copy): tagging it again would
            # overwrite the same workbook and double every count.
            run_warnings.append(
                f"{input_path.name}: same document as {seen_docs[doc.key]}; skipped"
            )
            continue
        seen_docs[doc.key] = input_path.name
        documents.append(doc.key)

        out_dir = project / "out"
        lock_path = out_dir / f"~${doc.key}-tagged.xlsx"
        if lock_path.exists():
            raise WorkbookOpenError(f"close {lock_path} in Excel and re-run")

        say(f"tagging {doc.key}…")
        units, skipped = select_units(doc, profile, rows=opts.rows, limit=opts.limit)
        total_units += len(units)

        results: dict[tuple[str, str], TagRecord] = {}
        reused = 0
        searched = 0

        with ExitStack() as stack:
            suites: dict[str, MeasuredSuite | None] = {}
            for name in suite_names:
                try:
                    runtime = stack.enter_context(registry.open(name))
                    suites[name] = MeasuredSuite(runtime.suite)
                except Exception as exc:  # noqa: BLE001 - recorded per-suite, run continues
                    run_warnings.append(f"{name}: suite unavailable ({type(exc).__name__})")
                    suites[name] = None

            router = Router()

            for index, unit in enumerate(units, start=1):
                if progress is not None and index % 10 == 0:
                    say(f"{doc.key}: {index}/{len(units)} units")

                queries = queries_for(unit)
                routing_kinds = profile.routing.get(unit.kind, [])

                for suite_name in suite_names:
                    fp = fingerprint(unit, suite_name, pinned_versions)
                    existing = None if opts.fresh else ledger.lookup(fp)
                    reusable = (
                        existing is not None
                        and existing.verdict not in _NOT_CACHED
                        and not (opts.refresh and existing.verdict in _RETRY_ON_REFRESH)
                    )
                    if reusable and existing is not None:
                        results[(unit.unit_key, suite_name)] = existing
                        reused += 1
                        continue

                    measured = suites.get(suite_name)
                    if measured is None:
                        outcome = SearchOutcome(
                            suite=suite_name,
                            status="unavailable",
                            routed_kind=None,
                            error="suite_open_failed",
                        )
                    elif not routing_kinds or not queries:
                        outcome = SearchOutcome(suite=suite_name, status="n/a", routed_kind=None)
                    else:

                        def bound_search(
                            text: str, kind: str | None, _measured: MeasuredSuite = measured
                        ) -> object:
                            return _measured.search_nodes(text, kind=kind)

                        try:
                            routed_kinds = router.route(
                                suite_name, bound_search, unit.kind, routing_kinds, queries[0]
                            )
                        except Exception as exc:  # noqa: BLE001 - recorded as unavailable
                            outcome = SearchOutcome(
                                suite=suite_name,
                                status="unavailable",
                                routed_kind=None,
                                error=type(exc).__name__,
                            )
                        else:
                            outcome = _search_routed(
                                bound_search, suite_name, unit, queries, routed_kinds
                            )

                    verdict_result = decide(outcome, llm_on=False)
                    searched += 1
                    record = TagRecord(
                        run_id=run_id,
                        ts=datetime.now(UTC).isoformat(),
                        doc_key=doc.key,
                        unit_key=unit.unit_key,
                        row_id=unit.row_id,
                        suite=suite_name,
                        fingerprint=fp,
                        versions=pinned_versions,
                        kind=unit.kind,
                        clean_text=unit.clean_text,
                        context_path=unit.context_path,
                        queries=outcome.queries,
                        candidates=outcome.candidates,
                        verdict=verdict_result.verdict,
                        reason_code=verdict_result.reason_code,
                        tags=verdict_result.tags,
                        alternatives=verdict_result.alternatives,
                    )
                    ledger.append(record)
                    results[(unit.unit_key, suite_name)] = record

            doc_tool_calls: list[ToolCall] = []
            for measured in suites.values():
                if measured is not None:
                    doc_tool_calls.extend(measured.tool_calls)

        total_reused += reused
        total_searched += searched

        for (_unit_key, suite_name), record in results.items():
            counts = overall_counts.setdefault(suite_name, {})
            counts[record.verdict.value] = counts.get(record.verdict.value, 0) + 1

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc.key}-tagged.xlsx"
        write_workbook(
            out_path,
            doc,
            units,
            skipped,
            results,
            suite_names,
            {
                "title": _document_title(doc, profile),
                "model": model,
                "versions": f"{pinned_versions.pipeline}/{pinned_versions.policy}",
                "units": len(units),
                "seconds": time.perf_counter() - start,
                "reused": reused,
                "searched": searched,
            },
        )
        outputs.append(str(out_path))
        say(f"{doc.key}: done ({len(units)} units, {reused} reused, {searched} searched)")

        # One record per document (not per run): a multi-document project
        # folder should show up as several turns, each about its own doc.
        append_record(
            RunLogRecord(
                run_id=run_id,
                suite=",".join(suite_names),
                plan=["tag"],
                question=f"tag:{doc.key}",
                tools=doc_tool_calls,
            ),
            path=ledger.runlog_path,
        )

    for suite_name, counts in overall_counts.items():
        total = sum(counts.values())
        if total and counts.get(Verdict.UNAVAILABLE.value, 0) == total:
            run_warnings.append(
                f"{suite_name}: unavailable for every row — is its database (Neo4j) running?"
            )

    elapsed = time.perf_counter() - start
    return RunSummary(
        project=str(project),
        documents=documents,
        units=total_units,
        counts=overall_counts,
        reused=total_reused,
        searched=total_searched,
        outputs=outputs,
        seconds=elapsed,
        warnings=run_warnings,
    )
