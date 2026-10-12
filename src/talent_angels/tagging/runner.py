"""Orchestrates one `ta-agent tag` run: read -> select -> search -> verdict -> write.

Never calls `run_turn`/`interpret_question` (PILOT-PLAN.md "Architecture") —
no plan cache, no episodes, no profile writes, and project memory lives under
`<project>/.ta-tag/`, never `TA_AGENTS_HOME`.

Model steps (PR3) add two phases between search and verdict: Phase R rewrites
a client line into short search phrases (one call per parent group, cached in
`.ta-tag/rewrites.jsonl`), Phase B asks the model which of the search hits
actually match (`--workers` makes these calls concurrent; search itself
always stays sequential — embedding search is not thread-safe). A `none`/
no-valid-pick choice gets one widened search+choose attempt before becoming
a gap.
"""

from __future__ import annotations

import contextvars
import json
import shutil
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from talent_angels.llm.protocol import LLMClient
from talent_angels.runlog import (
    GenAIUsage,
    RunLogRecord,
    StageUsage,
    ToolCall,
    append_record,
    estimate_turn_cost_usd,
)
from talent_angels.suites import SuiteRegistry, UnknownSuiteError
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.tagging import retry as retry_context
from talent_angels.tagging.choose import Choice, ValidatedChoice, choose, validate
from talent_angels.tagging.keys import REWRITE_VERSION, fingerprint, versions
from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import (
    Document,
    RunOptions,
    RunSummary,
    SearchOutcome,
    TagRecord,
    TagUnit,
    Verdict,
    VerdictResult,
    Versions,
)
from talent_angels.tagging.profile import Profile, load_profile
from talent_angels.tagging.prompts import choose_system, rewrite_system
from talent_angels.tagging.readers import READERS, read_document
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.review import (
    RetryRequest,
    ReviewOutcome,
    apply_decisions,
    read_decisions,
    review_stats,
    unrecognised_decisions,
)
from talent_angels.tagging.rewrite import Rewrite, rewrite_cache_key, rewrite_group
from talent_angels.tagging.search import Router, queries_for, search_unit
from talent_angels.tagging.select import document_context, select_units
from talent_angels.tagging.twins import describe_csv_twin
from talent_angels.tagging.verdict import decide
from talent_angels.tagging.writers.csv import write_gaps_csv
from talent_angels.tagging.writers.xlsx import build_gap_rows, display_name, write_workbook

#: Not cached — rules 1/3 of the verdict policy (verdict.py) — so a transient outage or a
#: botched model call is retried next run instead of frozen into the ledger.
_NOT_CACHED = (Verdict.ERROR, Verdict.UNAVAILABLE)
_RETRY_ON_REFRESH = (Verdict.REVIEW, Verdict.ADJACENT, Verdict.GAP)
_REWRITE_CHUNK = 6


class WorkbookOpenError(RuntimeError):
    """A previous `out/<doc>-tagged.xlsx` is still open (Excel's `~$` lock file)."""


@dataclass
class _Pending:
    """One unit x suite not reused from the ledger: everything needed to decide."""

    unit: TagUnit
    suite_name: str
    outcome: SearchOutcome
    rewrite: Rewrite
    queries: list[str]
    attempt: int = 0
    choice: Choice | None = None
    #: A reviewer's "Retry with hint"/"Reject" (review.py): the `choose` hint
    #: for this attempt, and whether `verdict.decide` should treat it as
    #: already-widened (a `none` fit becomes `gap`, not another widen round).
    hint: str = ""
    is_retry: bool = False


def _scan(folder: Path) -> tuple[list[Path], list[str]]:
    """Readable files directly under `folder` (not recursive): candidates + warnings."""
    files: list[Path] = []
    warnings: list[str] = []
    if not folder.is_dir():
        return files, warnings
    for entry in sorted(p for p in folder.iterdir() if p.is_file()):
        if entry.suffix.lower() == ".csv" and entry.with_suffix(".json").exists():
            warnings.append(describe_csv_twin(entry, entry.with_suffix(".json")))
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


def _history_copy(out_path: Path) -> None:
    """Snapshot the previous workbook to `out/history/` before it is overwritten."""
    if not out_path.is_file():
        return
    history_dir = out_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(out_path, history_dir / f"{out_path.stem}-{stamp}{out_path.suffix}")


def _search_by_suite(
    suites: dict[str, MeasuredSuite | None],
) -> Callable[[str], Callable[[str, str | None], object] | None]:
    """`review.apply_decisions`'s `search_by_suite`: a bound `search_nodes` per suite name."""

    def _get(suite_name: str) -> Callable[[str, str | None], object] | None:
        measured = suites.get(suite_name)
        if measured is None:
            return None

        def _search(text: str, kind: str | None, _measured: MeasuredSuite = measured) -> object:
            return _measured.search_nodes(text, kind=kind)

        return _search

    return _get


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


def _needs_rewrite(unit: TagUnit) -> bool:
    """Occupation rows with seed phrases use code-derived queries; no model call."""
    return not (unit.role == "row_rule" and unit.seed_phrases)


def _rewrite_phase(
    client: LLMClient,
    units: list[TagUnit],
    doc_context: list[str],
    ledger: Ledger,
    model: str,
    *,
    system: str,
    document_label: str,
) -> tuple[dict[str, Rewrite], list[StageUsage]]:
    """Fill in a `Rewrite` for every unit that needs one, cache-first, grouped by parent."""
    rewrites: dict[str, Rewrite] = {}
    stages: list[StageUsage] = []
    pending_by_group: dict[str, list[TagUnit]] = defaultdict(list)
    for unit in units:
        if not _needs_rewrite(unit):
            continue
        key = rewrite_cache_key(unit, model, REWRITE_VERSION)
        cached = ledger.rewrite_lookup(key)
        if cached is not None:
            rewrites[unit.unit_key] = cached
            continue
        pending_by_group[unit.group_key].append(unit)

    for group_units in pending_by_group.values():
        for start in range(0, len(group_units), _REWRITE_CHUNK):
            chunk = group_units[start : start + _REWRITE_CHUNK]
            chunk_map, chunk_stages = rewrite_group(
                client, chunk, doc_context, system=system, document_label=document_label
            )
            stages.extend(chunk_stages)
            for unit in chunk:
                rw = chunk_map.get(unit.unit_key, Rewrite())
                rewrites[unit.unit_key] = rw
                ledger.rewrite_append(rewrite_cache_key(unit, model, REWRITE_VERSION), rw)
    return rewrites, stages


def run(
    path: Path,
    *,
    registry: SuiteRegistry,
    opts: RunOptions,
    llm_client: LLMClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    start = time.perf_counter()
    run_id = str(uuid.uuid4())
    llm_on = not opts.no_llm and llm_client is not None

    project, input_files = discover(path)
    if not input_files:
        raise ReaderError(f"no readable input files under {path}")
    run_warnings = _input_warnings(path)

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    profile = load_profile(project, name=opts.profile)
    if opts.depth is not None:
        profile = profile.model_copy(update={"depth": opts.depth})

    model = str(getattr(llm_client, "model", "") or "none") if llm_on else "none"
    pinned_versions = versions(model)

    suite_names = list(opts.suites) if opts.suites is not None else list(registry.available)
    unknown = [name for name in suite_names if name not in registry.available]
    if unknown:
        available = ", ".join(registry.available)
        raise UnknownSuiteError(f"unknown suite(s) {unknown}; available: {available}")

    # Computed once per run: the two system prompts only vary by the profile's own
    # document_type/attached suites, never per unit or per call (keeps the CONOCER
    # `competency_standard` + esco/onet rendering byte-identical to before PR5).
    suite_display_names = [display_name(name) for name in suite_names]
    rewrite_sys = rewrite_system(profile.document_type, suite_display_names)
    choose_sys = choose_system(profile.document_type)

    ledger = Ledger(project / ".ta-tag")
    documents: list[str] = []
    outputs: list[str] = []
    overall_counts: dict[str, dict[str, int]] = {}
    total_units = 0
    total_reused = 0
    total_searched = 0
    total_reviews_applied = 0
    total_retried = 0
    all_stages: list[StageUsage] = []

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
        out_path = out_dir / f"{doc.key}-tagged.xlsx"

        # Read the expert's Decision/Hint off the *previous* workbook before any
        # search or model spend happens for this doc (PILOT-PLAN.md "Review loop").
        review_decisions, decision_warnings = read_decisions(out_path)
        kept_decisions = unrecognised_decisions(out_path)
        run_warnings.extend(decision_warnings)

        say(f"tagging {doc.key}…")
        units, skipped = select_units(doc, profile, rows=opts.rows, limit=opts.limit)
        total_units += len(units)

        results: dict[tuple[str, str], TagRecord] = {}
        reused = 0
        searched = 0
        retried = 0
        doc_stages: list[StageUsage] = []

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

            review_outcome: ReviewOutcome = apply_decisions(
                review_decisions,
                ledger,
                search_by_suite=_search_by_suite(suites),
                run_id=run_id,
            )
            run_warnings.extend(review_outcome.warnings)
            review_retries: dict[tuple[str, str], RetryRequest] = review_outcome.retries

            rewrites: dict[str, Rewrite] = {}
            if llm_on and llm_client is not None:
                say(f"{doc.key}: step 1/4 skill phrases for {len(units)} rows…")
                rw_map, rw_stages = _rewrite_phase(
                    llm_client,
                    units,
                    document_context(doc, profile),
                    ledger,
                    model,
                    system=rewrite_sys,
                    document_label=profile.document_label,
                )
                rewrites = rw_map
                doc_stages.extend(rw_stages)
                say(
                    f"{doc.key}: step 1/4 done "
                    + (
                        f"({len(rw_stages)} model calls, the rest reused)"
                        if rw_stages
                        else "(all reused, no model calls)"
                    )
                )

            pending: dict[tuple[str, str], _Pending] = {}

            for index, unit in enumerate(units, start=1):
                rw = rewrites.get(unit.unit_key, Rewrite())
                base_queries = queries_for(unit, rw.labels, rw.description)
                routing_kinds = profile.routing.get(unit.kind, [])

                for suite_name in suite_names:
                    key_pair = (unit.unit_key, suite_name)
                    retry_req = review_retries.get(key_pair)
                    latest = ledger.latest_for(unit.unit_key, suite_name)

                    if retry_req is not None:
                        # A reviewer's retry/reject always re-does the row this
                        # run, even if the ledger (or a human decision) would
                        # otherwise reuse it.
                        reusable = False
                    elif latest is not None and latest.decided_by == "human":
                        # Human decisions are never overwritten automatically —
                        # not by `--refresh`, not by `--fresh`.
                        reusable = True
                    else:
                        existing = None if opts.fresh else latest
                        reusable = (
                            existing is not None
                            and existing.versions == pinned_versions
                            and existing.verdict not in _NOT_CACHED
                            and not (opts.refresh and existing.verdict in _RETRY_ON_REFRESH)
                        )
                    if reusable and latest is not None:
                        results[key_pair] = latest
                        reused += 1
                        continue

                    queries = base_queries
                    hint = ""
                    attempt = 0
                    is_retry = False
                    if retry_req is not None:
                        attempt = retry_req.attempt
                        is_retry = True
                        retried += 1
                        extra = retry_context.extra_queries_for(unit, rw, profile, attempt)
                        queries = [*base_queries, *extra]
                        siblings = retry_context.siblings_for(unit, units)
                        hint = retry_context.choose_hint_for(retry_req.hint, attempt, siblings)

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

                    searched += 1
                    pending[key_pair] = _Pending(
                        unit=unit,
                        suite_name=suite_name,
                        outcome=outcome,
                        rewrite=rw,
                        queries=queries,
                        attempt=attempt,
                        hint=hint,
                        is_retry=is_retry,
                    )

                # After the row, so the counts include it (the last line adds up).
                if progress is not None and (index % 10 == 0 or index == len(units)):
                    say(
                        f"{doc.key}: step 2/4 rows {index}/{len(units)} "
                        f"({searched} answers searched, {reused} reused)"
                    )

            verdicts, choose_stages = _decide_pending(
                pending,
                llm_client=llm_client,
                llm_on=llm_on,
                progress=_step_counter(say, f"{doc.key}: step 3/4 model choosing among candidates"),
                workers=opts.workers,
                system=choose_sys,
                document_label=profile.document_label,
            )
            doc_stages.extend(choose_stages)

            to_widen = sum(1 for verdict in verdicts.values() if verdict.needs_widening)
            if to_widen:
                say(f"{doc.key}: step 4/4 second look with wider context for {to_widen} answers…")
            fps, widened_stages = _widen_pending(
                pending,
                verdicts,
                llm_client=llm_client,
                llm_on=llm_on,
                router=router,
                suites=suites,
                profile=profile,
                pinned_versions=pinned_versions,
                system=choose_sys,
                document_label=profile.document_label,
            )
            doc_stages.extend(widened_stages)

            gap_rows: list[dict[str, object]] = []
            for key, p in pending.items():
                verdict_result = verdicts[key]
                attempt = p.attempt
                record_fp = fps[key]
                record = TagRecord(
                    run_id=run_id,
                    ts=datetime.now(UTC).isoformat(),
                    doc_key=doc.key,
                    unit_key=p.unit.unit_key,
                    row_id=p.unit.row_id,
                    suite=p.suite_name,
                    fingerprint=record_fp,
                    attempt=attempt,
                    versions=pinned_versions,
                    kind=p.unit.kind,
                    clean_text=p.unit.clean_text,
                    context_path=p.unit.context_path,
                    hint=p.hint,
                    queries=p.outcome.queries,
                    candidates=p.outcome.candidates,
                    rewrite=p.rewrite.model_dump() if llm_on else None,
                    choice=p.choice.model_dump() if p.choice is not None else None,
                    verdict=verdict_result.verdict,
                    reason_code=verdict_result.reason_code,
                    tags=verdict_result.tags,
                    alternatives=verdict_result.alternatives,
                    covers=verdict_result.covers,
                    missing_skill=verdict_result.missing_skill,
                    model_reason=verdict_result.model_reason,
                )
                ledger.append(record)
                results[key] = record

                if (
                    llm_on
                    and record.verdict == Verdict.GAP
                    and record.reason_code == "model_none"
                    and p.unit.kind in profile.report_kinds
                ):
                    gap_rows.append(
                        {
                            "doc_key": doc.key,
                            "unit_key": p.unit.unit_key,
                            "row_id": p.unit.row_id,
                            "suite": p.suite_name,
                            "kind": p.unit.kind,
                            "clean_text": p.unit.clean_text,
                            "context_path": p.unit.context_path,
                            "missing_skill": record.missing_skill,
                            "ts": record.ts,
                        }
                    )

            if gap_rows:
                _append_gaps(ledger, gap_rows)

            doc_tool_calls: list[ToolCall] = []
            for measured in suites.values():
                if measured is not None:
                    doc_tool_calls.extend(measured.tool_calls)

        total_reused += reused
        total_searched += searched
        total_reviews_applied += review_outcome.applied
        total_retried += retried
        all_stages.extend(doc_stages)

        for (_unit_key, suite_name), record in results.items():
            counts = overall_counts.setdefault(suite_name, {})
            counts[record.verdict.value] = counts.get(record.verdict.value, 0) + 1

        previous_decisions = {
            key: payload
            for key in results
            if (payload := ledger.latest_review_for(*key)) is not None
        }
        stats = review_stats(ledger, units)

        out_dir.mkdir(parents=True, exist_ok=True)
        if review_outcome.applied:
            # Only a workbook whose decisions were consumed is worth keeping:
            # a plain re-run would otherwise pile up identical copies.
            _history_copy(out_path)
        doc_cost = estimate_turn_cost_usd(doc_stages, model)
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
                "profile": profile.name,
                "units": len(units),
                "seconds": time.perf_counter() - start,
                "reused": reused,
                "searched": searched,
                "model_calls": sum(stage.calls for stage in doc_stages),
                "cost_usd": doc_cost.total if doc_cost.known else None,
                "reviews_applied": review_outcome.applied,
                "retried": retried,
            },
            report_kinds=frozenset(profile.report_kinds),
            previous_decisions=previous_decisions,
            kept_decisions=kept_decisions,
            stats=stats,
            document_label=profile.document_label,
        )
        workbook_gap_rows = build_gap_rows(
            units, results, suite_names, frozenset(profile.report_kinds)
        )
        write_gaps_csv(out_dir / f"{doc.key}-gaps.csv", workbook_gap_rows)
        outputs.append(str(out_path))
        say(
            f"{doc.key}: done ({len(units)} units, {reused} reused, {searched} searched, "
            f"{review_outcome.applied} reviews applied, {retried} retried)"
        )

        # One record per document (not per run): a multi-document project
        # folder should show up as several turns, each about its own doc.
        gen_ai = GenAIUsage(
            provider_name=getattr(llm_client, "provider", "none") if llm_on else "none",
            request_model=model,
            input_tokens=sum(s.input_tokens for s in doc_stages),
            output_tokens=sum(s.output_tokens for s in doc_stages),
            reasoning_tokens=sum(s.reasoning_tokens for s in doc_stages),
            cache_read_input_tokens=sum(s.cache_read_input_tokens for s in doc_stages),
            cache_creation_input_tokens=sum(s.cache_creation_input_tokens for s in doc_stages),
            calls=sum(s.calls for s in doc_stages),
            stages=doc_stages,
        )
        append_record(
            RunLogRecord(
                run_id=run_id,
                suite=",".join(suite_names),
                plan=["tag"],
                question=f"tag:{doc.key}",
                gen_ai=gen_ai,
                cost_usd=doc_cost,
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
    total_cost = estimate_turn_cost_usd(all_stages, model)
    return RunSummary(
        project=str(project),
        profile=profile.name,
        documents=documents,
        units=total_units,
        counts=overall_counts,
        reused=total_reused,
        searched=total_searched,
        outputs=outputs,
        seconds=elapsed,
        warnings=run_warnings,
        model_calls=sum(stage.calls for stage in all_stages),
        cost_usd=total_cost.total if (all_stages and total_cost.known) else None,
        reviews_applied=total_reviews_applied,
        retried=total_retried,
    )


def _step_counter(say: Callable[[str], None], label: str) -> Callable[[int, int], None]:
    """A `done/total` reporter bound to one document's step label."""

    def report(done: int, total: int) -> None:
        say(f"{label} {done}/{total}")

    return report


def _decide_pending(
    pending: dict[tuple[str, str], _Pending],
    *,
    llm_client: LLMClient | None,
    llm_on: bool,
    workers: int,
    system: str,
    document_label: str,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[tuple[str, str], VerdictResult], list[StageUsage]]:
    """First decide pass: immediate for non-`ok`/no-candidate outcomes, modelled for the rest."""
    verdicts: dict[tuple[str, str], VerdictResult] = {}
    stages: list[StageUsage] = []
    needs_choose: list[tuple[str, str]] = []

    for key, p in pending.items():
        if llm_on and p.outcome.status == "ok" and p.outcome.candidates:
            needs_choose.append(key)
        else:
            verdicts[key] = decide(p.outcome, None, llm_on=llm_on, labels=p.rewrite.labels)

    def _choose_one(key: tuple[str, str]) -> tuple[Choice | None, list[StageUsage]]:
        p = pending[key]
        assert llm_client is not None
        return choose(
            llm_client,
            p.unit,
            p.outcome,
            p.outcome.descriptions,
            suite_display=display_name(p.suite_name),
            hint=p.hint,
            widened=p.is_retry,
            system=system,
            document_label=document_label,
        )

    def _choose_one_in_context(key: tuple[str, str]) -> tuple[Choice | None, list[StageUsage]]:
        ctx = contextvars.copy_context()
        return ctx.run(_choose_one, key)

    choose_results: dict[tuple[str, str], tuple[Choice | None, list[StageUsage]]] = {}
    total = len(needs_choose)

    def _tick() -> None:
        # The longest phase of a run: without a counter it looks frozen.
        done = len(choose_results)
        if progress is not None and (done % 10 == 0 or done == total):
            progress(done, total)

    if needs_choose:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_choose_one_in_context, key): key for key in needs_choose
                }
                # Results are keyed, so finishing order never changes the output.
                for future in as_completed(futures):
                    choose_results[futures[future]] = future.result()
                    _tick()
        else:
            for key in needs_choose:
                choose_results[key] = _choose_one(key)
                _tick()

    for key in needs_choose:
        p = pending[key]
        choice, call_stages = choose_results[key]
        stages.extend(call_stages)
        p.choice = choice
        validated: ValidatedChoice | None = (
            validate(p.outcome.candidates, p.unit.kind, choice) if choice is not None else None
        )
        verdicts[key] = decide(
            p.outcome, validated, llm_on=True, widened=p.is_retry, labels=p.rewrite.labels
        )

    return verdicts, stages


def _widen_pending(
    pending: dict[tuple[str, str], _Pending],
    verdicts: dict[tuple[str, str], VerdictResult],
    *,
    llm_client: LLMClient | None,
    llm_on: bool,
    router: Router,
    suites: dict[str, MeasuredSuite | None],
    profile: Profile,
    pinned_versions: Versions,
    system: str,
    document_label: str,
) -> tuple[dict[tuple[str, str], str], list[StageUsage]]:
    """The widened re-attempt (rule 6) for every unit x suite that asked for one.

    Returns every key's final ledger `fingerprint` (attempt 0, or 1 if widened)
    alongside the stages the widened `choose` calls made.
    """
    stages: list[StageUsage] = []
    fps: dict[tuple[str, str], str] = {}

    for key, p in pending.items():
        verdict_result = verdicts.get(key)
        widen = (
            llm_on
            and llm_client is not None
            and verdict_result is not None
            and verdict_result.needs_widening
        )
        if not widen:
            fps[key] = fingerprint(
                p.unit, p.suite_name, pinned_versions, hint=p.hint, attempt=p.attempt
            )
            continue
        assert llm_client is not None

        measured = suites.get(p.suite_name)
        if measured is None:
            fps[key] = fingerprint(
                p.unit, p.suite_name, pinned_versions, hint=p.hint, attempt=p.attempt
            )
            continue

        def bound_search(
            text: str, kind: str | None, _measured: MeasuredSuite = measured
        ) -> object:
            return _measured.search_nodes(text, kind=kind)

        description = p.rewrite.description or p.unit.clean_text
        parent_heading = profile.clean(p.unit.context_path[-1]) if p.unit.context_path else ""
        extra_query = f"{description} {parent_heading}".strip() if parent_heading else description

        routing_kinds = profile.routing.get(p.unit.kind, [])
        try:
            routed_kinds = router.route(
                p.suite_name, bound_search, p.unit.kind, routing_kinds, extra_query
            )
        except Exception:  # noqa: BLE001 - keep the original outcome on a flaky widen
            fps[key] = fingerprint(
                p.unit, p.suite_name, pinned_versions, hint=p.hint, attempt=p.attempt
            )
            continue

        widened_outcome = search_unit(
            bound_search, p.suite_name, p.unit, [*p.queries, extra_query], routed_kinds
        )
        choice, call_stages = choose(
            llm_client,
            p.unit,
            widened_outcome,
            widened_outcome.descriptions,
            suite_display=display_name(p.suite_name),
            hint="",
            widened=True,
            system=system,
            document_label=document_label,
        )
        stages.extend(call_stages)
        p.outcome = widened_outcome
        p.choice = choice
        p.attempt = 1
        validated: ValidatedChoice | None = (
            validate(widened_outcome.candidates, p.unit.kind, choice)
            if choice is not None
            else None
        )
        verdicts[key] = decide(
            widened_outcome, validated, llm_on=True, widened=True, labels=p.rewrite.labels
        )
        fps[key] = fingerprint(p.unit, p.suite_name, pinned_versions, attempt=1)

    return fps, stages


def _append_gaps(ledger: Ledger, rows: list[dict[str, object]]) -> None:
    ledger.root.mkdir(parents=True, exist_ok=True)
    with ledger.gaps_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
