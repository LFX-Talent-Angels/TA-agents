"""`Ledger`: append/lookup/latest, and tolerance of a corrupt line."""

from __future__ import annotations

from pathlib import Path

from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import TagRecord, Verdict, Versions


def _record(
    unit_key: str, suite: str, fingerprint: str, verdict: Verdict = Verdict.REVIEW
) -> TagRecord:
    return TagRecord(
        run_id="run-1",
        ts="2026-10-11T00:00:00+00:00",
        doc_key="doc",
        unit_key=unit_key,
        row_id="1",
        suite=suite,
        fingerprint=fingerprint,
        versions=Versions(
            pipeline="tag-1",
            policy="tag-v1",
            prompts="rw-1/ch-1",
            ta_taxonomies="unknown",
            model="none",
        ),
        kind="performance",
        clean_text="do the thing",
        verdict=verdict,
        reason_code="search_only",
    )


def test_append_then_lookup_by_fingerprint(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    record = _record("u1", "esco", "fp1")
    ledger.append(record)

    reloaded = Ledger(tmp_path / ".ta-tag")
    assert reloaded.lookup("fp1") is not None
    assert reloaded.lookup("fp1").unit_key == "u1"  # type: ignore[union-attr]
    assert reloaded.lookup("missing") is None


def test_latest_for_unit_suite_returns_most_recent(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record("u1", "esco", "fp1", verdict=Verdict.REVIEW))
    ledger.append(_record("u1", "esco", "fp2", verdict=Verdict.ACCEPTED))

    latest = ledger.latest_for("u1", "esco")
    assert latest is not None
    assert latest.fingerprint == "fp2"
    assert latest.verdict == Verdict.ACCEPTED


def test_records_lists_every_known_fingerprint(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record("u1", "esco", "fp1"))
    ledger.append(_record("u2", "onet", "fp2"))
    assert {r.fingerprint for r in ledger.records()} == {"fp1", "fp2"}


def test_append_creates_directory_lazily(tmp_path: Path) -> None:
    root = tmp_path / ".ta-tag"
    assert not root.exists()
    ledger = Ledger(root)
    assert not root.exists()
    ledger.append(_record("u1", "esco", "fp1"))
    assert root.exists()


def test_corrupt_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / ".ta-tag"
    root.mkdir(parents=True)
    good = _record("u1", "esco", "fp1")
    path = root / "ledger.jsonl"
    path.write_text(good.model_dump_json() + "\n" + "{not valid json\n")

    ledger = Ledger(root)
    assert ledger.corrupt_lines == 1
    assert ledger.lookup("fp1") is not None
