"""Project memory: append-only JSONL under `<project>/.ta-tag/`.

Separate from `TA_AGENTS_HOME` on purpose (PILOT-PLAN.md "Project memory lives
in the project folder") — a batch run must never touch the user's chat
memory, session store, or run-log.
"""

from __future__ import annotations

import json
from pathlib import Path

from talent_angels.tagging.models import TagRecord
from talent_angels.tagging.rewrite import Rewrite

LEDGER_FILE = "ledger.jsonl"
REVIEWS_FILE = "reviews.jsonl"
GAPS_FILE = "gaps.jsonl"
RUNLOG_FILE = "runlog.jsonl"
REWRITES_FILE = "rewrites.jsonl"


class Ledger:
    """`<project>/.ta-tag/ledger.jsonl` and friends. Loaded once, into dicts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ledger_path = root / LEDGER_FILE
        self.reviews_path = root / REVIEWS_FILE
        self.gaps_path = root / GAPS_FILE
        self.runlog_path = root / RUNLOG_FILE
        self.rewrites_path = root / REWRITES_FILE

        self._by_fingerprint: dict[str, TagRecord] = {}
        self._by_unit_suite: dict[tuple[str, str], TagRecord] = {}
        self.corrupt_lines = 0
        for record in self._read_ledger():
            self._by_fingerprint[record.fingerprint] = record
            self._by_unit_suite[(record.unit_key, record.suite)] = record

        self._rewrites: dict[str, Rewrite] = {}
        self.corrupt_rewrite_lines = 0
        for key, rewrite in self._read_rewrites():
            self._rewrites[key] = rewrite

        self._review_keys: set[str] = set()
        self._reviews: list[dict[str, object]] = []
        self._latest_review: dict[tuple[str, str], dict[str, object]] = {}
        self.corrupt_review_lines = 0
        for payload in self._read_reviews():
            self._review_keys.add(str(payload.get("review_key", "")))
            self._reviews.append(payload)
            unit_key = payload.get("unit_key")
            suite = payload.get("suite")
            if isinstance(unit_key, str) and isinstance(suite, str):
                self._latest_review[(unit_key, suite)] = payload

    def _read_ledger(self) -> list[TagRecord]:
        if not self.ledger_path.is_file():
            return []
        records: list[TagRecord] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TagRecord.model_validate_json(line))
            except ValueError:
                self.corrupt_lines += 1
        return records

    def records(self) -> list[TagRecord]:
        """Every record currently known, latest write per fingerprint."""
        return list(self._by_fingerprint.values())

    def lookup(self, fingerprint: str) -> TagRecord | None:
        """The latest record written for this exact fingerprint, if any."""
        return self._by_fingerprint.get(fingerprint)

    def latest_for(self, unit_key: str, suite: str) -> TagRecord | None:
        """The latest record for this unit x suite, across any attempt/hint."""
        return self._by_unit_suite.get((unit_key, suite))

    def append(self, record: TagRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")
            handle.flush()
        self._by_fingerprint[record.fingerprint] = record
        self._by_unit_suite[(record.unit_key, record.suite)] = record

    def _read_rewrites(self) -> list[tuple[str, Rewrite]]:
        if not self.rewrites_path.is_file():
            return []
        items: list[tuple[str, Rewrite]] = []
        for line in self.rewrites_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                key = payload["key"]
                rewrite = Rewrite.model_validate(payload["rewrite"])
            except (ValueError, KeyError, TypeError):
                self.corrupt_rewrite_lines += 1
                continue
            items.append((key, rewrite))
        return items

    def rewrite_lookup(self, key: str) -> Rewrite | None:
        """The cached rw-1 result for this `rewrite.rewrite_cache_key`, if any."""
        return self._rewrites.get(key)

    def rewrite_append(self, key: str, rewrite: Rewrite) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.rewrites_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "rewrite": rewrite.model_dump()}))
            handle.write("\n")
            handle.flush()
        self._rewrites[key] = rewrite

    def _read_reviews(self) -> list[dict[str, object]]:
        if not self.reviews_path.is_file():
            return []
        payloads: list[dict[str, object]] = []
        for line in self.reviews_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                self.corrupt_review_lines += 1
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
            else:
                self.corrupt_review_lines += 1
        return payloads

    def review_applied(self, review_key: str) -> bool:
        """`True` once this exact decision (same key, action, hint, base fp) has run."""
        return review_key in self._review_keys

    def review_append(self, record: dict[str, object]) -> None:
        """Append one `reviews.jsonl` line; marks `record["review_key"]` as applied."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self.reviews_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
            handle.flush()
        review_key = str(record.get("review_key", ""))
        self._review_keys.add(review_key)
        self._reviews.append(record)
        unit_key = record.get("unit_key")
        suite = record.get("suite")
        if isinstance(unit_key, str) and isinstance(suite, str):
            self._latest_review[(unit_key, suite)] = record

    def reviews(self) -> list[dict[str, object]]:
        """Every review decision ever applied, in file order (oldest first)."""
        return list(self._reviews)

    def latest_review_for(self, unit_key: str, suite: str) -> dict[str, object] | None:
        """The most recently applied review decision for this unit x suite, if any."""
        return self._latest_review.get((unit_key, suite))
