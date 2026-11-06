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
