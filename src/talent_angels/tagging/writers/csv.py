"""`out/<doc>-gaps.csv`: the Gaps sheet's own rows, for a tool that isn't Excel.

UTF-8 with a BOM (`utf-8-sig`) so Excel itself opens the file with accents
intact instead of guessing the wrong codepage.
"""

from __future__ import annotations

import csv
from pathlib import Path

from talent_angels.tagging.writers.xlsx import GapRow

_HEADERS = [
    "Row ID",
    "Where",
    "Line",
    "Kind",
    "Taxonomy",
    "Verdict",
    "Proposed missing skill (by TA — not taxonomy fact)",
    "Nearest existing (adjacent picks)",
]


def write_gaps_csv(path: Path, rows: list[GapRow]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADERS)
        for row in rows:
            writer.writerow(
                [
                    row.row_id,
                    row.where,
                    row.line,
                    row.kind,
                    row.taxonomy,
                    row.verdict,
                    row.missing_skill,
                    row.nearest_existing,
                ]
            )
    return path
