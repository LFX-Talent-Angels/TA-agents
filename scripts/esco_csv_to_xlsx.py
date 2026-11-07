"""Convert the official ESCO CSV distribution into the xlsx tables the loader reads.

Why: https://esco.ec.europa.eu/en/use-esco/download no longer publishes an xlsx
DATABASE bundle — the English classification ships as CSV (or RDF/TTL). The
loader ``ta_taxonomies.suites.esco.load --mode full`` reads ``*_en.xlsx`` tables,
so this is a one-off shim from the official CSVs to that shape. Column names are
copied verbatim; no values are invented, reordered, or dropped.

    python scripts/esco_csv_to_xlsx.py <csv_dir> <xlsx_out_dir>
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from openpyxl import Workbook

# Tables the loader requires (plus the optional skill-skill relations).
TABLES = [
    "occupations_en",
    "skills_en",
    "ISCOGroups_en",
    "skillGroups_en",
    "occupationSkillRelations_en",
    "broaderRelationsOccPillar_en",
    "broaderRelationsSkillPillar_en",
    "skillSkillRelations_en",
]


def convert(csv_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(10 * 1024 * 1024)
    for name in TABLES:
        src = csv_dir / f"{name}.csv"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        started = time.perf_counter()
        wb = Workbook(write_only=True)
        ws = wb.create_sheet(title=name[:31])
        rows = 0
        with src.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                # openpyxl rejects a leading "=" as a formula; ESCO labels never
                # start with one, but keep values as plain strings regardless.
                ws.append([c if c != "" else None for c in row])
                rows += 1
        dest = out_dir / f"{name}.xlsx"
        wb.save(dest)
        wb.close()
        elapsed = time.perf_counter() - started
        print(f"{name}: {rows - 1:,} data rows -> {dest.name} ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
