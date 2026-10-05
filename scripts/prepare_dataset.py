"""Convert the original Kaggle XLS to CSV and report data quality.

Requires xlrd==2.0.2 (pip install xlrd==2.0.2).
Run from any directory: python scripts/prepare_dataset.py
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "python"))
import xlrd


def main():
    source = ROOT / "data/raw/titanic3/titanic3.xls"
    sheet = xlrd.open_workbook(source).sheet_by_index(0)
    columns = sheet.row_values(0)
    rows = [sheet.row_values(i) for i in range(1, sheet.nrows)]
    rows = [row for row in rows if any(value != "" for value in row)]
    assert len(columns) == len(set(columns)), "Duplicate column names"
    target = columns.index("survived")
    assert all(row[target] in (0, 1) for row in rows), "Invalid survival label"
    output = ROOT / "data/titanic.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([
                int(value) if isinstance(value, float) and value.is_integer() else value
                for value in row
            ])
    report = {
        "source_url": "https://www.kaggle.com/datasets/vinicius150987/titanic3",
        "download_url": "https://www.kaggle.com/api/v1/datasets/download/vinicius150987/titanic3",
        "download_date": "2026-10-02",
        "rows": len(rows),
        "columns": columns,
        "survival_counts": dict(Counter(int(row[target]) for row in rows)),
        "missing": {name: sum(row[i] == "" for row in rows) for i, name in enumerate(columns)},
        "duplicate_rows": len(rows) - len(set(tuple(row) for row in rows)),
        "exclude_from_features": ["survived", "boat", "body"],
        "sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "data/raw/titanic3.zip", source, output)
        },
    }
    with output.open(newline="", encoding="utf-8") as handle:
        reread = list(csv.DictReader(handle))
    assert len(reread) == len(rows)
    assert all(len(row) == len(columns) and None not in row for row in reread)
    (ROOT / "data/quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
