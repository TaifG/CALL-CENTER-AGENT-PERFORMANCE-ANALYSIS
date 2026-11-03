"""Compute accuracy only from explicit human labels. Blank cells remain unscored."""

import csv, json
from pathlib import Path

file = Path(__file__).parent / "results/manual_review.csv"
if not file.exists():
    raise SystemExit("No review file yet. Run the analysis stage.")
rows = list(csv.DictReader(file.read_text().splitlines()))
report = {}
for field in [
    "summary_correct_0_1",
    "outcome_correct_0_1",
    "sentiment_correct_0_1",
    "evidence_correct_0_1",
]:
    eligible = [
        r for r in rows if r[field].strip() in {"0", "1"} and r["reviewer"].strip()
    ]
    report[field] = {
        "reviewed": len(eligible),
        "total_calls": len(rows),
        "accuracy": (
            sum(int(r[field]) for r in eligible) / len(eligible) if eligible else None
        ),
    }
file.with_name("manual_metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
