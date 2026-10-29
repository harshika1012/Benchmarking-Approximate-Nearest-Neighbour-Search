import argparse
import csv
from pathlib import Path

from report_part2 import read

PARTS = [
    ("1", "exact_*.csv"),
    ("2", "part2/*.csv"),
    ("3", "part3/*.csv"),
    ("4", "part4/*.csv"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path("results_quick" if args.quick else "results")

    rows = []
    for part, pattern in PARTS:
        for path in sorted(root.glob(pattern)):
            rows += [dict(part=part, source=path.name, **r) for r in read(path)]
    if not rows:
        raise SystemExit(f"No results found under {root}/")

    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(root / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {root / 'results.csv'}")


if __name__ == "__main__":
    main()
