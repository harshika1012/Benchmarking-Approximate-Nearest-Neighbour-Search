
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
INPUTS = [
    "hnsw_sift_final.csv",
    "ivfflat_sift_final.csv",
    "ivfpq_sift_final.csv",
    "hnsw_glove_final.csv",
    "ivfflat_glove_final.csv",
    "ivfpq_glove_final.csv",
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def number(row, *keys):
    for key in keys:
        if row.get(key) not in (None, ""):
            return float(row[key])
    raise KeyError(f"None of these columns exist: {keys}")


def main():
    rows = []
    missing = [name for name in INPUTS if not (RESULTS / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing result files: {missing}")

    for name in INPUTS:
        rows.extend(read_csv(RESULTS / name))

    # Save a combined table without discarding index-specific parameters.
    columns = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (RESULTS / "all_final_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Plot recall against QPS separately for each dataset.
    for dataset in sorted(set(r.get("dataset", "") for r in rows)):
        subset = [r for r in rows if r.get("dataset") == dataset]
        plt.figure(figsize=(8, 5))

        for index_name in sorted(set(r.get("index", "") for r in subset)):
            points = []
            for row in subset:
                if row.get("index") != index_name:
                    continue
                try:
                    recall = number(
                        row, "final_recall_at_10", "recall_at_10"
                    )
                    qps = number(
                        row, "final_single_qps_median",
                        "single_query_qps_median", "single_qps"
                    )
                    points.append((recall, qps))
                except (KeyError, ValueError):
                    continue

            points.sort()
            if points:
                plt.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    label=index_name,
                )

        plt.xlabel("Recall@10")
        plt.ylabel("Single-query QPS (median)")
        plt.title(f"{dataset}: Recall vs QPS")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        safe_name = dataset.lower().replace("-", "_")
        plt.savefig(RESULTS / f"{safe_name}_recall_vs_qps.png", dpi=200)
        plt.close()

    print(f"Combined {len(rows)} rows into results/all_final_results.csv")
    print("Generated recall-vs-QPS plots where matching columns were found.")


if __name__ == "__main__":
    main()
