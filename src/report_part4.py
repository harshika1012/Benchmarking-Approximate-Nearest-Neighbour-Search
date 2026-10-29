import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_part2 import COLORS, read

INDEXES = ["HNSW", "IVF-Flat", "IVF-PQ"]
PANELS = [
    ("final_recall", "Recall@10 (fixed setting)", False),
    ("single_qps_median", "Queries per second", True),
    ("ndis_per_query", "Distance computations / query", True),
]


def series(rows, index, state, column):
    return sorted((float(r["shift"]), float(r[column])) for r in rows
                  if r["index"] == index and r["state"] == state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = Path("results_quick" if args.quick else "results")
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    rows = read(root / "part4" / "drift.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (column, label, log) in zip(axes, PANELS):
        for name in INDEXES:
            for state, style in (("stale", "-"), ("rebuilt", "--")):
                pts = series(rows, name, state, column)
                if pts:
                    ax.plot(*zip(*pts), style, marker="o", ms=4,
                            color=COLORS[name], label=f"{name} ({state})")
        ax.set_xlabel("Shift of new data (centre-spread units)")
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Drift: index built on old data (solid) vs rebuilt (dashed); "
                 "search setting fixed on the control")
    fig.tight_layout()
    fig.savefig(root / "figures" / "part4_drift.png", dpi=200)
    plt.close(fig)

    lines = ["| Shift | Index | State | Setting | Recall | QPS | Dist. / query "
             "| List imbalance | Update time (s) |",
             "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (float(r["shift"]),
                                         INDEXES.index(r["index"]), r["state"])):
        imb = f"{float(r['list_imbalance']):.2f}" if r["list_imbalance"] else "-"
        lines.append(
            f"| {float(r['shift']):g} | {r['index']} | {r['state']} | {r['search']} "
            f"| {float(r['final_recall']):.3f} | {float(r['single_qps_median']):,.0f} "
            f"| {float(r['ndis_per_query']):,.0f} | {imb} "
            f"| {float(r['update_time_sec']):.1f} |")

    note = ("Stale = built on old data, new data added afterwards; rebuilt = "
            "trained and built on all data. Update time = adding the new vectors "
            "(stale) or the full rebuild.\n\n")
    (root / "tables" / "part4_drift.md").write_text(note + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
