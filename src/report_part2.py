import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = [0.90, 0.95]
DATASETS = ["SIFT", "GloVe-100"]
INDEXES = ["HNSW", "IVF-Flat", "IVF-PQ"]
COLORS = {"HNSW": "tab:blue", "IVF-Flat": "tab:orange", "IVF-PQ": "tab:green"}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pareto(points):
    """Points that no other point beats on both recall and QPS."""
    front = []
    for recall, qps in sorted(points, key=lambda p: (-p[0], -p[1])):
        if not front or qps > front[-1][1]:
            front.append((recall, qps))
    return front[::-1]


def pick(rows, target):
    """Fastest setting whose tuning-half recall reaches the target."""
    ok = [r for r in rows if float(r["tune_recall"]) >= target]
    return max(ok, key=lambda r: float(r["single_qps_median"]), default=None)


def describe(r):
    return (f"{float(r['single_qps_median']):,.0f} "
            f"({r['build']}, {r['search']}; recall {float(r['final_recall']):.3f})")


def plot(dataset, rows, exact, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in INDEXES:
        pts = [(float(r["final_recall"]), float(r["single_qps_median"]))
               for r in rows if r["index"] == name]
        if not pts:
            continue
        ax.scatter(*zip(*pts), s=10, alpha=0.25, color=COLORS[name])
        ax.plot(*zip(*pareto(pts)), marker="o", ms=4,
                color=COLORS[name], label=name)

    if exact:
        ax.scatter([float(exact["recall_at_10"])],
                   [float(exact["single_qps_median"])],
                   marker="*", s=150, color="black", zorder=3,
                   label="Exact (IndexFlatL2)")

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10 (final 5,000 queries)")
    ax.set_ylabel("Queries per second (1 thread, 1 query at a time)")
    ax.set_title(f"{dataset}: recall vs speed")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    name = dataset.lower().replace("-", "")
    fig.savefig(fig_dir / f"part2_{name}_recall_qps.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = Path("results_quick" if args.quick else "results")
    fig_dir, tab_dir = root / "figures", root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    exact_rows = [r for p in sorted(root.glob("exact_*.csv")) for r in read(p)]
    part2_rows = [r for p in sorted((root / "part2").glob("*.csv")) for r in read(p)]
    exact = {r["dataset"]: r for r in exact_rows}

    qps_lines = ["| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 |",
                 "|---|---|---|---|"]
    cost_lines = ["| Dataset | Index | Setting | Recall | Build (s) "
                  "| Size (MB) | Bytes/vector |",
                  "|---|---|---|---|---:|---:|---:|"]

    for ds in DATASETS:
        ds_rows = [r for r in part2_rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        plot(ds, ds_rows, exact.get(ds), fig_dir)

        if ds in exact:
            e = exact[ds]
            qps = f"{float(e['single_qps_median']):,.0f} (recall 1.000)"
            qps_lines.append(f"| {ds} | Exact | {qps} | {qps} |")
            cost_lines.append(
                f"| {ds} | Exact | - | 1.000 | {float(e['build_time_sec']):.1f} "
                f"| {int(e['serialized_index_size_bytes']) / 1e6:.1f} "
                f"| {float(e['bytes_per_vector']):.1f} |")

        for name in INDEXES:
            rs = [r for r in ds_rows if r["index"] == name]
            if not rs:
                continue
            best_tune = max(float(r["tune_recall"]) for r in rs)

            cells = []
            for t in TARGETS:
                r = pick(rs, t)
                cells.append(describe(r) if r
                             else f"not reached (max {best_tune:.3f})")
            qps_lines.append(f"| {ds} | {name} | " + " | ".join(cells) + " |")

            r = pick(rs, 0.95)
            note = ""
            if r is None:
                r = max(rs, key=lambda x: float(x["tune_recall"]))
                note = " (below 0.95)"
            cost_lines.append(
                f"| {ds} | {name} | {r['build']}, {r['search']}{note} "
                f"| {float(r['final_recall']):.3f} "
                f"| {float(r['build_time_sec']):.1f} "
                f"| {int(r['index_size_bytes']) / 1e6:.1f} "
                f"| {float(r['bytes_per_vector']):.1f} |")

    note = ("Setting chosen on the tuning half (first 5,000 queries); "
            "QPS and recall reported on the final 5,000.\n\n")
    (tab_dir / "part2_qps_at_recall.md").write_text(note + "\n".join(qps_lines) + "\n")
    (tab_dir / "part2_cost_at_095.md").write_text(note + "\n".join(cost_lines) + "\n")

    # One row per benchmark run behind a reported number.
    all_rows = ([dict(part="1", **r) for r in exact_rows]
                + [dict(part="2", **r) for r in part2_rows])
    columns = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(root / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n".join(qps_lines), "\n")
    print("\n".join(cost_lines))


if __name__ == "__main__":
    main()
