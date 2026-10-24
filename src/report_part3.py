import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_part2 import COLORS, pick, read

TARGET = 0.95
SIZES = [100_000, 300_000, 1_000_000]
QUICK_SIZES = [10_000, 30_000, 100_000]
INDEXES = {"exact": "Exact", "hnsw": "HNSW", "ivfflat": "IVF-Flat"}


def rows_for(root, ix, n, full_n):
    path = root / "part3" / f"sift_{ix}_{n}.csv"
    if not path.exists() and n == full_n:
        # Largest-size HNSW / IVF-Flat runs are shared with Part 2.
        path = root / "part2" / f"sift_{ix}.csv"
    return read(path) if path.exists() else []


def slope(pts):
    """Log-log slope of QPS against N between the smallest and largest size."""
    (n0, q0), (n1, q1) = pts[0], pts[-1]
    return math.log(q1 / q0) / math.log(n1 / n0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    root = Path("results_quick" if args.quick else "results")
    sizes = QUICK_SIZES if args.quick else SIZES
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)

    points = {name: [] for name in INDEXES.values()}
    exact_qps = {}
    lines = ["| Index | Vectors | Setting | Recall | QPS | Speed-up vs exact "
             "| Build (s) |",
             "|---|---:|---|---:|---:|---:|---:|"]

    for ix, name in INDEXES.items():
        for n in sizes:
            rs = rows_for(root, ix, n, sizes[-1])
            r = pick(rs, TARGET) if rs else None
            if r is None:
                lines.append(f"| {name} | {n:,} | not reached / missing | | | | |")
                continue

            qps = float(r["single_qps_median"])
            if ix == "exact":
                exact_qps[n] = qps
            points[name].append((n, qps))
            speedup = qps / exact_qps[n] if n in exact_qps else float("nan")
            setting = ", ".join(s for s in (r["build"], r["search"]) if s not in ("", "-"))
            lines.append(
                f"| {name} | {n:,} | {setting or '-'} "
                f"| {float(r['final_recall']):.3f} | {qps:,.0f} "
                f"| {speedup:,.1f}x | {float(r['build_time_sec']):.1f} |")

    slope_lines = ["| Index | QPS at smallest N | QPS at largest N | Log-log slope |",
                   "|---|---:|---:|---:|"]
    for name, pts in points.items():
        if len(pts) >= 2:
            slope_lines.append(f"| {name} | {pts[0][1]:,.0f} | {pts[-1][1]:,.0f} "
                               f"| {slope(pts):.2f} |")

    fig, (ax_qps, ax_gain) = plt.subplots(1, 2, figsize=(10, 4))
    for name, pts in points.items():
        if not pts:
            continue
        ns, qs = zip(*pts)
        color = COLORS.get(name, "black")
        ax_qps.plot(ns, qs, marker="o", color=color, label=name)
        if name != "Exact":
            gain = [(n, q / exact_qps[n]) for n, q in pts if n in exact_qps]
            if gain:
                ax_gain.plot(*zip(*gain), marker="o", color=color, label=name)

    ax_qps.set(xscale="log", yscale="log", xlabel="Base vectors (N)",
               ylabel=f"QPS at recall ≥ {TARGET}", title="Speed at recall 0.95")
    ax_gain.set(xscale="log", yscale="log", xlabel="Base vectors (N)",
                ylabel="Speed-up over exact search", title="ANN advantage")
    for ax in (ax_qps, ax_gain):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("SIFT scaling (1 thread, final 5,000 queries)")
    fig.tight_layout()
    fig.savefig(root / "figures" / "part3_scaling.png", dpi=200)
    plt.close(fig)

    note = ("Fastest setting whose tuning recall ≥ 0.95; QPS and recall on the "
            "final 5,000 queries.\n\n")
    (root / "tables" / "part3_scaling.md").write_text(
        note + "\n".join(lines) + "\n\n" + "\n".join(slope_lines) + "\n")
    print("\n".join(lines), "\n")
    print("\n".join(slope_lines))


if __name__ == "__main__":
    main()
