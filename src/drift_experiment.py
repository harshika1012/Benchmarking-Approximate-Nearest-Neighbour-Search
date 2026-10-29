import argparse
import os
import socket
import time

import faiss
import numpy as np

from benchmark import (
    EF_CONSTRUCTION, EF_SEARCH, NBITS, NPROBE, score, set_search, write_csv,
)
from drift_data import D, make_drift_data
from exact_baseline import K, N_TUNE, REPEATS, median_single_query_qps
from recall import true_distances

SHIFTS = [0.0, 0.5, 1.0, 2.0]  # the first entry is the control
TARGET = 0.95
N_QUERIES = 10_000
HNSW_M = 16
PQ_M = 32
FULL = dict(n_old=200_000, n_new=200_000, nlist=1024, n_timing=5000)
QUICK = dict(n_old=25_000, n_new=25_000, nlist=256, n_timing=1000)
INDEXES = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ"}


def new_index(name, nlist):
    if name == "hnsw":
        index = faiss.IndexHNSWFlat(D, HNSW_M)
        index.hnsw.efConstruction = EF_CONSTRUCTION
        return index
    quantizer = faiss.IndexFlatL2(D)
    if name == "ivfflat":
        return faiss.IndexIVFFlat(quantizer, D, nlist)
    return faiss.IndexIVFPQ(quantizer, D, nlist, PQ_M, NBITS)


def build(name, nlist, vectors):
    index = new_index(name, nlist)
    start = time.perf_counter()
    if not index.is_trained:
        index.train(vectors)
    index.add(vectors)
    return index, time.perf_counter() - start


def copy_index(index):
    return faiss.deserialize_index(faiss.serialize_index(index))


def search_values(name, nlist):
    return EF_SEARCH if name == "hnsw" else [n for n in NPROBE if n <= nlist]


def ndis_per_query(index, name, queries):
    """Distance computations per query, from FAISS's own counters."""
    stats = faiss.cvar.hnsw_stats if name == "hnsw" else faiss.cvar.indexIVF_stats
    stats.reset()
    index.search(queries, K)
    return getattr(stats, "ndis", float("nan")) / len(queries)


def imbalance(index):
    """Sum of squared list sizes relative to perfectly even lists (= 1.0)."""
    sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)],
                     dtype=np.float64)
    return float((sizes ** 2).sum() * len(sizes) / sizes.sum() ** 2)


def choose_setting(index, name, nlist, vectors, queries, gt_ids, gt_dist):
    """Smallest setting reaching TARGET tuning recall, else the most accurate."""
    best = (None, -1.0)
    for value in search_values(name, nlist):
        set_search(index, name, value)
        recall = score(index, vectors, queries, gt_ids, gt_dist)
        if recall >= TARGET:
            return value
        if recall > best[1]:
            best = (value, recall)
    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = QUICK if args.quick else FULL
    faiss.omp_set_num_threads(1)
    out_dir = "results_quick/part4" if args.quick else "results/part4"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/drift.csv"

    bases, settings, rows = {}, {}, []
    for shift in SHIFTS:
        old, new, queries = make_drift_data(
            shift, cfg["n_old"], cfg["n_new"], N_QUERIES
        )
        data = np.concatenate([old, new])
        shuffled = data[np.random.default_rng(0).permutation(len(data))]

        flat = faiss.IndexFlatL2(D)
        flat.add(data)
        _, gt_ids = flat.search(queries, K)
        gt_dist = true_distances(data, queries, gt_ids)
        tune = (queries[:N_TUNE], gt_ids[:N_TUNE], gt_dist[:N_TUNE])
        final = (queries[N_TUNE:], gt_ids[N_TUNE:], gt_dist[N_TUNE:])
        timing_q = final[0][:cfg["n_timing"]]
        print(f"\n=== shift={shift}: {len(data):,} vectors ===", flush=True)

        for name, label in INDEXES.items():
            # Old data is identical for every shift, so build each base once.
            if name not in bases:
                bases[name] = build(name, cfg["nlist"], old)
            base, base_time = bases[name]

            stale = copy_index(base)
            start = time.perf_counter()
            stale.add(new)
            add_time = time.perf_counter() - start

            rebuilt, rebuild_time = build(name, cfg["nlist"], shuffled)

            # The search setting is tuned once, on the control, then held fixed.
            if name not in settings:
                settings[name] = choose_setting(
                    stale, name, cfg["nlist"], data, *tune
                )
            value = settings[name]
            param = "efSearch" if name == "hnsw" else "nprobe"

            for state, index, vectors, cost in (
                ("stale", stale, data, add_time),
                ("rebuilt", rebuilt, shuffled, rebuild_time),
            ):
                set_search(index, name, value)
                row = {
                    "experiment": "drift",
                    "shift": shift,
                    "index": label,
                    "state": state,
                    "search": f"{param}={value}",
                    "n_old": len(old),
                    "n_new": len(new),
                    "tune_recall": score(index, vectors, *tune),
                    "final_recall": score(index, vectors, *final),
                    "single_qps_median": median_single_query_qps(index, timing_q),
                    "ndis_per_query": ndis_per_query(index, name, final[0]),
                    "list_imbalance": imbalance(index) if name != "hnsw" else "",
                    # Time to add the new vectors (stale) or to rebuild everything.
                    "update_time_sec": cost,
                    "base_build_time_sec": base_time if state == "stale" else "",
                    "timing_queries": len(timing_q),
                    "repeats": REPEATS,
                    "hostname": socket.gethostname(),
                    "faiss_version": faiss.__version__,
                }
                rows.append(row)
                write_csv(out_path, rows)
                print(f"{label:8s} {state:7s} {param}={value}: "
                      f"recall={row['final_recall']:.4f} "
                      f"qps={row['single_qps_median']:.0f} "
                      f"ndis={row['ndis_per_query']:.0f} "
                      f"imbalance={row['list_imbalance']} "
                      f"update={cost:.1f}s", flush=True)

            del stale, rebuilt

    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
