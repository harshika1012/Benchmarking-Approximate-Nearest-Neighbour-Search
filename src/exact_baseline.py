import argparse
import csv
import os
import platform
import socket
import time

import faiss
import h5py
import numpy as np

from recall import recall_at_k, true_distances

K = 10
N_TUNE = 5000
REPEATS = 3

DATASETS = {
    "sift": ("SIFT", "data/sift/sift-128-euclidean.hdf5"),
    "glove": ("GloVe-100", "data/glove/glove-100-angular.hdf5"),
}


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def load(dataset):
    name, path = DATASETS[dataset]
    with h5py.File(path, "r") as f:
        train = f["train"][:].astype(np.float32)
        queries = f["test"][:].astype(np.float32)
        gt_ids = f["neighbors"][:, :K].astype(np.int64)
        gt_dist = f["distances"][:, :K].astype(np.float64)

    if dataset == "glove":
        # Stored distances are 1 - cos; on unit vectors L2 = sqrt(2 (1 - cos)).
        train = normalize(train)
        queries = normalize(queries)
        gt_dist = np.sqrt(np.maximum(2.0 * gt_dist, 0.0))

    return name, train, queries, gt_ids, gt_dist


def median_single_query_qps(index, queries, repeats=REPEATS):
    """Median QPS over repeated passes, one query per search call."""
    for q in queries[:10]:
        index.search(q.reshape(1, -1), K)

    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for q in queries:
            index.search(q.reshape(1, -1), K)
        values.append(len(queries) / (time.perf_counter() - start))
    return float(np.median(values))


def median_batch_time(index, queries, repeats=REPEATS):
    """Median wall time to search all queries in one call."""
    index.search(queries[:100], K)

    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(queries, K)
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    faiss.omp_set_num_threads(1)

    name, train, queries, gt_ids, gt_dist = load(args.dataset)
    final_q = queries[N_TUNE:]
    print(f"{name}: {train.shape[0]:,} vectors, dim {train.shape[1]}", flush=True)

    index = faiss.IndexFlatL2(train.shape[1])
    start = time.perf_counter()
    index.add(train)
    build_time = time.perf_counter() - start

    # Validate against the stored neighbours on all 10,000 queries.
    _, I = index.search(queries, K)
    D = true_distances(train, queries, I)

    recall = recall_at_k(I, D, gt_ids, gt_dist, K)
    set_match = float(np.mean([set(a) == set(b) for a, b in zip(I, gt_ids)]))
    pos_match = float(np.mean(I == gt_ids))
    max_rel_err = float(np.max(
        np.abs(D - gt_dist) / np.maximum(gt_dist, 1e-12)
    ))

    print(f"Recall@{K} (all {len(queries)} queries): {recall:.6f}", flush=True)
    print(f"Queries with identical top-{K} id set: {set_match:.4%}", flush=True)
    print(f"Positions with identical id: {pos_match:.4%}", flush=True)
    print(f"Max relative distance error vs file: {max_rel_err:.2e}", flush=True)

    # Timing on the held-out final 5,000 queries.
    single_qps = median_single_query_qps(index, final_q)
    batch_time = median_batch_time(index, final_q)
    batch_qps = len(final_q) / batch_time

    print(f"Single-query QPS (median of {REPEATS}): {single_qps:.2f}", flush=True)
    print(f"Batch time (median of {REPEATS}): {batch_time:.3f}s "
          f"-> {batch_qps:.2f} QPS", flush=True)
    print(f"Batch speed-up: {batch_qps / single_qps:.1f}x", flush=True)

    row = {
        "dataset": name,
        "index": "Exact (IndexFlatL2)",
        "n_base": train.shape[0],
        "dim": train.shape[1],
        "build_time_sec": build_time,
        "serialized_index_size_bytes": train.nbytes,
        "bytes_per_vector": train.nbytes / train.shape[0],
        "validation_queries": len(queries),
        "recall_at_10": recall,
        "topk_set_match_rate": set_match,
        "position_match_rate": pos_match,
        "max_rel_dist_error": max_rel_err,
        "timing_queries": len(final_q),
        "single_qps_median": single_qps,
        "batch_time_sec_median": batch_time,
        "batch_qps": batch_qps,
        "threads": 1,
        "repeats": REPEATS,
        "hostname": socket.gethostname(),
        "cpu": cpu_model(),
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
    }

    out = f"results/exact_{args.dataset}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
