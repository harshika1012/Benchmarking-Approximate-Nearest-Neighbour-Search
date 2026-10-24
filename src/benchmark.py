import argparse
import csv
import os
import socket
import time

import faiss
import numpy as np

from exact_baseline import K, N_TUNE, REPEATS, load, median_single_query_qps
from recall import recall_at_k, true_distances

EF_CONSTRUCTION = 200
NBITS = 8
QUICK_N_BASE = 100_000
QUICK_N_TIMING = 1000

# Stop a sweep once tuning recall is this high, or stops improving.
STOP_RECALL = 0.995
MIN_GAIN = 0.001

EF_SEARCH = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]
NPROBE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

LABELS = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ"}


def build_grid(dataset, index_name, quick):
    if index_name == "hnsw":
        ms = [8, 16, 32] if dataset == "sift" else [16, 32, 48]
        return [{"M": m} for m in (ms[:2] if quick else ms)]
    if index_name == "ivfflat":
        nlists = [256, 1024] if quick else [256, 1024, 4096]
        return [{"nlist": n} for n in nlists]
    # pq_m must divide the dimension (128 for SIFT, 100 for GloVe).
    pq_ms = [16, 32, 64] if dataset == "sift" else [10, 25, 50]
    return [{"nlist": 1024, "pq_m": m} for m in (pq_ms[:2] if quick else pq_ms)]


def build_index(index_name, d, p):
    if index_name == "hnsw":
        index = faiss.IndexHNSWFlat(d, p["M"])
        index.hnsw.efConstruction = EF_CONSTRUCTION
        return index
    quantizer = faiss.IndexFlatL2(d)
    if index_name == "ivfflat":
        return faiss.IndexIVFFlat(quantizer, d, p["nlist"])
    return faiss.IndexIVFPQ(quantizer, d, p["nlist"], p["pq_m"], NBITS)


def search_values(index_name, p):
    if index_name == "hnsw":
        return EF_SEARCH
    return [n for n in NPROBE if n <= p["nlist"]]


def set_search(index, index_name, value):
    if index_name == "hnsw":
        index.hnsw.efSearch = value
    else:
        index.nprobe = value


def load_data(dataset, n_base=None):
    name, train, queries, gt_ids, gt_dist = load(dataset)
    if n_base:
        # Stored neighbours refer to the full base, so recompute them.
        train = train[:n_base]
        flat = faiss.IndexFlatL2(train.shape[1])
        flat.add(train)
        _, gt_ids = flat.search(queries, K)
        gt_dist = true_distances(train, queries, gt_ids)
    return name, train, queries, gt_ids, gt_dist


def score(index, train, queries, gt_ids, gt_dist):
    """Recall@K scored on true distances, not the index's own estimates."""
    _, I = index.search(queries, K)
    return recall_at_k(I, true_distances(train, queries, I), gt_ids, gt_dist, K)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sift", "glove"], required=True)
    parser.add_argument("--index", choices=list(LABELS), required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)
    out_dir = "results_quick/part2" if args.quick else "results/part2"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.dataset}_{args.index}.csv"

    name, train, queries, gt_ids, gt_dist = load_data(
        args.dataset, QUICK_N_BASE if args.quick else None
    )
    tune_q, final_q = queries[:N_TUNE], queries[N_TUNE:]
    tune_ids, final_ids = gt_ids[:N_TUNE], gt_ids[N_TUNE:]
    tune_dist, final_dist = gt_dist[:N_TUNE], gt_dist[N_TUNE:]
    timing_q = final_q[:QUICK_N_TIMING] if args.quick else final_q
    param = "efSearch" if args.index == "hnsw" else "nprobe"

    rows = []
    for p in build_grid(args.dataset, args.index, args.quick):
        print(f"\nBuilding {LABELS[args.index]} {p} on {len(train):,} vectors",
              flush=True)
        index = build_index(args.index, train.shape[1], p)

        start = time.perf_counter()
        if not index.is_trained:
            index.train(train)
        index.add(train)
        build_time = time.perf_counter() - start
        size = faiss.serialize_index(index).nbytes
        print(f"build={build_time:.1f}s size={size / 1e6:.1f} MB", flush=True)

        prev = 0.0
        for value in search_values(args.index, p):
            set_search(index, args.index, value)
            tune_recall = score(index, train, tune_q, tune_ids, tune_dist)
            final_recall = score(index, train, final_q, final_ids, final_dist)
            qps = median_single_query_qps(index, timing_q)

            rows.append({
                "dataset": name,
                "index": LABELS[args.index],
                "build": ",".join(f"{k}={v}" for k, v in p.items()),
                "search": f"{param}={value}",
                "M": p.get("M", ""),
                "efConstruction": EF_CONSTRUCTION if args.index == "hnsw" else "",
                "nlist": p.get("nlist", ""),
                "pq_m": p.get("pq_m", ""),
                "nbits": NBITS if args.index == "ivfpq" else "",
                "search_param": param,
                "search_value": value,
                "n_base": len(train),
                "tune_recall": tune_recall,
                "final_recall": final_recall,
                "single_qps_median": qps,
                "timing_queries": len(timing_q),
                "repeats": REPEATS,
                "build_time_sec": build_time,
                "index_size_bytes": size,
                "bytes_per_vector": size / len(train),
                "hostname": socket.gethostname(),
                "faiss_version": faiss.__version__,
            })
            write_csv(out_path, rows)
            print(f"{param}={value}: tune={tune_recall:.4f} "
                  f"final={final_recall:.4f} qps={qps:.1f}", flush=True)

            if tune_recall >= STOP_RECALL or tune_recall - prev < MIN_GAIN:
                break
            prev = tune_recall

        del index

    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
