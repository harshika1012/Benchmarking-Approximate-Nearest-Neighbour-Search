import time
import h5py
import faiss
import numpy as np

from recall import recall_at_k

DATA_PATH = "data/sift/sift-128-euclidean.hdf5"
K = 10
N_CHECK = 100


def main():
    with h5py.File(DATA_PATH, "r") as f:
        train = f["train"][:]
        queries = f["test"][:]
        gt_ids = f["neighbors"][:]
        gt_distances = f["distances"][:]

    # Use one thread for reproducible CPU timings.
    faiss.omp_set_num_threads(1)

    index = faiss.IndexFlatL2(train.shape[1])

    start = time.perf_counter()
    index.add(train)
    build_time = time.perf_counter() - start

    print(f"Index build time: {build_time:.3f} seconds")
    print(f"Index size: {index.ntotal:,} vectors")

    # Validate exact search on 100 queries.
    D, I = index.search(queries[:N_CHECK], K)

    # FAISS IndexFlatL2 returns squared Euclidean distances.
    D = np.sqrt(np.maximum(D, 0))

    recall = recall_at_k(
        I, D, gt_ids[:N_CHECK, :K], gt_distances[:N_CHECK, :K], K
    )
    print(f"Recall@{K} on first {N_CHECK} queries: {recall:.4f}")

    # Single-query timing.
    times = []
    for q in queries:
        start = time.perf_counter()
        index.search(q.reshape(1, -1), K)
        times.append(time.perf_counter() - start)

    single_qps = len(queries) / sum(times)
    print(f"Single-query throughput: {single_qps:.2f} queries/sec")

    # Batch timing.
    start = time.perf_counter()
    index.search(queries, K)
    batch_time = time.perf_counter() - start

    batch_qps = len(queries) / batch_time
    print(f"Batch search time: {batch_time:.3f} seconds")
    print(f"Batch throughput: {batch_qps:.2f} queries/sec")


if __name__ == "__main__":
    main()