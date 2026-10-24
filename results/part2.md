# Benchmarking Approximate Nearest Neighbour Search

## Setup

All experiments ran on a single ADA cluster node (`gnode002`, Intel Xeon E5-2640 v4 @ 2.40 GHz, 40 logical CPUs, 50 MiB L3), pinned with `--nodelist` and run one job at a time. Software: FAISS 1.15.1 (CPU), NumPy 2.2.6, h5py 3.16.0, Python 3.10. All searches use **one thread** (`faiss.omp_set_num_threads(1)` and `OMP/OPENBLAS/MKL_NUM_THREADS=1`).

Data: SIFT (1,000,000 × 128, Euclidean) and GloVe-100 (1,183,514 × 100, angular) from ann-benchmarks. GloVe vectors and queries are L2-normalised and searched with Euclidean distance; the stored angular distances (1 − cos) are converted with L2 = √(2(1 − cos)). The 10,000 queries are split in half: settings are chosen on the **first 5,000 (tuning)**, and every reported number is measured on the **last 5,000 (final)**.

**Recall@10.** A returned vector counts as correct if its *true* distance to the query is no larger than the distance of the 10th ground-truth neighbour (relative tolerance 10⁻⁶ for float rounding). This accepts alternative neighbours tied at rank 10. True distances are recomputed from the returned IDs instead of taken from the index, because IVF-PQ returns approximate distances; scoring on those inflated IVF-PQ recall in an earlier version of the code (0.93 with 10-byte codes on GloVe). Missing results (ID −1) count as misses. Nine unit tests cover perfect, zero and partial recall, ties at rank k, results just beyond the cutoff, rounding noise, missing results and averaging over queries.

**Timing.** Queries per second (QPS) is measured one query per `search` call, over the 5,000 final queries, after a short warm-up; each measurement is repeated 3 times and the median is reported.

## Part 1: Exact baseline

| Dataset | Recall@10 (10k queries) | Identical top-10 ID set | Max rel. distance error | Single-query QPS | Batch QPS (5k queries) | Speed-up |
|---|---:|---:|---:|---:|---:|---:|
| SIFT | 1.000000 | 99.35% | 5.9 × 10⁻⁸ | 26.1 | 239.4 | 9.2× |
| GloVe-100 | 1.000000 | 100.00% | 1.7 × 10⁻⁷ | 27.4 | 260.0 | 9.5× |

FAISS `IndexFlatL2` agrees with the stored neighbours: recall is exactly 1.0 on all 10,000 queries of both datasets, and distances match to float32 precision. On SIFT, 0.65% of queries return a different top-10 ID set, yet recall is still 1.0. The swapped IDs therefore lie at exactly the 10th-neighbour distance. Such ties are common in SIFT because its coordinates are integers and it contains duplicate vectors, and they are the reason the recall function must be tie-aware.

**Why batching is ~9× faster.** A single query is a matrix-vector product: the whole base (512 MB for SIFT) is streamed from RAM once per query, with one multiply-add per value loaded, so speed is limited by memory bandwidth (512 MB / 38 ms ≈ 13 GB/s, about what one core can draw). A batch becomes a matrix-matrix product (BLAS GEMM) that reuses each cache-resident block of the base across many queries, so it is limited by arithmetic instead (≈ 61 GFLOP/s on both datasets, close to one core's AVX2 peak).

## Part 2: HNSW, IVF-Flat and IVF-PQ

**Grid.** For each build setting, the index is built once and the search parameter is swept upward until tuning recall reaches 0.995 or stops improving (gain < 0.001).

| Index | Build settings | Search parameter |
|---|---|---|
| HNSW (efConstruction = 200) | M ∈ {8, 16, 32} SIFT; {16, 32, 48} GloVe | efSearch 10 … 1536 |
| IVF-Flat | nlist ∈ {256, 1024, 4096} | nprobe 1 … 512 |
| IVF-PQ (nlist = 1024, 8 bits) | pq_m ∈ {16, 32, 64} SIFT; {10, 25, 50} GloVe | nprobe 1 … 512 |

![SIFT recall vs QPS](results/figures/part2_sift_recall_qps.png)
![GloVe-100 recall vs QPS](results/figures/part2_glove100_recall_qps.png)

*Figure 1. Recall@10 against single-query QPS (log scale) on the final 5,000 queries. Lines are the best trade-off curve of each index across all its build settings (points no other point beats on both recall and QPS); faint dots are all other measured settings. The star is exact search.*

### QPS at target recall

The setting is the fastest one whose **tuning** recall reaches the target; its QPS and recall are then reported on the **final** half. No interpolation is used.

| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 |
|---|---|---|---|
| SIFT | Exact | 26 (recall 1.000) | 26 (recall 1.000) |
| SIFT | HNSW | **8,513** (M=16, efSearch=32; 0.918) | **5,026** (M=16, efSearch=64; 0.971) |
| SIFT | IVF-Flat | 2,245 (nlist=4096, nprobe=32; 0.933) | 1,268 (nlist=4096, nprobe=64; 0.977) |
| SIFT | IVF-PQ | not reached (max 0.876) | not reached (max 0.876) |
| GloVe-100 | Exact | 27 (recall 1.000) | 27 (recall 1.000) |
| GloVe-100 | HNSW | **1,489** (M=48, efSearch=128; 0.909) | **660** (M=32, efSearch=384; 0.955) |
| GloVe-100 | IVF-Flat | 433 (nlist=1024, nprobe=64; 0.908) | 222 (nlist=4096, nprobe=512; 0.963) |
| GloVe-100 | IVF-PQ | not reached (max 0.866) | not reached (max 0.866) |

### Cost at recall 0.95

| Dataset | Index | Setting | Recall | Build (s) | Size (MB) | Bytes / vector |
|---|---|---|---:|---:|---:|---:|
| SIFT | Exact | – | 1.000 | 0.4 | 512.0 | 512.0 |
| SIFT | HNSW | M=16, efSearch=64 | 0.971 | 507.5 | 656.3 | 656.3 |
| SIFT | IVF-Flat | nlist=4096, nprobe=64 | 0.977 | 171.7 | 522.1 | 522.1 |
| SIFT | IVF-PQ | pq_m=64, nprobe=256 *(best; below 0.95)* | 0.880 | 31.0 | 72.7 | 72.7 |
| GloVe-100 | Exact | – | 1.000 | 0.3 | 473.4 | 400.0 |
| GloVe-100 | HNSW | M=32, efSearch=384 | 0.955 | 938.7 | 795.5 | 672.2 |
| GloVe-100 | IVF-Flat | nlist=4096, nprobe=512 | 0.963 | 157.9 | 484.5 | 409.4 |
| GloVe-100 | IVF-PQ | pq_m=50, nprobe=512 *(best; below 0.95)* | 0.866 | 27.0 | 69.2 | 58.4 |

Sizes come from the serialised FAISS index. IVF-Flat stores the raw vector plus an 8-byte ID; IVF-PQ stores pq_m bytes of code plus the 8-byte ID; HNSW stores the raw vector plus about 2·M four-byte graph links on the bottom layer.

### Why IVF-PQ does not reach high recall

IVF-PQ ranks candidates by their distance to a compressed code, not to the real vector. The compression error is larger than the gap between the 10th and 11th true neighbours, so the order near the cutoff is partly random. Raising nprobe only brings in more candidates scored with the same error, so recall levels off: at 0.876 on SIFT (64-byte codes) and 0.866 on GloVe (50-byte codes). The ceiling depends on code size: on GloVe, 10-byte codes stop near 0.31 and 25-byte codes near 0.61. Re-ranking the best candidates with exact distances would remove this ceiling.

## Discussion (Parts 1–2)

**Q1. Fastest index at recall 0.95.** HNSW is fastest on both datasets: 4.0× faster than IVF-Flat on SIFT (5,026 vs 1,268 QPS) and 3.0× on GloVe (660 vs 222 QPS). So the ranking does not change between datasets, but the cost does. HNSW is 7.6× slower on GloVe and needs efSearch 384 instead of 64. IVF-Flat must scan 12.5% of its lists on GloVe (512 of 4096) instead of 1.6% (64 of 4096). Both follow from GloVe's low relative contrast: a query's nearest neighbours are barely closer than the rest of the data. Greedy graph search is then easily led off course and needs a much wider beam, and the true neighbours are spread over many more IVF cells.

**Q2. Build time and memory.** Exact search is free to build (0.4 s). IVF-PQ takes about 30 s, IVF-Flat 160–170 s (mostly k-means with 4096 centroids), and HNSW 508–939 s, because every insertion runs a graph search with efConstruction = 200. On memory, IVF-Flat is almost the size of the raw data (+8 bytes per vector for IDs). HNSW adds 28% on SIFT (M=16) and 68% on GloVe (M=32) for its graph. PQ makes the index 7–9× smaller than the raw vectors (72.7 vs 512 bytes per vector on SIFT, 58.4 vs 400 on GloVe) and builds fastest. In exchange it gives up high recall: it cannot reach 0.90 on either dataset without re-ranking. It is also not faster than IVF-Flat at equal recall here: each candidate costs one table lookup per sub-vector at scattered memory addresses, whereas IVF-Flat streams contiguous floats that SIMD (vector) instructions process quickly.

**Q4 (exact-search input).** Exact search costs about 38 ms per query per million vectors on this machine and grows linearly with the number of vectors. At 150,000 vectors that is about 6 ms per query, well within a 100 ms budget, so exact search is enough. At 10 million vectors it is about 380 ms per query, so ANN is needed even though the 5.1 GB of raw data fits in 16 GB of RAM.

**Q5 (one way this benchmark can mislead).** The QPS-at-target table depends on how fine the search grid is. HNSW with M=48 and efSearch=256 reaches 0.949 at 807 QPS, just below the 0.95 bar, so the table reports M=32 at 660 QPS. A finer efSearch grid would likely show M=48 reaching 0.95 at a higher QPS. This was observed in the data but not tested with a finer grid.
