## Part 3: Scaling on SIFT

**Hypothesis** : Every method gets slower as N grows, but at different rates, so ANN's advantage over exact search grows with N. Predicted log-log slopes of QPS@0.95 against N: exact ≈ −1; HNSW between 0 and −0.3 (≈ log N); IVF-Flat ≈ −0.5 (≈ √N).

**Method.** First N ∈ {100K, 300K, 1M} SIFT vectors, same queries, ground truth recomputed exactly for each N. Same grids and selection rule as Part 2 (fastest setting whose tuning recall ≥ 0.95, reported on the final 5,000 queries). The 1M HNSW and IVF-Flat runs are the Part 2 runs (same code, grid and machine).

![SIFT scaling](results/figures/part3_scaling.png)

*Figure 2. Left: QPS at recall ≥ 0.95 against N (log-log). Right: speed-up over exact search.*

| Index | N | Setting | Recall | QPS | Speed-up | Build (s) |
|---|---:|---|---:|---:|---:|---:|
| Exact | 100K / 300K / 1M | – | 1.000 | 259 / 87 / 26 | 1× | < 1 |
| HNSW | 100K | M=16, efSearch=32 | 0.958 | 13,127 | 51× | 27 |
| HNSW | 300K | M=32, efSearch=32 | 0.956 | 7,914 | 91× | 141 |
| HNSW | 1M | M=16, efSearch=64 | 0.971 | 5,026 | 193× | 508 |
| IVF-Flat | 100K | nlist=1024, nprobe=32 | 0.965 | 6,489 | 25× | 4.5 |
| IVF-Flat | 300K | nlist=4096, nprobe=64 | 0.970 | 2,964 | 34× | 52 |
| IVF-Flat | 1M | nlist=4096, nprobe=64 | 0.977 | 1,268 | 49× | 172 |

Measured log-log slopes (100K → 1M): **exact −1.00, HNSW −0.42, IVF-Flat −0.71.**

**Result.** The hypothesis was **partly supported**. Exact search scales exactly as 1/N (259 → 87 → 26 QPS, slope −1.00), because every query reads the whole base. ANN's advantage does grow with N: the HNSW speed-up rises from 51× to 193×, and IVF-Flat's from 25× to 49×. But both ANN indexes slowed down more than predicted.

- **HNSW (−0.42, not ≥ −0.3).** Two effects add up. Reaching 0.95 needs twice the beam at 1M (efSearch 64 vs 32). Each step also gets slower: at a fixed M=16, efSearch=32, QPS drops 1.5× from 100K to 1M. The 66 MB index at 100K stays partly in cache, while at 1M (656 MB) almost every neighbour visit is a random read from main memory. The log N argument counts graph hops but ignores that each hop becomes slower.
- **IVF-Flat (−0.71, not −0.5).** Work per query is nlist centroid comparisons plus nprobe·N/nlist scanned vectors: 4,149 → 8,784 → 19,721 distance computations across the three sizes. That is a 4.75× increase, matching the 5.1× QPS drop. The √N argument assumes nprobe stays fixed. Here it doubled (32 → 64) once cells became finer, because a query's neighbours then spread over more cells. With nlist capped at 4096, list sizes grow in step with N between 300K and 1M.

**Implication.** Headline numbers measured on a small subset overstate absolute speed at scale. Going from 100K to 1M, HNSW is 2.6× slower and IVF-Flat 5.1× slower, and the efSearch or nprobe chosen at 100K would not reach 0.95 at 1M. Extrapolating along the measured slopes is safer than assuming textbook complexity. Caveats: the chosen HNSW M differs between sizes (16, 32, 16), and the coarse efSearch grid (32 → 48 → 64) adds error to each point.
