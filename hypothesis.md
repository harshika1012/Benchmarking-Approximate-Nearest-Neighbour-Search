# Hypotheses

## Part 3: Scaling (SIFT, 100K / 300K / 1M vectors)

**Question.** How does the cost of reaching recall@10 ≥ 0.95 grow with the number
of base vectors N for exact search, HNSW and IVF-Flat?

**Setup.** The first N SIFT vectors, the same queries, ground truth recomputed exactly
for each N. Same build grids and selection rule as Part 2 (HNSW M ∈ {8,16,32},
IVF-Flat nlist ∈ {256,1024,4096}; fastest setting whose tuning recall ≥ 0.95).
The 1M results were already measured in Parts 1–2; these predictions are about the
*trend* across sizes.

**Hypothesis.** Every method gets slower as N grows, but at very different rates,
so the advantage of ANN over exact search grows with N.

1. **Exact:** QPS falls in proportion to 1/N (log-log slope ≈ −1), because every
   query reads the whole base from memory. From 26 QPS at 1M, I expect about 260 QPS
   at 100K and about 87 at 300K. At 100K the base (51 MB) nearly fits in the L3
   cache, so it may do slightly better than linear there.
2. **HNSW:** cost grows slowly, roughly with log N. The efSearch needed for 0.95
   stays the same or rises slightly, and QPS falls by less than 2× from 100K to 1M
   (slope between 0 and −0.3). Build time grows a little faster than N (N log N).
3. **IVF-Flat:** the best nlist grows with N and QPS falls roughly with √N (slope
   ≈ −0.5, about 3× from 100K to 1M), because both the number of cells and their
   size grow.
4. **Consequence:** the speed-up of HNSW and IVF-Flat over exact search grows with
   N; it will be smallest at 100K and largest at 1M.

## Part 4
