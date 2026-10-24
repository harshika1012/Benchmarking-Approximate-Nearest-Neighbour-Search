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

## Part 4: Drift (synthetic data)

**Question.** If an index is built on one distribution and new vectors and queries then
arrive from a shifted one, which index degrades, how, and does rebuilding fix it?

**Data.** Seeded generator (`src/drift_data.py`): 64-d mixture of 100 Gaussians
(centres ~ N(0, 1) per dimension, within-cluster std 0.5). 200K "old" vectors; 200K
"new" vectors and 10K queries from the same mixture translated by shift·√64 along a
fixed random direction. Varied property: shift ∈ {0 (control), 0.5, 1, 2}. Everything
else (seeds, sizes, index parameters, search setting) is held fixed.

**Indexes.** HNSW (M=16), IVF-Flat (nlist=1024), IVF-PQ (nlist=1024, 32 bytes).
"Stale" = built on old data, new data added afterwards. "Rebuilt" = trained and built on
all data. The search setting is tuned once on the control and then held fixed.

**Hypothesis.**
1. **IVF-Flat (stale):** the centroids only cover the old data, so the new vectors pile
   into the few old cells nearest to the shifted cloud. List imbalance and distance
   computations per query rise sharply with shift, so QPS at the fixed nprobe falls
   several-fold at shift ≥ 1. Recall stays about the same or even rises, because those
   bloated cells contain most of each query's neighbours. The cost is in speed, not
   accuracy.
2. **IVF-PQ (stale):** the PQ codebooks were trained on residuals (vector minus its
   centroid) of the old data. New vectors lie far from every centroid, so their
   residuals are much larger than anything the codebooks saw, the quantisation error
   grows, and recall drops sharply with shift. Speed also drops, as for IVF-Flat.
3. **HNSW (stale):** each new vector is linked to its nearest neighbours as it is
   inserted, so the graph adapts. Recall at the fixed efSearch stays within about 0.02
   of the control and QPS barely changes at every shift.
4. **Rebuilding** brings IVF-Flat's speed and IVF-PQ's recall back to control levels,
   and makes no meaningful difference for HNSW. At shift 0, stale and rebuilt are
   about equal for every index.

**What would refute it:** stale IVF-Flat losing recall rather than speed; IVF-PQ recall
not falling with shift; HNSW recall falling by more than 0.02; or rebuilding not
restoring IVF.

