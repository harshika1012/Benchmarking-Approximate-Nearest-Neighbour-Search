## Part 4: Drift on synthetic data

**What I changed.** A seeded generator (`src/drift_data.py`) produces a 64-d mixture of 100 Gaussian clusters. Each index is built on 200K "old" vectors; then 200K "new" vectors are added, drawn from the same mixture translated by a distance *shift* (in units of the spread of the cluster centres) along a fixed random direction. Queries (10K, split 5K/5K) come from the shifted distribution. The only varied property is shift ∈ {0 (control), 0.5, 1, 2}. Each index is tested **stale** (built on old data, new vectors added afterwards; IVF keeps its old centroids and PQ its old codebooks) and **rebuilt** (trained and built from scratch on all 400K vectors). The search setting is chosen once on the control (smallest with tuning recall ≥ 0.95) and then held fixed: HNSW M=16 efSearch=96; IVF-Flat nlist=1024 nprobe=16; IVF-PQ nlist=1024, 32 bytes, nprobe=32.

![Drift](results/figures/part4_drift.png)

*Figure 3. Recall, QPS and distance computations per query against shift, at the fixed search setting. Solid = stale, dashed = rebuilt.*

**What happened.**

| Shift | IVF-PQ recall (stale / rebuilt) | IVF-Flat QPS (stale / rebuilt) | IVF list imbalance (stale) | HNSW recall (stale / rebuilt) |
|---:|---|---|---:|---|
| 0 | 0.856 / 0.853 | 5,740 / 5,944 | 1.18 | 0.972 / 0.974 |
| 0.5 | 0.718 / 0.860 | 5,817 / 6,167 | 1.64 | 0.965 / 0.989 |
| 1 | 0.433 / 0.862 | 5,222 / 5,992 | 2.20 | 0.946 / 0.992 |
| 2 | **0.180** / 0.860 | **3,641** / 6,096 | 3.17 | 0.992 / 0.994 |

**Mechanism.**
- **IVF-PQ** fails silently. Its codebooks were trained on residuals (vector minus nearest centroid) of the old data. New vectors lie far from every centroid, so their residuals are much larger than anything the codebooks saw, and the quantisation error swamps the distances between neighbours. Recall falls from 0.86 to 0.18 with no change to the index or the search setting.
- **IVF-Flat** keeps its recall (≥ 0.989) because it compares full vectors, but the stale centroids send new vectors into the few cells nearest the shifted cloud. List imbalance rises from 1.18 to 3.17, vectors scanned per query double (6.4K → 13.6K), and QPS falls 1.6×.
- **HNSW** adapts because each new vector is linked to its nearest neighbours on insertion. Its worst case is partial overlap (shift 1: 0.946 vs 0.992 rebuilt), not full separation (shift 2: 0.992). At shift 2 the new region is effectively a fresh graph. At shift 1 the new points join old neighbourhoods whose neighbour lists were already full and pruned, which weakens the links between old and new regions; a shuffled rebuild removes the gap.

Rebuilding fully restores both IVF indexes (IVF-PQ back to 0.86 recall, IVF-Flat back to about 6,000 QPS with imbalance 1.27) and is cheap. Adding the new vectors takes 0.7 s versus about 10 s for a full rebuild, and HNSW insertion is only about 2× cheaper than an HNSW rebuild.

**Hypothesis.** Partly supported. IVF-PQ's recall collapse, IVF-Flat losing speed rather than recall, and recovery by rebuilding were all predicted. I overestimated the IVF-Flat slowdown ("several-fold"; observed 1.6×), because the new data still spreads over a few hundred cells rather than a handful. HNSW stayed within 0.02 of the control except at shift 1 (−0.026), and, contrary to my prediction, rebuilding HNSW does help at that shift.

**What I did not test.** Queries from the *old* distribution after drift; shifts by rotation or by new clusters appearing; deletions; measuring residual norms directly (the PQ explanation is inferred, not measured); separating the drift effect from the change in data density around queries (at shift 0 all 400K points share one region, which is why rebuilt HNSW recall rises with shift); and a single timing per configuration on one machine.


**Q3. What Parts 3–4 show, and when headline numbers mislead.** Part 3 shows that numbers measured on a subset overstate speed at scale: from 100K to 1M vectors, HNSW at recall 0.95 slows 2.6× and IVF-Flat 5.1×, faster than textbook log N and √N scaling, because the beam or probe count needed grows and the index stops fitting in cache. Settings tuned on a small sample would not reach 0.95 at full size. Part 4 shows that numbers measured on static data overstate quality under drift: an IVF-PQ index tuned to 0.86 recall drops to 0.18 once new data arrives from a shifted distribution, with no visible error, while IVF-Flat keeps its recall but slows down 1.6×. Headline recall and QPS are therefore only valid for the data size and the data distribution they were measured on. Production benchmarks should measure at the target scale, re-measure recall on recent queries, and retrain IVF centroids or codebooks when the data moves.


**Q4 (recommendation).** At 10M vectors, exact search is ruled out: extending the
measured slope of −1.00 gives about 385 ms per query, well over the 100 ms budget.
Memory then decides which ANN indexes are possible. From the Part 2 cost table, HNSW
(M=16) needs about 656 B per vector (≈6.6 GB at 10M), IVF-Flat about 525 B (≈5.3 GB)
and IVF-PQ about 47 B (≈0.5 GB). All three fit in 16 GB. However, IVF-PQ never reached
recall 0.95 on either dataset (maximum 0.876 on SIFT and 0.866 on GloVe), because the
quantisation error sets a ceiling that nprobe cannot remove.

I recommend **HNSW**. At 1M it was 4× faster than IVF-Flat at recall 0.95 (5,026 vs
1,268 QPS on SIFT; 660 vs 222 on GloVe). Its QPS also fell more slowly with size
(slope −0.42 vs −0.71), so extrapolating gives roughly 0.5 ms per query at 10M,
far inside the budget. Part 4 also showed it was the most robust to drift: a stale
HNSW kept recall ≥ 0.95 at every shift but one (0.946), while stale IVF-PQ fell
to 0.18.
The costs are build time (507 s for 1M SIFT single-threaded, so roughly 1.5–2 h at
10M) and the highest memory use. GloVe-like data needs a larger graph and search
budget (M=32, efSearch≈384).

**IVF-Flat** is the better choice if the index must be rebuilt often or memory is
tighter. It builds about 3× faster, adding vectors takes under a second, and it still
meets the budget by a wide margin. But it must be retrained when the data
distribution shifts (Part 4: list imbalance rose from 1.18 to 3.17). **IVF-PQ** only
makes sense if memory is far below 16 GB and recall around 0.85 is acceptable,
or if its candidates are re-ranked with exact distances.

These 10M figures are extrapolations from 100K–1M, not measurements. efSearch and
nprobe should be re-tuned on held-out queries at the real size.


**Q5 (second way this benchmark can mislead): run-to-run noise on a shared machine.**
All timings come from one node of a shared cluster (gnode002), and the median of 3
repeats only removes noise within a run, not between runs. Two runs of the identical
quick pipeline on the same node differed by about 25%. For example, the Part 4 HNSW
control (efSearch=24) measured 32,349 QPS in one run and 21,282 in the next, with the
same recall (0.963). Differences smaller than this between indexes, or between
neighbouring settings in a sweep, cannot be trusted. The large gaps reported here
(HNSW ≈4× IVF-Flat) are well outside this noise, but small ones (e.g. M=16 vs M=32
at the same recall) are not. Running all configurations interleaved in a single job,
or repeating whole runs on different days, would give an error bar.
