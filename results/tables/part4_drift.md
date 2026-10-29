Stale = built on old data, new data added afterwards; rebuilt = trained and built on all data. Update time = adding the new vectors (stale) or the full rebuild.

| Shift | Index | State | Setting | Recall | QPS | Dist. / query | List imbalance | Update time (s) |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | HNSW | rebuilt | efSearch=96 | 0.974 | 4,531 | 1,690 | - | 113.3 |
| 0 | HNSW | stale | efSearch=96 | 0.972 | 4,376 | 1,704 | - | 69.8 |
| 0 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 5,944 | 6,521 | 1.12 | 10.4 |
| 0 | IVF-Flat | stale | nprobe=16 | 0.999 | 5,740 | 6,374 | 1.18 | 0.7 |
| 0 | IVF-PQ | rebuilt | nprobe=32 | 0.853 | 2,399 | 13,175 | 1.12 | 16.2 |
| 0 | IVF-PQ | stale | nprobe=32 | 0.856 | 2,528 | 12,790 | 1.18 | 1.3 |
| 0.5 | HNSW | rebuilt | efSearch=96 | 0.989 | 5,467 | 1,350 | - | 96.3 |
| 0.5 | HNSW | stale | efSearch=96 | 0.965 | 5,362 | 1,333 | - | 50.5 |
| 0.5 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,167 | 6,517 | 1.25 | 9.8 |
| 0.5 | IVF-Flat | stale | nprobe=16 | 1.000 | 5,817 | 6,738 | 1.64 | 0.7 |
| 0.5 | IVF-PQ | rebuilt | nprobe=32 | 0.860 | 2,757 | 13,185 | 1.25 | 15.7 |
| 0.5 | IVF-PQ | stale | nprobe=32 | 0.718 | 2,708 | 13,587 | 1.64 | 1.3 |
| 1 | HNSW | rebuilt | efSearch=96 | 0.992 | 5,797 | 1,334 | - | 90.1 |
| 1 | HNSW | stale | efSearch=96 | 0.946 | 6,020 | 1,292 | - | 43.1 |
| 1 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 5,992 | 6,527 | 1.26 | 9.4 |
| 1 | IVF-Flat | stale | nprobe=16 | 1.000 | 5,222 | 7,434 | 2.20 | 0.7 |
| 1 | IVF-PQ | rebuilt | nprobe=32 | 0.862 | 2,754 | 13,088 | 1.26 | 15.6 |
| 1 | IVF-PQ | stale | nprobe=32 | 0.433 | 2,514 | 14,998 | 2.20 | 1.4 |
| 2 | HNSW | rebuilt | efSearch=96 | 0.994 | 5,626 | 1,335 | - | 93.8 |
| 2 | HNSW | stale | efSearch=96 | 0.992 | 5,723 | 1,330 | - | 43.3 |
| 2 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,096 | 6,692 | 1.27 | 8.8 |
| 2 | IVF-Flat | stale | nprobe=16 | 0.989 | 3,641 | 13,604 | 3.17 | 0.6 |
| 2 | IVF-PQ | rebuilt | nprobe=32 | 0.860 | 2,702 | 13,238 | 1.27 | 15.3 |
| 2 | IVF-PQ | stale | nprobe=32 | 0.180 | 1,763 | 25,316 | 3.17 | 1.3 |
