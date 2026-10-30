Stale = built on old data, new data added afterwards; rebuilt = trained and built on all data. Update time = adding the new vectors (stale) or the full rebuild.

| Shift | Index | State | Setting | Recall | QPS | Dist. / query | List imbalance | Update time (s) |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 | HNSW | rebuilt | efSearch=96 | 0.974 | 4,912 | 1,690 | - | 108.5 |
| 0 | HNSW | stale | efSearch=96 | 0.972 | 4,817 | 1,704 | - | 68.7 |
| 0 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,040 | 6,521 | 1.12 | 8.9 |
| 0 | IVF-Flat | stale | nprobe=16 | 0.999 | 6,133 | 6,374 | 1.18 | 0.6 |
| 0 | IVF-PQ | rebuilt | nprobe=32 | 0.853 | 2,583 | 13,175 | 1.12 | 14.3 |
| 0 | IVF-PQ | stale | nprobe=32 | 0.856 | 2,848 | 12,790 | 1.18 | 1.3 |
| 0.5 | HNSW | rebuilt | efSearch=96 | 0.989 | 5,986 | 1,350 | - | 92.5 |
| 0.5 | HNSW | stale | efSearch=96 | 0.965 | 4,894 | 1,333 | - | 46.5 |
| 0.5 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,169 | 6,517 | 1.25 | 7.9 |
| 0.5 | IVF-Flat | stale | nprobe=16 | 1.000 | 5,880 | 6,738 | 1.64 | 0.6 |
| 0.5 | IVF-PQ | rebuilt | nprobe=32 | 0.860 | 2,595 | 13,185 | 1.25 | 15.2 |
| 0.5 | IVF-PQ | stale | nprobe=32 | 0.718 | 2,285 | 13,587 | 1.64 | 1.3 |
| 1 | HNSW | rebuilt | efSearch=96 | 0.992 | 6,036 | 1,334 | - | 93.4 |
| 1 | HNSW | stale | efSearch=96 | 0.946 | 6,315 | 1,292 | - | 40.6 |
| 1 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,446 | 6,527 | 1.26 | 8.8 |
| 1 | IVF-Flat | stale | nprobe=16 | 1.000 | 5,924 | 7,434 | 2.20 | 0.6 |
| 1 | IVF-PQ | rebuilt | nprobe=32 | 0.862 | 2,698 | 13,088 | 1.26 | 15.8 |
| 1 | IVF-PQ | stale | nprobe=32 | 0.433 | 2,418 | 14,998 | 2.20 | 1.3 |
| 2 | HNSW | rebuilt | efSearch=96 | 0.994 | 6,051 | 1,335 | - | 95.5 |
| 2 | HNSW | stale | efSearch=96 | 0.992 | 5,874 | 1,330 | - | 42.9 |
| 2 | IVF-Flat | rebuilt | nprobe=16 | 1.000 | 6,444 | 6,692 | 1.27 | 8.7 |
| 2 | IVF-Flat | stale | nprobe=16 | 0.989 | 3,894 | 13,604 | 3.17 | 0.6 |
| 2 | IVF-PQ | rebuilt | nprobe=32 | 0.860 | 2,817 | 13,238 | 1.27 | 15.2 |
| 2 | IVF-PQ | stale | nprobe=32 | 0.180 | 1,814 | 25,316 | 3.17 | 1.3 |
