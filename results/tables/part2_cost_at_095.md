Setting chosen on the tuning half (first 5,000 queries); QPS and recall reported on the final 5,000.

| Dataset | Index | Setting | Recall | Build (s) | Size (MB) | Bytes/vector |
|---|---|---|---|---:|---:|---:|
| SIFT | Exact | - | 1.000 | 0.4 | 512.0 | 512.0 |
| SIFT | HNSW | M=16, efSearch=64 | 0.971 | 507.5 | 656.3 | 656.3 |
| SIFT | IVF-Flat | nlist=4096, nprobe=64 | 0.977 | 171.7 | 522.1 | 522.1 |
| SIFT | IVF-PQ | nlist=1024,pq_m=64, nprobe=256 (below 0.95) | 0.880 | 31.0 | 72.7 | 72.7 |
| GloVe-100 | Exact | - | 1.000 | 0.3 | 473.4 | 400.0 |
| GloVe-100 | HNSW | M=32, efSearch=384 | 0.955 | 938.7 | 795.5 | 672.2 |
| GloVe-100 | IVF-Flat | nlist=4096, nprobe=512 | 0.963 | 157.9 | 484.5 | 409.4 |
| GloVe-100 | IVF-PQ | nlist=1024,pq_m=50, nprobe=512 (below 0.95) | 0.866 | 27.0 | 69.2 | 58.4 |
