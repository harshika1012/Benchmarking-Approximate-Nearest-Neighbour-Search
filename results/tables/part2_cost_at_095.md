Setting chosen on the tuning half (first 5,000 queries); QPS and recall reported on the final 5,000.

| Dataset | Index | Setting | Recall | Build (s) | Size (MB) | Bytes/vector |
|---|---|---|---|---:|---:|---:|
| SIFT | Exact | - | 1.000 | 0.4 | 512.0 | 512.0 |
| SIFT | HNSW | M=32, efSearch=48 | 0.971 | 697.0 | 784.1 | 784.1 |
| SIFT | IVF-Flat | nlist=4096, nprobe=64 | 0.977 | 195.4 | 522.1 | 522.1 |
| SIFT | IVF-PQ | nlist=1024,pq_m=64, nprobe=256 (below 0.95) | 0.880 | 37.7 | 72.7 | 72.7 |
| GloVe-100 | Exact | - | 1.000 | 0.3 | 473.4 | 400.0 |
| GloVe-100 | HNSW | M=32, efSearch=384 | 0.955 | 1076.0 | 795.5 | 672.2 |
| GloVe-100 | IVF-Flat | nlist=4096, nprobe=512 | 0.963 | 177.3 | 484.5 | 409.4 |
| GloVe-100 | IVF-PQ | nlist=1024,pq_m=50, nprobe=512 (below 0.95) | 0.866 | 29.7 | 69.2 | 58.4 |
