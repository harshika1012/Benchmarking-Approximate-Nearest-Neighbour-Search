Setting chosen on the tuning half (first 5,000 queries); QPS and recall reported on the final 5,000.

| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 |
|---|---|---|---|
| SIFT | Exact | 26 (recall 1.000) | 26 (recall 1.000) |
| SIFT | HNSW | 8,513 (M=16, efSearch=32; recall 0.918) | 5,026 (M=16, efSearch=64; recall 0.971) |
| SIFT | IVF-Flat | 2,245 (nlist=4096, nprobe=32; recall 0.933) | 1,268 (nlist=4096, nprobe=64; recall 0.977) |
| SIFT | IVF-PQ | not reached (max 0.876) | not reached (max 0.876) |
| GloVe-100 | Exact | 27 (recall 1.000) | 27 (recall 1.000) |
| GloVe-100 | HNSW | 1,489 (M=48, efSearch=128; recall 0.909) | 660 (M=32, efSearch=384; recall 0.955) |
| GloVe-100 | IVF-Flat | 433 (nlist=1024, nprobe=64; recall 0.908) | 222 (nlist=4096, nprobe=512; recall 0.963) |
| GloVe-100 | IVF-PQ | not reached (max 0.866) | not reached (max 0.866) |
