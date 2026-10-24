Fastest setting whose tuning recall ≥ 0.95; QPS and recall on the final 5,000 queries.

| Index | Vectors | Setting | Recall | QPS | Speed-up vs exact | Build (s) |
|---|---:|---|---:|---:|---:|---:|
| Exact | 100,000 | - | 1.000 | 259 | 1.0x | 0.0 |
| Exact | 300,000 | - | 1.000 | 87 | 1.0x | 0.1 |
| Exact | 1,000,000 | - | 1.000 | 26 | 1.0x | 0.3 |
| HNSW | 100,000 | M=16, efSearch=32 | 0.958 | 13,127 | 50.7x | 26.8 |
| HNSW | 300,000 | M=32, efSearch=32 | 0.956 | 7,914 | 90.6x | 140.8 |
| HNSW | 1,000,000 | M=16, efSearch=64 | 0.971 | 5,026 | 192.7x | 507.5 |
| IVF-Flat | 100,000 | nlist=1024, nprobe=32 | 0.965 | 6,489 | 25.1x | 4.5 |
| IVF-Flat | 300,000 | nlist=4096, nprobe=64 | 0.970 | 2,964 | 33.9x | 52.4 |
| IVF-Flat | 1,000,000 | nlist=4096, nprobe=64 | 0.977 | 1,268 | 48.6x | 171.7 |

| Index | QPS at smallest N | QPS at largest N | Log-log slope |
|---|---:|---:|---:|
| Exact | 259 | 26 | -1.00 |
| HNSW | 13,127 | 5,026 | -0.42 |
| IVF-Flat | 6,489 | 1,268 | -0.71 |
