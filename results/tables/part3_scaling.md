Fastest setting whose tuning recall ≥ 0.95; QPS and recall on the final 5,000 queries.

| Index | Vectors | Setting | Recall | QPS | Speed-up vs exact | Build (s) |
|---|---:|---|---:|---:|---:|---:|
| Exact | 100,000 | - | 1.000 | 243 | 1.0x | 0.0 |
| Exact | 300,000 | - | 1.000 | 86 | 1.0x | 0.1 |
| Exact | 1,000,000 | - | 1.000 | 24 | 1.0x | 0.3 |
| HNSW | 100,000 | M=16, efSearch=32 | 0.958 | 12,522 | 51.5x | 31.2 |
| HNSW | 300,000 | M=32, efSearch=32 | 0.956 | 7,361 | 86.0x | 153.0 |
| HNSW | 1,000,000 | M=32, efSearch=48 | 0.971 | 4,501 | 185.0x | 697.0 |
| IVF-Flat | 100,000 | nlist=1024, nprobe=32 | 0.965 | 5,575 | 22.9x | 5.3 |
| IVF-Flat | 300,000 | nlist=4096, nprobe=64 | 0.970 | 2,945 | 34.4x | 54.2 |
| IVF-Flat | 1,000,000 | nlist=4096, nprobe=64 | 0.977 | 1,189 | 48.9x | 195.4 |

| Index | QPS at smallest N | QPS at largest N | Log-log slope |
|---|---:|---:|---:|
| Exact | 243 | 24 | -1.00 |
| HNSW | 12,522 | 4,501 | -0.44 |
| IVF-Flat | 5,575 | 1,189 | -0.67 |
