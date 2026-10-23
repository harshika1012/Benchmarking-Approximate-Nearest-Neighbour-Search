# import h5py
# from pathlib import Path

# path = Path("data/sift/sift-128-euclidean.hdf5")

# with h5py.File(path, "r") as f:
#     print("Dataset keys:")
#     for key in f.keys():
#         item = f[key]
#         print(f"  {key}: shape={item.shape}, dtype={item.dtype}")


import h5py
import faiss
import numpy as np

path = "data/sift/sift-128-euclidean.hdf5"

with h5py.File(path, "r") as f:
    train = f["train"][:]
    queries = f["test"][:]
    ground_truth = f["neighbors"][:]

print("Building exact FAISS index...")

index = faiss.IndexFlatL2(train.shape[1])
index.add(train)

print("Searching first 100 queries...")
distances, indices = index.search(queries[:100], 10)

# Compare the top 10 exact neighbours with the stored ground truth.
matches = 0

for i in range(100):
    expected = set(ground_truth[i, :10])
    actual = set(indices[i])
    matches += len(expected & actual)

recall = matches / (100 * 10)

print(f"Recall@10 against stored ground truth: {recall:.4f}")