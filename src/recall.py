import numpy as np


def recall_at_k(
    retrieved_ids: np.ndarray,
    retrieved_distances: np.ndarray,
    ground_truth_ids: np.ndarray,
    ground_truth_distances: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute tie-aware recall@k.

    A retrieved vector is correct if its distance is no greater
    than the ground-truth kth-neighbour distance.

    Each query contributes between 0 and k correct neighbours.
    """
    retrieved_ids = np.asarray(retrieved_ids)
    retrieved_distances = np.asarray(retrieved_distances)
    ground_truth_ids = np.asarray(ground_truth_ids)
    ground_truth_distances = np.asarray(ground_truth_distances)

    if k <= 0:
        raise ValueError("k must be positive")

    if any(a.ndim != 2 for a in (
        retrieved_ids,
        retrieved_distances,
        ground_truth_ids,
        ground_truth_distances,
    )):
        raise ValueError("All inputs must be 2D arrays")

    n = retrieved_ids.shape[0]

    if any(a.shape[0] != n for a in (
        retrieved_distances,
        ground_truth_ids,
        ground_truth_distances,
    )):
        raise ValueError("Number of queries must match")

    if retrieved_ids.shape[1] < k or retrieved_distances.shape[1] < k:
        raise ValueError("Retrieved results must contain at least k neighbours")

    if ground_truth_ids.shape[1] < k or ground_truth_distances.shape[1] < k:
        raise ValueError("Ground truth must contain at least k neighbours")

    # The distance of the kth true neighbour is the cutoff.
    cutoff = ground_truth_distances[:, k - 1:k]

    # Count returned neighbours at or inside the cutoff.
    # This handles alternative neighbours tied at rank k.
    correct = retrieved_distances[:, :k] <= cutoff

    return float(correct.sum() / (n * k))