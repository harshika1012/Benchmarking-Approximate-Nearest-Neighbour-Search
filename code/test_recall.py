import numpy as np
from recall import recall_at_k, true_distances

GT_IDS = np.array([[0, 1, 2, 3]])
GT_DIST = np.array([[1.0, 2.0, 3.0, 4.0]])


def test_perfect_recall():
    assert recall_at_k(GT_IDS, GT_DIST, GT_IDS, GT_DIST, k=4) == 1.0


def test_zero_recall():
    ids = np.array([[5, 6, 7, 8]])
    dist = np.array([[10.0, 11.0, 12.0, 13.0]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 0.0


def test_partial_recall():
    ids = np.array([[0, 1, 8, 9]])
    dist = np.array([[1.0, 2.0, 9.0, 10.0]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 0.5


def test_tie_at_k_is_accepted():
    # id 9 is not in the ground truth but ties with the 4th neighbour.
    ids = np.array([[0, 1, 2, 9]])
    dist = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 1.0


def test_beyond_cutoff_is_rejected():
    ids = np.array([[0, 1, 2, 9]])
    dist = np.array([[1.0, 2.0, 3.0, 4.01]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 0.75


def test_rounding_noise_is_tolerated():
    ids = np.array([[0, 1, 2, 3]])
    dist = np.array([[1.0, 2.0, 3.0, 4.0 * (1 + 1e-7)]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 1.0


def test_missing_result_is_rejected():
    ids = np.array([[0, 1, 2, -1]])
    dist = np.array([[1.0, 2.0, 3.0, np.inf]])
    assert recall_at_k(ids, dist, GT_IDS, GT_DIST, k=4) == 0.75


def test_averages_over_queries():
    gt_ids = np.array([[0, 1], [0, 1]])
    gt_dist = np.array([[1.0, 2.0], [1.0, 2.0]])
    ids = np.array([[0, 1], [7, 8]])
    dist = np.array([[1.0, 2.0], [5.0, 6.0]])
    assert recall_at_k(ids, dist, gt_ids, gt_dist, k=2) == 0.5


def test_true_distances():
    base = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    d = true_distances(base, queries, np.array([[1, 0, -1]]))
    assert np.allclose(d[0, :2], [5.0, 0.0])
    assert np.isinf(d[0, 2])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
    print(f"All {len(tests)} recall tests passed!")
