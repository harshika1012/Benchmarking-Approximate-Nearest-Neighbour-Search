import numpy as np
from recall import recall_at_k


def test_perfect_recall():
    ids = np.array([[0, 1, 2]])
    distances = np.array([[1.0, 2.0, 3.0]])

    assert recall_at_k(ids, distances, ids, distances, k=3) == 1.0


def test_zero_recall():
    retrieved_ids = np.array([[4, 5, 6]])
    retrieved_distances = np.array([[10.0, 11.0, 12.0]])

    gt_ids = np.array([[0, 1, 2]])
    gt_distances = np.array([[1.0, 2.0, 3.0]])

    assert recall_at_k(
        retrieved_ids, retrieved_distances, gt_ids, gt_distances, k=3
    ) == 0.0


def test_tie_at_k_is_accepted():
    retrieved_ids = np.array([[0, 1, 9]])
    retrieved_distances = np.array([[1.0, 2.0, 3.0]])

    gt_ids = np.array([[0, 1, 2]])
    gt_distances = np.array([[1.0, 2.0, 3.0]])

    assert recall_at_k(
        retrieved_ids, retrieved_distances, gt_ids, gt_distances, k=3
    ) == 1.0


if __name__ == "__main__":
    test_perfect_recall()
    test_zero_recall()
    test_tie_at_k_is_accepted()
    print("All recall tests passed!")