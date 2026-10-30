"""Seeded data generator for the Part 4 drift experiment."""
import numpy as np

D = 64
N_CLUSTERS = 100
CLUSTER_STD = 0.5
SEED = 2026


def _mixture(rng, centres, n):
    labels = rng.integers(len(centres), size=n)
    noise = rng.standard_normal((n, centres.shape[1]))
    return centres[labels] + CLUSTER_STD * noise


def make_drift_data(shift, n_old, n_new, n_queries, seed=SEED):
    """
    Old vectors: a mixture of N_CLUSTERS Gaussians with N(0, 1) centres.
    New vectors and queries: the same mixture translated by shift * sqrt(D)
    along a fixed random direction, so `shift` is in units of the
    per-dimension spread of the centres.

    Each part uses its own random stream, so changing `shift` moves the new
    data and queries without changing anything else.
    """
    streams = [np.random.default_rng([seed, i]) for i in range(4)]
    centres = streams[0].standard_normal((N_CLUSTERS, D))
    direction = streams[0].standard_normal(D)
    direction /= np.linalg.norm(direction)
    offset = shift * np.sqrt(D) * direction

    old = _mixture(streams[1], centres, n_old)
    new = _mixture(streams[2], centres, n_new) + offset
    queries = _mixture(streams[3], centres, n_queries) + offset
    return old.astype(np.float32), new.astype(np.float32), queries.astype(np.float32)
