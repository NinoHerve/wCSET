"""Overlap between the Study III spike epochs. The exported epochs carry no timestamps, so spikes closer than
one epoch length are found from the samples their epochs share."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = ["find_overlaps", "window_in_baseline", "find_bursts"]


def find_overlaps(epochs):
    """Pairs of epochs cut from the same stretch of recording, one row per pair: the later epoch starts
    `shift_s` seconds after the earlier one, and both are equal on every channel over the shared samples."""
    X = epochs.get_data()
    n, _, T = X.shape
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in np.flatnonzero(np.abs(X[i] - X[j][:, [0]]).max(0) < 1e-12):
                if k > 0 and np.allclose(X[i][:, k:], X[j][:, :T - k], atol=1e-12):
                    rows.append((i, j, k / epochs.info["sfreq"]))
    return pd.DataFrame(rows, columns=["earlier", "later", "shift_s"])


def window_in_baseline(overlaps, window, baseline):
    """The overlap pairs where the earlier spike's window falls inside the later spike's baseline."""
    (tmin, tmax), (b0, b1), s = window, baseline, overlaps["shift_s"]
    return overlaps[(tmin - s < b1) & (tmax - s > b0)]


def find_bursts(links, n):
    """One burst label per spike (n spikes): spikes joined by `links`, chains included, share a label."""
    graph = coo_matrix((np.ones(len(links)), (links["earlier"], links["later"])), shape=(n, n))
    return connected_components(graph, directed=False)[1]
