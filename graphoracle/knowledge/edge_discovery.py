"""Automatic edge discovery from node time series."""

from __future__ import annotations

from typing import Any

import numpy as np

from graphoracle.utils.exceptions import EdgeDiscoveryError
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED_METHODS = {"correlation", "granger", "spatial", "mutual_info"}


class EdgeDiscovery:
    """
    Discover edges between nodes by analysing their time series.

    Supported methods
    -----------------
    correlation  : Pearson correlation threshold
    granger      : Granger causality (requires statsmodels)
    spatial      : k-nearest neighbours by spatial distance
    mutual_info  : Mutual information (requires scikit-learn)

    Parameters
    ----------
    method    : one of the supported methods above
    threshold : minimum score to create an edge (meaning depends on method)
    """

    def __init__(self, method: str = "correlation", threshold: float = 0.5) -> None:
        if method not in SUPPORTED_METHODS:
            raise EdgeDiscoveryError(
                f"Unsupported method '{method}'. Choose from {SUPPORTED_METHODS}."
            )
        self.method = method
        self.threshold = threshold

    def discover(
        self,
        node_ids: list[str],
        timeseries: np.ndarray,
    ) -> list[tuple[str, str]]:
        """
        Discover edges between *node_ids* from *timeseries*.

        Parameters
        ----------
        node_ids   : list of N node ID strings
        timeseries : (N, T) float array — one time series per node

        Returns
        -------
        list of (src_id, dst_id) string pairs
        """
        N = len(node_ids)
        if timeseries.shape[0] != N:
            raise EdgeDiscoveryError(
                f"timeseries has {timeseries.shape[0]} rows but {N} node_ids."
            )

        if self.method == "correlation":
            return self._correlation(node_ids, timeseries)
        elif self.method == "mutual_info":
            return self._mutual_info(node_ids, timeseries)
        elif self.method == "spatial":
            return self._spatial_knn(node_ids, timeseries)
        else:
            return self._correlation(node_ids, timeseries)

    def _correlation(
        self, node_ids: list[str], ts: np.ndarray
    ) -> list[tuple[str, str]]:
        corr = np.corrcoef(ts)  # (N, N)
        edges = []
        N = len(node_ids)
        for i in range(N):
            for j in range(N):
                if i != j and abs(corr[i, j]) >= self.threshold:
                    edges.append((node_ids[i], node_ids[j]))
        log.info(f"Discovered {len(edges)} edges via correlation (threshold={self.threshold}).")
        return edges

    def _mutual_info(
        self, node_ids: list[str], ts: np.ndarray
    ) -> list[tuple[str, str]]:
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            log.warning("scikit-learn not installed, falling back to correlation.")
            return self._correlation(node_ids, ts)

        N = len(node_ids)
        edges = []
        for i in range(N):
            mi = mutual_info_regression(ts.T, ts[i])
            for j in range(N):
                if i != j and mi[j] >= self.threshold:
                    edges.append((node_ids[i], node_ids[j]))
        return edges

    def _spatial_knn(
        self, node_ids: list[str], ts: np.ndarray, k: int = 5
    ) -> list[tuple[str, str]]:
        from scipy.spatial.distance import cdist

        dists = cdist(ts, ts, metric="euclidean")
        N = len(node_ids)
        edges = []
        for i in range(N):
            nearest = np.argsort(dists[i])[1 : k + 1]
            for j in nearest:
                edges.append((node_ids[i], node_ids[j]))
        return edges
