"""CSV data connector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from graphoracle.connectors.base import DataConnector
from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.utils.exceptions import ConnectorError
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


class CSVConnector(DataConnector):
    """
    Load graph data from CSV files.

    Expected formats
    ----------------
    nodes CSV   : node_id, node_type, feature1, feature2, ...
    edges CSV   : src_id, dst_id, edge_type, [edge_feature1, ...]
    timeseries  : node_id, timestamp, feature1, feature2, ...
                  (long format, one row per node-timestep)
    """

    def load(
        self,
        nodes: str | Path | dict[str, str | Path],
        edges: str | Path | dict[str, str | Path],
        timeseries: str | Path | dict[str, str | Path] | None = None,
        timestamp_col: str = "timestamp",
        node_id_col: str = "node_id",
        node_type_col: str = "node_type",
        src_col: str = "src_id",
        dst_col: str = "dst_id",
        edge_type_col: str = "edge_type",
        **kwargs: Any,
    ) -> HeterogeneousTemporalGraph:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ConnectorError("pandas is required for CSVConnector.") from exc

        builder = GraphBuilder(self.schema)

        # ------------------------------------------------------------------
        # Load nodes
        # ------------------------------------------------------------------
        node_dfs = self._load_split_or_combined(
            nodes, pd, split_col=node_type_col
        )

        ts_data: dict[str, dict[str, Any]] = {}
        if timeseries is not None:
            ts_data = self._load_timeseries(
                timeseries, pd, node_id_col, timestamp_col
            )

        for nt in self.schema.node_types:
            df = node_dfs.get(nt.name)
            if df is None:
                continue
            ids = df[node_id_col].tolist()
            feat_cols = [c for c in nt.features if c in df.columns]
            if nt.name in ts_data:
                features, timestamps = self._build_ts_features(
                    ids, ts_data[nt.name], nt.features
                )
            elif feat_cols:
                features = df[feat_cols].values.astype(np.float32)
                timestamps = None
            else:
                features = np.zeros((len(ids), len(nt.features)), dtype=np.float32)
                timestamps = None
            builder.add_nodes(nt.name, ids, features, timestamps)

        # ------------------------------------------------------------------
        # Load edges
        # ------------------------------------------------------------------
        edge_dfs = self._load_split_or_combined(
            edges, pd, split_col=edge_type_col
        )
        for et in self.schema.edge_types:
            df = edge_dfs.get(et.name)
            if df is None:
                continue
            src_ids = df[src_col].tolist()
            dst_ids = df[dst_col].tolist()
            feat_cols = [c for c in et.features if c in df.columns]
            ef = df[feat_cols].values.astype(np.float32) if feat_cols else None
            try:
                builder.add_edges(et.name, src_ids, dst_ids, ef)
            except Exception as exc:
                log.warning(f"Skipping edge type '{et.name}': {exc}")

        return builder.build()

    # ------------------------------------------------------------------

    def _load_split_or_combined(
        self,
        source: str | Path | dict[str, str | Path],
        pd: Any,
        split_col: str,
    ) -> dict[str, Any]:
        if isinstance(source, dict):
            return {k: pd.read_csv(v) for k, v in source.items()}
        df = pd.read_csv(source)
        if split_col in df.columns:
            return {k: grp for k, grp in df.groupby(split_col)}
        return {"_all": df}

    def _load_timeseries(
        self,
        ts_source: str | Path | dict[str, str | Path],
        pd: Any,
        node_id_col: str,
        timestamp_col: str,
    ) -> dict[str, dict[str, Any]]:
        """Returns {node_id: {timestamp: {feature: value}}}."""
        if isinstance(ts_source, dict):
            dfs = [pd.read_csv(v) for v in ts_source.values()]
            df = pd.concat(dfs)
        else:
            df = pd.read_csv(ts_source)

        data: dict[str, dict[str, Any]] = {}
        if node_id_col not in df.columns:
            return data
        for _, row in df.iterrows():
            nid = str(row[node_id_col])
            ts = row.get(timestamp_col, "")
            if nid not in data:
                data[nid] = {}
            data[nid][ts] = row.drop([node_id_col, timestamp_col], errors="ignore").to_dict()
        return data

    def _build_ts_features(
        self,
        node_ids: list[str],
        ts_data: dict[str, dict[str, Any]],
        feature_names: list[str],
    ) -> tuple[np.ndarray, list[Any] | None]:
        from datetime import datetime

        all_timestamps: list[str] = []
        for nid in node_ids:
            if nid in ts_data:
                all_timestamps = sorted(ts_data[nid].keys())
                break

        T = len(all_timestamps) or 1
        F = len(feature_names) or 1
        N = len(node_ids)
        features = np.zeros((N, T, F), dtype=np.float32)

        for i, nid in enumerate(node_ids):
            node_ts = ts_data.get(nid, {})
            for t, ts in enumerate(all_timestamps):
                row = node_ts.get(ts, {})
                for fi, feat in enumerate(feature_names):
                    features[i, t, fi] = float(row.get(feat, 0.0))

        timestamps = None
        if all_timestamps:
            try:
                timestamps = [datetime.fromisoformat(ts) for ts in all_timestamps]
            except ValueError:
                timestamps = None

        return features, timestamps
