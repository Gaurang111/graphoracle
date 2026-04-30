"""JSON data connector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from graphoracle.connectors.base import DataConnector
from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.utils.exceptions import ConnectorError
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


class JSONConnector(DataConnector):
    """
    Load graph data from a JSON file.

    Expected format
    ---------------
    {
        "nodes": {
            "<node_type>": [
                {"id": "...", "feature1": ..., "feature2": ...}, ...
            ]
        },
        "edges": {
            "<edge_type>": [
                {"src": "...", "dst": "...", [optional edge features]}, ...
            ]
        },
        "timeseries": {                 (optional)
            "<node_id>": [
                {"timestamp": "...", "feature1": ..., ...}, ...
            ]
        }
    }
    """

    def load(
        self,
        path: str | Path,
        **kwargs: Any,
    ) -> HeterogeneousTemporalGraph:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            raise ConnectorError(f"Failed to load JSON from '{path}': {exc}") from exc

        builder = GraphBuilder(self.schema)
        nodes_data: dict = data.get("nodes", {})
        edges_data: dict = data.get("edges", {})
        ts_data: dict = data.get("timeseries", {})

        for nt in self.schema.node_types:
            records: list[dict] = nodes_data.get(nt.name, [])
            if not records:
                continue
            ids = [str(r.get("id", i)) for i, r in enumerate(records)]
            feat_cols = nt.features or []
            N, F = len(ids), max(len(feat_cols), 1)
            features = np.zeros((N, F), dtype=np.float32)
            for i, r in enumerate(records):
                for j, col in enumerate(feat_cols):
                    features[i, j] = float(r.get(col, 0.0))

            if ts_data:
                T_steps: list[str] = []
                for nid in ids:
                    if nid in ts_data and ts_data[nid]:
                        T_steps = [row.get("timestamp", str(k)) for k, row in enumerate(ts_data[nid])]
                        break
                if T_steps:
                    T = len(T_steps)
                    feat_3d = np.zeros((N, T, F), dtype=np.float32)
                    for i, nid in enumerate(ids):
                        for t, row in enumerate(ts_data.get(nid, [])):
                            for j, col in enumerate(feat_cols):
                                feat_3d[i, t, j] = float(row.get(col, 0.0))
                    builder.add_nodes(nt.name, ids, feat_3d, T_steps)
                    continue

            builder.add_nodes(nt.name, ids, features)

        for et in self.schema.edge_types:
            records = edges_data.get(et.name, [])
            if not records:
                continue
            src_ids = [str(r["src"]) for r in records]
            dst_ids = [str(r["dst"]) for r in records]
            try:
                builder.add_edges(et.name, src_ids, dst_ids)
            except Exception as exc:
                log.warning(f"Skipping edge type '{et.name}': {exc}")

        return builder.build()
