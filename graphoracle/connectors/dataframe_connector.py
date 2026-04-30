"""Pandas DataFrame connector."""

from __future__ import annotations

from typing import Any

import numpy as np

from graphoracle.connectors.base import DataConnector
from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.utils.exceptions import ConnectorError
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


class DataFrameConnector(DataConnector):
    """
    Load graph data from pandas DataFrames.

    Parameters
    ----------
    schema      : GraphSchema

    Usage
    -----
    connector = DataFrameConnector(schema)
    graph = connector.load(
        node_dfs={"sensor": df_sensor, ...},
        edge_dfs={"link": df_link, ...},
        id_col="node_id",
    )
    """

    def load(
        self,
        node_dfs: dict[str, Any],
        edge_dfs: dict[str, Any] | None = None,
        id_col: str = "node_id",
        src_col: str = "src_id",
        dst_col: str = "dst_id",
        timestamps: list[Any] | None = None,
        **kwargs: Any,
    ) -> HeterogeneousTemporalGraph:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ConnectorError("pandas is required for DataFrameConnector.") from exc

        builder = GraphBuilder(self.schema)

        for nt in self.schema.node_types:
            df = node_dfs.get(nt.name)
            if df is None:
                continue

            if id_col in df.columns:
                ids = df[id_col].astype(str).tolist()
                feat_df = df.drop(columns=[id_col], errors="ignore")
            else:
                ids = [str(i) for i in df.index.tolist()]
                feat_df = df

            feat_cols = [c for c in nt.features if c in feat_df.columns]
            if feat_cols:
                features = feat_df[feat_cols].values.astype(np.float32)
            else:
                features = np.zeros((len(ids), max(len(nt.features), 1)), dtype=np.float32)

            builder.add_nodes(nt.name, ids, features, timestamps)

        for et in self.schema.edge_types:
            df = (edge_dfs or {}).get(et.name)
            if df is None:
                continue

            src_ids_col = src_col if src_col in df.columns else df.columns[0]
            dst_ids_col = dst_col if dst_col in df.columns else df.columns[1]
            src_ids = df[src_ids_col].astype(str).tolist()
            dst_ids = df[dst_ids_col].astype(str).tolist()

            feat_cols = [c for c in et.features if c in df.columns]
            ef = df[feat_cols].values.astype(np.float32) if feat_cols else None
            try:
                builder.add_edges(et.name, src_ids, dst_ids, ef)
            except Exception as exc:
                log.warning(f"Skipping edge type '{et.name}': {exc}")

        return builder.build()
