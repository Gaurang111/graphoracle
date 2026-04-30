"""Fluent builder for HeterogeneousTemporalGraph."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph, TemporalEvent
from graphoracle.graph.schema import GraphSchema
from graphoracle.utils.exceptions import GraphSchemaError


class GraphBuilder:
    """
    Fluent builder for HeterogeneousTemporalGraph.

    Usage
    -----
    graph = (
        GraphBuilder(schema)
        .add_nodes("sensor", ids, features, timestamps)
        .add_edges("link", src_ids, dst_ids)
        .build()
    )
    """

    def __init__(self, schema: GraphSchema) -> None:
        self.schema = schema
        self._node_ids: dict[str, list[str]] = {}
        self._node_features: dict[str, Tensor] = {}
        self._node_timestamps: dict[str, list[Any] | None] = {}
        self._edge_src: dict[str, list[Any]] = {}
        self._edge_dst: dict[str, list[Any]] = {}
        self._edge_features: dict[str, Tensor | None] = {}
        self._events: list[TemporalEvent] = []

    # ------------------------------------------------------------------
    # Node builder
    # ------------------------------------------------------------------

    def add_nodes(
        self,
        node_type: str,
        node_ids: list[str],
        features: np.ndarray | Tensor,
        timestamps: list[Any] | None = None,
    ) -> "GraphBuilder":
        """
        Register *node_ids* with their *features*.

        features shape: (N, F) or (N, T, F) — both are accepted.
        """
        if node_type not in self.schema.node_type_names:
            raise GraphSchemaError(f"Node type '{node_type}' not in schema.")

        if isinstance(features, np.ndarray):
            feat = torch.from_numpy(features.astype(np.float32))
        else:
            feat = features.float()

        if feat.ndim == 2:
            feat = feat.unsqueeze(1)  # (N, F) → (N, 1, F)

        self._node_ids[node_type] = [str(nid) for nid in node_ids]
        self._node_features[node_type] = feat
        self._node_timestamps[node_type] = timestamps
        return self

    # ------------------------------------------------------------------
    # Edge builder
    # ------------------------------------------------------------------

    def add_edges(
        self,
        edge_type: str,
        src_ids: list[Any],
        dst_ids: list[Any],
        edge_features: np.ndarray | Tensor | None = None,
    ) -> "GraphBuilder":
        """
        Register edges of *edge_type* from *src_ids* to *dst_ids*.

        src_ids and dst_ids are node ID strings matching those passed to
        add_nodes for the corresponding types.
        """
        et = self._get_edge_type(edge_type)

        src_map = {nid: i for i, nid in enumerate(self._node_ids.get(et.src_type, []))}
        dst_map = {nid: i for i, nid in enumerate(self._node_ids.get(et.dst_type, []))}

        for s in src_ids:
            if str(s) not in src_map:
                raise ValueError(
                    f"Source node '{s}' not registered for type '{et.src_type}'."
                )
        for d in dst_ids:
            if str(d) not in dst_map:
                raise ValueError(
                    f"Destination node '{d}' not registered for type '{et.dst_type}'."
                )

        self._edge_src[edge_type] = [str(s) for s in src_ids]
        self._edge_dst[edge_type] = [str(d) for d in dst_ids]

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray):
                edge_features = torch.from_numpy(edge_features.astype(np.float32))
            self._edge_features[edge_type] = edge_features
        else:
            self._edge_features[edge_type] = None

        return self

    def inject_event(
        self,
        event_type: str,
        affected_nodes: list[str],
        features: dict[str, float],
        start: Any,
        end: Any,
    ) -> "GraphBuilder":
        self._events.append(TemporalEvent(event_type, affected_nodes, features, start, end))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> HeterogeneousTemporalGraph:
        node_data: dict[str, dict[str, Any]] = {}
        for nt_name, ids in self._node_ids.items():
            node_data[nt_name] = {
                "ids": ids,
                "features": self._node_features[nt_name],
                "timestamps": self._node_timestamps.get(nt_name),
            }

        edge_data: dict[str, dict[str, Any]] = {}
        for et_name, src_ids in self._edge_src.items():
            et = self._get_edge_type(et_name)
            dst_ids = self._edge_dst[et_name]
            src_map = {nid: i for i, nid in enumerate(self._node_ids.get(et.src_type, []))}
            dst_map = {nid: i for i, nid in enumerate(self._node_ids.get(et.dst_type, []))}
            src_idx = [src_map[s] for s in src_ids]
            dst_idx = [dst_map[d] for d in dst_ids]
            edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            edge_data[et_name] = {
                "edge_index": edge_index,
                "edge_features": self._edge_features.get(et_name),
            }

        return HeterogeneousTemporalGraph(self.schema, node_data, edge_data, list(self._events))

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframes(
        cls,
        schema: GraphSchema,
        node_dfs: dict[str, Any],
        edge_dfs: dict[str, Any],
        feature_dfs: dict[str, Any] | None = None,
        timestamps: list[Any] | None = None,
    ) -> HeterogeneousTemporalGraph:
        """Build a graph from pandas DataFrames keyed by type name."""
        builder = cls(schema)
        for nt in schema.node_types:
            df = node_dfs.get(nt.name)
            if df is None:
                continue
            ids = [str(i) for i in df.index.tolist()]
            feat_cols = [c for c in nt.features if c in df.columns]
            if feat_cols:
                feat = df[feat_cols].values.astype(np.float32)
            else:
                feat = np.zeros((len(ids), max(nt.feature_dim, 1)), dtype=np.float32)
            builder.add_nodes(nt.name, ids, feat, timestamps)

        for et in schema.edge_types:
            df = edge_dfs.get(et.name)
            if df is None:
                continue
            src_col = "src_id" if "src_id" in df.columns else df.columns[0]
            dst_col = "dst_id" if "dst_id" in df.columns else df.columns[1]
            try:
                builder.add_edges(et.name, df[src_col].tolist(), df[dst_col].tolist())
            except Exception:
                pass

        return builder.build()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_edge_type(self, name: str):
        for et in self.schema.edge_types:
            if et.name == name:
                return et
        raise GraphSchemaError(f"Edge type '{name}' not in schema.")
