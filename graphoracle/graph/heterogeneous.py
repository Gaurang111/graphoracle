"""Core heterogeneous temporal graph data structure."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from graphoracle.graph.schema import GraphSchema


@dataclass
class TemporalEvent:
    """An external event injected into the graph."""

    event_type: str
    affected_nodes: list[str]
    features: dict[str, float]
    start: Any
    end: Any


class HeterogeneousTemporalGraph:
    """
    Container for a heterogeneous temporal graph.

    Stores per-type node features as (N, T, F) tensors and edge indices
    as (2, E) tensors.  Timestamps, node IDs, and injected events are
    stored alongside the tensors.
    """

    def __init__(
        self,
        schema: GraphSchema,
        node_data: dict[str, dict[str, Any]],
        edge_data: dict[str, dict[str, Any]],
        events: list[TemporalEvent] | None = None,
    ) -> None:
        self.schema = schema
        self._node_data = node_data      # {nt_name: {ids, features, timestamps}}
        self._edge_data = edge_data      # {et_name: {edge_index, edge_features}}
        self._events: list[TemporalEvent] = events or []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        for v in self._node_data.values():
            feat = v["features"]
            if isinstance(feat, Tensor):
                return feat.shape[1] if feat.ndim == 3 else 1
            return 1
        return 0

    @property
    def timestamps(self) -> list[Any] | None:
        """Return the timestamp list from the first node type that has one."""
        for v in self._node_data.values():
            ts = v.get("timestamps")
            if ts is not None:
                return ts
        return None

    @property
    def events(self) -> list[TemporalEvent]:
        return self._events

    # ------------------------------------------------------------------
    # Node accessors
    # ------------------------------------------------------------------

    def num_nodes(self, node_type: str) -> int:
        v = self._node_data.get(node_type)
        if v is None:
            return 0
        feat = v["features"]
        return feat.shape[0] if isinstance(feat, Tensor) else len(v["ids"])

    def all_node_ids(self, node_type: str) -> list[str]:
        v = self._node_data.get(node_type)
        return v["ids"] if v is not None else []

    def get_node_features(self, node_type: str) -> Tensor:
        """Return (N, T, F) float tensor for *node_type*."""
        v = self._node_data.get(node_type)
        if v is None:
            nt = self.schema.get_node_type(node_type)
            return torch.zeros(0, 1, max(nt.feature_dim, 1))
        feat = v["features"]
        if not isinstance(feat, Tensor):
            feat = torch.from_numpy(feat).float()
            v["features"] = feat
        if feat.ndim == 2:
            feat = feat.unsqueeze(1)
            v["features"] = feat
        return feat

    # ------------------------------------------------------------------
    # Edge accessors
    # ------------------------------------------------------------------

    def get_edge_index(self, edge_type: str) -> Tensor:
        """Return (2, E) long tensor for *edge_type*."""
        v = self._edge_data.get(edge_type)
        if v is None:
            return torch.zeros(2, 0, dtype=torch.long)
        return v["edge_index"]

    def get_edge_features(self, edge_type: str) -> Tensor | None:
        v = self._edge_data.get(edge_type)
        if v is None:
            return None
        return v.get("edge_features")

    # ------------------------------------------------------------------
    # Event injection
    # ------------------------------------------------------------------

    def inject_event(
        self,
        event_type: str,
        affected_nodes: list[str],
        features: dict[str, float],
        start: Any,
        end: Any,
    ) -> None:
        self._events.append(TemporalEvent(event_type, affected_nodes, features, start, end))

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "HeterogeneousTemporalGraph":
        for v in self._node_data.values():
            if isinstance(v["features"], Tensor):
                v["features"] = v["features"].to(device)
        for v in self._edge_data.values():
            v["edge_index"] = v["edge_index"].to(device)
            if v.get("edge_features") is not None:
                v["edge_features"] = v["edge_features"].to(device)
        return self

    def clone(self) -> "HeterogeneousTemporalGraph":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["HeterogeneousTemporalGraph"]
        for nt_name, v in self._node_data.items():
            feat = v["features"]
            shape = tuple(feat.shape) if isinstance(feat, Tensor) else "?"
            lines.append(f"  nodes/{nt_name}: {shape}")
        for et_name, v in self._edge_data.items():
            ei = v["edge_index"]
            lines.append(f"  edges/{et_name}: {ei.shape[1]} edges")
        if self._events:
            lines.append(f"  events: {len(self._events)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_types = len(self._node_data)
        e_types = len(self._edge_data)
        return (
            f"HeterogeneousTemporalGraph("
            f"node_types={n_types}, edge_types={e_types}, "
            f"T={self.num_timesteps})"
        )
