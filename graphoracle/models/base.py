"""Abstract base class for all forecast models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema


class BaseForecastModel(nn.Module, ABC):
    """
    Abstract base for every GraphOracle forecast model.

    Subclasses must implement:
      - ``forward(...)`` — return ``{node_type: {horizon: predictions}}``
      - ``required_history_steps()`` — minimum past timesteps needed

    Subclasses may override:
      - ``supports_missing_nodes()`` — default False
      - ``on_fit_start(graph)``       — called once before training
      - ``on_predict_start(graph)``   — called before inference

    graphoracle handles: data loading, training loop, evaluation,
    checkpointing, uncertainty wrapping, and explainability hooks.
    Users handle: the forward pass.
    """

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.horizons = sorted(horizons)
        self._kwargs = kwargs

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        temporal_encoding: Tensor,
        memory: dict[str, Tensor] | None = None,
    ) -> dict[str, dict[int, Tensor]]:
        """
        Perform a forward pass.

        Parameters
        ----------
        graph             : full graph object (for schema / event access)
        node_features     : {node_type: (N, T, F)} current window features
        edge_index        : {edge_type: (2, E)} adjacency
        temporal_encoding : (T, TEMPORAL_FEATURE_DIM) time features
        memory            : optional {node_type: (N, D)} persistent memory

        Returns
        -------
        {node_type: {horizon: (N, target_dim)}}
        """

    @abstractmethod
    def required_history_steps(self) -> int:
        """Minimum number of past timesteps the model consumes."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def supports_missing_nodes(self) -> bool:
        """Return True if the model can handle partially observed nodes."""
        return False

    def on_fit_start(self, graph: HeterogeneousTemporalGraph) -> None:
        """Called once before the training loop starts."""

    def on_predict_start(self, graph: HeterogeneousTemporalGraph) -> None:
        """Called once before inference starts."""

    def reset_memory(self) -> None:
        """Reset any persistent node memory (TGN-style).  No-op by default."""

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def node_types(self) -> list[str]:
        return self.schema.node_type_names

    def forecast_node_types(self) -> list[str]:
        return [nt.name for nt in self.schema.forecast_node_types]

    def target_dim(self, node_type: str) -> int:
        return self.schema.target_dim(node_type)

    def node_feature_dim(self, node_type: str) -> int:
        return self.schema.node_dim(node_type)

    def extra_repr(self) -> str:
        return (
            f"schema={self.schema!r}, "
            f"horizons={self.horizons}, "
            f"history={self.required_history_steps()}"
        )
