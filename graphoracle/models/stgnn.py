"""Spatiotemporal GNN (simplified DCRNN-style)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.registry import ModelRegistry


@ModelRegistry.register("stgnn")
class STGNN(BaseForecastModel):
    """
    Spatiotemporal GNN — spatial mixing via GCN, temporal via GRU.

    Inspired by DCRNN (Yao et al., 2018).  The spatial step uses
    mean-aggregation graph convolution; the temporal step uses a GRU.
    """

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        embed_dim: int = 64,
        num_spatial_layers: int = 2,
        history_steps: int = 12,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self._history_steps = history_steps

        self.encoders = nn.ModuleDict(
            {
                nt.name: nn.Linear(max(nt.feature_dim, 1), embed_dim)
                for nt in schema.node_types
            }
        )

        self.spatial_layers = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_spatial_layers)]
        )
        self.spatial_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_spatial_layers)]
        )

        self.grus = nn.ModuleDict(
            {
                nt.name: nn.GRU(embed_dim, embed_dim, batch_first=True)
                for nt in schema.node_types
            }
        )

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {
                        f"h{h}": nn.Linear(embed_dim, max(nt.target_dim, 1))
                        for h in horizons
                    }
                )
                for nt in schema.forecast_node_types
            }
        )

    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        temporal_encoding: Tensor,
        memory: dict[str, Tensor] | None = None,
    ) -> dict[str, dict[int, Tensor]]:
        # Pick first available edge index for spatial mixing
        ei = next(iter(edge_index.values()), torch.empty((2, 0), dtype=torch.long))

        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
            N, T, _ = feat.shape

            h = self.encoders[nt.name](feat)  # (N, T, E)

            # Spatial mixing on temporal mean
            if ei.numel() > 0 and ei.max() < N:
                src, dst = ei
                for gcn, norm in zip(self.spatial_layers, self.spatial_norms):
                    h_mean = h.mean(1)  # (N, E)
                    neigh = torch.zeros_like(h_mean)
                    neigh.scatter_add_(
                        0, dst.unsqueeze(1).expand_as(h_mean[src]), h_mean[src]
                    )
                    deg = torch.zeros(N, device=h.device)
                    deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=h.device))
                    deg = deg.clamp(min=1).unsqueeze(1)
                    h_spatial = norm(torch.relu(gcn(h_mean + neigh / deg)))
                    h = h + h_spatial.unsqueeze(1)

            # Temporal GRU
            h_gru, _ = self.grus[nt.name](h)
            h_last = h_gru[:, -1, :]

            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](self.dropout(h_last))
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return self._history_steps
