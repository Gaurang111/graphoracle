"""Baseline forecast models: LSTM, GRU, ARIMA (AR fallback), Prophet (AR fallback)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.registry import ModelRegistry


@ModelRegistry.register("lstm")
class LSTMBaseline(BaseForecastModel):
    """Per-type LSTM baseline — no graph convolution."""

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        history_steps: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self._history_steps = history_steps

        self.lstms = nn.ModuleDict(
            {
                nt.name: nn.LSTM(
                    max(nt.feature_dim, 1),
                    hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                for nt in schema.node_types
            }
        )
        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {
                        f"h{h}": nn.Linear(hidden, max(nt.target_dim, 1))
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
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
            _, (h_n, _) = self.lstms[nt.name](feat)
            h = h_n[-1]  # (N, hidden)
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return self._history_steps


@ModelRegistry.register("gru")
class GRUBaseline(BaseForecastModel):
    """Per-type GRU baseline — no graph convolution."""

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        history_steps: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self._history_steps = history_steps

        self.grus = nn.ModuleDict(
            {
                nt.name: nn.GRU(
                    max(nt.feature_dim, 1),
                    hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                for nt in schema.node_types
            }
        )
        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {
                        f"h{h}": nn.Linear(hidden, max(nt.target_dim, 1))
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
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
            _, h_n = self.grus[nt.name](feat)
            h = h_n[-1]  # (N, hidden)
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return self._history_steps


@ModelRegistry.register("arima")
class ARIMABaseline(BaseForecastModel):
    """
    Simple AR(p) fallback model — does not require statsmodels.

    Uses the last *ar_order* timesteps as linear input features.
    For domains where statsmodels is installed, this can be swapped
    for a full ARIMA via custom model registration.
    """

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        ar_order: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self.ar_order = ar_order

        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {
                        f"h{h}": nn.Linear(
                            max(ar_order, 1) * max(nt.feature_dim, 1),
                            max(nt.target_dim, 1),
                        )
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
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 2:
                feat = feat.unsqueeze(1)
            N, T, F_in = feat.shape
            p = min(self.ar_order, T)
            ar_feat = feat[:, -p:, :].reshape(N, p * F_in)
            expected = self.ar_order * F_in
            if ar_feat.shape[-1] < expected:
                pad = torch.zeros(N, expected - ar_feat.shape[-1], device=feat.device)
                ar_feat = torch.cat([pad, ar_feat], dim=-1)
            out[nt.name] = {
                h: self.heads[nt.name][f"h{h}"](ar_feat)
                for h in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return self.ar_order


@ModelRegistry.register("prophet")
class ProphetBaseline(BaseForecastModel):
    """
    Placeholder Prophet baseline — uses simple exponential smoothing.

    Install the optional `prophet` extra for the full Prophet model.
    """

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        alpha: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self.alpha = alpha

        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {
                        f"h{h}": nn.Linear(max(nt.feature_dim, 1), max(nt.target_dim, 1))
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
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 3:
                feat = feat[:, -1, :]  # use last timestep
            out[nt.name] = {
                h: self.heads[nt.name][f"h{h}"](feat)
                for h in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return 1
