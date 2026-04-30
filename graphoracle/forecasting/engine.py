"""Forecast engine — runs inference and returns ForecastResult."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from graphoracle.forecasting.horizon import ForecastResult, NodeForecast
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.temporal import TEMPORAL_FEATURE_DIM, build_temporal_tensor
from graphoracle.models.base import BaseForecastModel


class ForecastEngine:
    """
    Wraps a trained model and handles inference bookkeeping.

    Parameters
    ----------
    model  : a fitted BaseForecastModel
    device : 'cpu' | 'cuda' | 'mps'
    """

    def __init__(self, model: BaseForecastModel, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def run(
        self,
        graph: HeterogeneousTemporalGraph,
        reference_time: Any = None,
        horizon_unit_hours: int = 1,
    ) -> ForecastResult:
        """Run inference and return a ForecastResult."""
        raw = self._run_inference(graph)
        return ForecastResult(
            raw_predictions=raw,
            graph=graph,
            reference_time=reference_time,
            horizon_unit_hours=horizon_unit_hours,
        )

    @torch.no_grad()
    def _run_inference(
        self, graph: HeterogeneousTemporalGraph
    ) -> dict[str, dict[int, Tensor]]:
        """
        Run the model forward pass on *graph* and return raw predictions.

        Returns
        -------
        {node_type: {horizon: (N, target_dim) Tensor}}
        """
        model = self.model
        device = torch.device(self.device)
        model.eval()
        model.to(device)
        model.on_predict_start(graph)

        node_features: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = graph.get_node_features(nt.name)
            if feat.numel() > 0:
                node_features[nt.name] = feat.to(device)

        edge_index: dict[str, Tensor] = {}
        for et in graph.schema.edge_types:
            ei = graph.get_edge_index(et.name)
            edge_index[et.name] = ei.to(device)

        ts = graph.timestamps
        if ts:
            temporal_enc = build_temporal_tensor(ts, TEMPORAL_FEATURE_DIM).to(device)
        else:
            T = max((v.shape[1] for v in node_features.values()), default=1)
            temporal_enc = torch.zeros(T, TEMPORAL_FEATURE_DIM, device=device)

        return model(graph, node_features, edge_index, temporal_enc)
