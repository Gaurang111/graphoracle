"""
Custom Model Example — All Three Levels
========================================
Shows Level 1 (backbone swap), Level 2 (full subclass),
and Level 3 (global registry) APIs.

Run: python examples/custom_model_example.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle import GraphOracle
from graphoracle.domains import ElectricGridDomain
from graphoracle.graph import HeterogeneousTemporalGraph
from graphoracle.models import BaseForecastModel, ModelRegistry
from graphoracle.training import TrainingConfig

# ──────────────────────────────────────────────────────────────────────────────
# Shared graph
# ──────────────────────────────────────────────────────────────────────────────

domain = ElectricGridDomain()
graph = domain.build_synthetic_graph(
    nodes_per_type={
        "substation": 6,
        "weather_station": 2,
        "renewable_source": 2,
        "industrial_consumer": 2,
        "residential_zone": 2,
        "transmission_line": 1,
        "market_signal": 1,
        "event_node": 1,
    },
    num_timesteps=32,
)
QUICK_CFG = TrainingConfig(epochs=3, device="cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Level 1: Swap the backbone
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Level 1 — Backbone swap")

class TwoLayerMLP(nn.Module):
    """Drop in any nn.Module that takes (x, edge_index)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.ReLU(),
            nn.Linear(64, out_channels),
        )
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.net(x)

# substation has 5 features
oracle_l1 = GraphOracle(
    backbone=TwoLayerMLP,
    backbone_kwargs={"in_channels": 5, "out_channels": 32},
    horizons=[1, 6],
)
history = oracle_l1.fit(graph, config=QUICK_CFG)
print(f"  Best val loss: {history.best_val_loss():.4f}")
forecast = oracle_l1.predict(graph)
print(f"  Forecasted {len(forecast.all_nodes())} nodes")

# ──────────────────────────────────────────────────────────────────────────────
# Level 2: Subclass BaseForecastModel
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Level 2 — Full custom model")

class ResidualGNNModel(BaseForecastModel):
    """
    Simple GNN with residual connections.
    Demonstrates full forward-pass control.
    """

    def __init__(self, schema, horizons, hidden: int = 64, **kwargs):
        super().__init__(schema, horizons, **kwargs)
        self.encoders = nn.ModuleDict({
            nt.name: nn.Linear(max(nt.feature_dim, 1), hidden)
            for nt in schema.node_types
        })
        self.gnn = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleDict({
            nt.name: nn.ModuleDict({
                f"h{h}": nn.Linear(hidden, max(nt.target_dim, 1))
                for h in horizons
            })
            for nt in schema.forecast_node_types
        })

    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        temporal_encoding: Tensor,
        memory=None,
    ) -> dict[str, dict[int, Tensor]]:
        # Encode per node type
        x_dict: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 3:
                feat = feat.mean(1)   # (N, T, F) → (N, F)
            x_dict[nt.name] = torch.relu(self.encoders[nt.name](feat))

        # Simple neighbourhood aggregation
        for et in graph.schema.edge_types:
            ei = edge_index.get(et.name)
            if ei is None or ei.numel() == 0:
                continue
            src, dst = ei
            src_type, dst_type = et.src_type, et.dst_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            msgs = self.gnn(x_dict[src_type][src])
            agg = torch.zeros_like(x_dict[dst_type])
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
            x_dict[dst_type] = self.norm(x_dict[dst_type] + 0.1 * agg)

        # Forecast heads
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            if nt.name not in x_dict:
                continue
            h = x_dict[nt.name]
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return 8

    def supports_missing_nodes(self) -> bool:
        return True

ModelRegistry.register("residual_gnn", ResidualGNNModel)

oracle_l2 = GraphOracle(model="residual_gnn", horizons=[1, 6])
history = oracle_l2.fit(graph, config=QUICK_CFG)
print(f"  Best val loss: {history.best_val_loss():.4f}")
forecast = oracle_l2.predict(graph)
df = forecast.to_dataframe()
print(f"  Forecast DataFrame shape: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# Level 3: Register globally and use from config
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Level 3 — Config-driven usage")

config_dict = {
    "model": "residual_gnn",
    "horizons": [1, 6, 24],
    "device": "cpu",
    "model_kwargs": {"hidden": 32},
}
oracle_l3 = GraphOracle.from_config(config_dict)
oracle_l3.fit(graph, config=QUICK_CFG)
print(f"  Oracle from config: {oracle_l3}")
print(f"  Available models: {ModelRegistry.available()}")
