"""Tests for custom model registration and BaseForecastModel subclassing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.registry import ModelRegistry
from graphoracle.oracle import GraphOracle
from graphoracle.training.trainer import TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema():
    return GraphSchema(
        node_types=[NodeType("node", features=["x", "y"], targets=["x"])],
        edge_types=[EdgeType("conn", "node", "node")],
    )


def _graph(schema, N=4, T=6):
    feat = np.random.randn(N, T, 2).astype(np.float32)
    ids = [f"n{i}" for i in range(N)]
    return (
        GraphBuilder(schema)
        .add_nodes("node", ids, feat)
        .add_edges("conn", [ids[0], ids[1]], [ids[1], ids[2]])
        .build()
    )


# ---------------------------------------------------------------------------
# Level 1: Custom model registered with ModelRegistry
# ---------------------------------------------------------------------------

class TestLevel1CustomRegistration:
    def test_register_and_use_via_oracle(self):
        """Register a custom model class and use it via GraphOracle."""

        @ModelRegistry.register("_custom_mlp_test")
        class CustomMLP(BaseForecastModel):
            def __init__(self, schema, horizons, hidden=32, **kwargs):
                super().__init__(schema, horizons, **kwargs)
                self.linears = nn.ModuleDict({
                    nt.name: nn.Linear(max(nt.feature_dim, 1), max(nt.target_dim, 1))
                    for nt in schema.forecast_node_types
                })

            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                out = {}
                for nt in graph.schema.forecast_node_types:
                    feat = node_features.get(nt.name)
                    if feat is None:
                        continue
                    last = feat[:, -1, :]  # (N, F)
                    out[nt.name] = {h: self.linears[nt.name](last) for h in self.horizons}
                return out

            def required_history_steps(self):
                return 1

        schema = _schema()
        graph = _graph(schema)
        oracle = GraphOracle(model="_custom_mlp_test", horizons=[1])
        oracle.fit(graph, config=TrainingConfig(epochs=1))
        result = oracle.predict(graph)
        assert result is not None

    def test_registered_model_appears_in_available(self):
        @ModelRegistry.register("_avail_test_model")
        class _AvailModel(BaseForecastModel):
            def forward(self, *a, **kw):
                return {}
            def required_history_steps(self):
                return 1

        assert "_avail_test_model" in ModelRegistry.available()


# ---------------------------------------------------------------------------
# Level 2: Direct BaseForecastModel subclassing + Trainer
# ---------------------------------------------------------------------------

class TestLevel2DirectSubclassing:
    def test_custom_model_trains_with_trainer(self):
        from graphoracle.training.trainer import Trainer

        schema = _schema()
        graph = _graph(schema)

        class _GRUModel(BaseForecastModel):
            def __init__(self, schema, horizons, **kwargs):
                super().__init__(schema, horizons, **kwargs)
                self.gru = nn.GRU(
                    max(schema.get_node_type("node").feature_dim, 1),
                    16,
                    batch_first=True,
                )
                self.heads = nn.ModuleDict({
                    f"h{h}": nn.Linear(16, max(schema.target_dim("node"), 1))
                    for h in horizons
                })

            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                feat = node_features.get("node")
                if feat is None:
                    return {}
                _, h_n = self.gru(feat)
                h = h_n[-1]
                return {"node": {hop: self.heads[f"h{hop}"](h) for hop in self.horizons}}

            def required_history_steps(self):
                return 3

        model = _GRUModel(schema, horizons=[1, 3])
        trainer = Trainer(model, TrainingConfig(epochs=2))
        history = trainer.fit(graph)
        assert len(history.train_losses) > 0

    def test_custom_model_on_fit_start_called(self):
        schema = _schema()
        graph = _graph(schema)
        called = []

        class _Tracker(BaseForecastModel):
            def __init__(self, schema, horizons, **kw):
                super().__init__(schema, horizons, **kw)
                self.dummy = nn.Linear(2, 1)  # needs params for optimizer

            def on_fit_start(self, g):
                called.append("fit_start")

            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                out = {}
                for nt in graph.schema.forecast_node_types:
                    feat = node_features.get(nt.name)
                    if feat is not None:
                        N = feat.shape[0]
                        out[nt.name] = {h: self.dummy(feat[:, -1, :])
                                        for h in self.horizons}
                return out

            def required_history_steps(self):
                return 1

        from graphoracle.training.trainer import Trainer
        model = _Tracker(schema, horizons=[1])
        trainer = Trainer(model, TrainingConfig(epochs=1))
        trainer.fit(graph)
        assert "fit_start" in called

    def test_custom_model_supports_missing_nodes_override(self):
        schema = _schema()

        class _Partial(BaseForecastModel):
            def forward(self, *a, **kw):
                return {}
            def required_history_steps(self):
                return 1
            def supports_missing_nodes(self):
                return True

        m = _Partial(schema, horizons=[1])
        assert m.supports_missing_nodes() is True


# ---------------------------------------------------------------------------
# Quantile output head
# ---------------------------------------------------------------------------

class TestQuantileOutputModel:
    def test_quantile_output_trains(self):
        from graphoracle.training.loss import quantile_loss

        schema = _schema()
        graph = _graph(schema)

        # Model that outputs 3 quantiles per target
        class _QuantileModel(BaseForecastModel):
            def __init__(self, schema, horizons, **kwargs):
                super().__init__(schema, horizons, **kwargs)
                self.head = nn.Linear(2, 3)  # F → 3 quantiles

            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                feat = node_features.get("node")
                if feat is None:
                    return {}
                last = feat[:, -1, :]
                q_out = self.head(last)  # (N, 3)
                return {"node": {h: q_out for h in self.horizons}}

            def required_history_steps(self):
                return 1

        from graphoracle.training.trainer import Trainer
        model = _QuantileModel(schema, horizons=[1])
        trainer = Trainer(model, TrainingConfig(epochs=2, loss="mae"))
        history = trainer.fit(graph)
        assert len(history.train_losses) > 0
