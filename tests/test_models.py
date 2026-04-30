"""Tests for ModelRegistry, HGT, baselines, STGNN, and base model interface."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.baselines import ARIMABaseline, GRUBaseline, LSTMBaseline, ProphetBaseline
from graphoracle.models.registry import ModelRegistry
from graphoracle.utils.exceptions import ModelNotRegisteredError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_schema(with_edge: bool = True):
    nt = NodeType("sensor", features=["v", "t"], targets=["v"])
    edge_types = [EdgeType("link", "sensor", "sensor")] if with_edge else []
    return GraphSchema(node_types=[nt], edge_types=edge_types)


def _simple_graph(schema, N=5, T=10):
    feat = np.random.randn(N, T, len(schema.get_node_type("sensor").features)).astype(np.float32)
    ids = [f"s{i}" for i in range(N)]
    builder = GraphBuilder(schema).add_nodes("sensor", ids, feat)
    if schema.edge_types:
        builder.add_edges("link", [ids[0], ids[1]], [ids[1], ids[2]])
    return builder.build()


def _get_tensors(graph, device="cpu"):
    dev = torch.device(device)
    node_features = {
        nt.name: graph.get_node_features(nt.name).to(dev)
        for nt in graph.schema.node_types
    }
    edge_index = {
        et.name: graph.get_edge_index(et.name).to(dev)
        for et in graph.schema.edge_types
    }
    temporal_enc = torch.zeros(graph.num_timesteps, 16, device=dev)
    return node_features, edge_index, temporal_enc


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_get(self):
        class _Dummy:
            pass
        ModelRegistry.register("_test_dummy", _Dummy)
        assert ModelRegistry.get("_test_dummy") is _Dummy

    def test_register_as_decorator(self):
        @ModelRegistry.register("_test_deco")
        class _Deco:
            pass
        assert ModelRegistry.get("_test_deco") is _Deco

    def test_get_unknown_raises(self):
        with pytest.raises(ModelNotRegisteredError):
            ModelRegistry.get("this_model_does_not_exist_xyz")

    def test_available_lists_registered(self):
        avail = ModelRegistry.available()
        assert "lstm" in avail
        assert "gru" in avail
        assert "arima" in avail

    def test_register_overwrites(self):
        class _A:
            pass
        class _B:
            pass
        ModelRegistry.register("_overwrite_test", _A)
        ModelRegistry.register("_overwrite_test", _B)
        assert ModelRegistry.get("_overwrite_test") is _B


# ---------------------------------------------------------------------------
# LSTMBaseline
# ---------------------------------------------------------------------------

class TestLSTMBaseline:
    def test_forward_output_shape(self):
        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = LSTMBaseline(schema, horizons=[1, 6])
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert "sensor" in out
        assert 1 in out["sensor"]
        assert 6 in out["sensor"]
        # (N, target_dim=1)
        assert out["sensor"][1].shape == (5, 1)

    def test_required_history_steps(self):
        schema = _simple_schema()
        model = LSTMBaseline(schema, horizons=[1])
        assert model.required_history_steps() == 12

    def test_horizons_sorted(self):
        schema = _simple_schema()
        model = LSTMBaseline(schema, horizons=[6, 1, 3])
        assert model.horizons == [1, 3, 6]


# ---------------------------------------------------------------------------
# GRUBaseline
# ---------------------------------------------------------------------------

class TestGRUBaseline:
    def test_forward_output_shape(self):
        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = GRUBaseline(schema, horizons=[1])
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert out["sensor"][1].shape == (5, 1)

    def test_registered(self):
        assert ModelRegistry.get("gru") is GRUBaseline


# ---------------------------------------------------------------------------
# ARIMABaseline
# ---------------------------------------------------------------------------

class TestARIMABaseline:
    def test_forward_output_shape(self):
        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = ARIMABaseline(schema, horizons=[1, 3], ar_order=3)
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert out["sensor"][1].shape == (5, 1)

    def test_required_history_steps(self):
        schema = _simple_schema()
        model = ARIMABaseline(schema, horizons=[1], ar_order=5)
        assert model.required_history_steps() == 5


# ---------------------------------------------------------------------------
# ProphetBaseline
# ---------------------------------------------------------------------------

class TestProphetBaseline:
    def test_forward_output_shape(self):
        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = ProphetBaseline(schema, horizons=[1, 6])
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert out["sensor"][1].shape == (5, 1)


# ---------------------------------------------------------------------------
# HGT
# ---------------------------------------------------------------------------

class TestHGT:
    def test_hgt_forward(self):
        from graphoracle.models.hgt import HGT

        schema = _simple_schema()
        graph = _simple_graph(schema)
        node_types = schema.node_type_names
        edge_triplets = schema.edge_triplets()

        hgt = HGT(
            node_types=node_types,
            edge_triplets=edge_triplets,
            in_dim=2,
            hidden_dim=16,
            out_dim=8,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
        )
        hgt.eval()

        x_dict = {"sensor": torch.randn(5, 2)}
        # HGT expects {edge_name: (src_type, dst_type, edge_index)}
        ei = graph.get_edge_index("link")
        ei_dict = {"link": ("sensor", "sensor", ei)}
        out = hgt(x_dict, ei_dict)
        assert "sensor" in out
        assert out["sensor"].shape == (5, 8)

    def test_hgt_no_edges(self):
        from graphoracle.models.hgt import HGT

        hgt = HGT(
            node_types=["sensor"],
            edge_triplets=[("sensor", "link", "sensor")],
            in_dim=4,
            hidden_dim=8,
            out_dim=4,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        hgt.eval()
        x_dict = {"sensor": torch.randn(3, 4)}
        ei_dict = {"link": ("sensor", "sensor", torch.zeros(2, 0, dtype=torch.long))}
        out = hgt(x_dict, ei_dict)
        assert out["sensor"].shape == (3, 4)


# ---------------------------------------------------------------------------
# AdaptiveGraphLearner
# ---------------------------------------------------------------------------

class TestAdaptiveGraphLearner:
    def test_forward_shape(self):
        from graphoracle.models.adaptive_graph import AdaptiveGraphLearner

        learner = AdaptiveGraphLearner(num_nodes=6, embed_dim=8)
        adj = learner()
        assert adj.shape == (6, 6)

    def test_output_is_differentiable(self):
        from graphoracle.models.adaptive_graph import AdaptiveGraphLearner

        learner = AdaptiveGraphLearner(num_nodes=4, embed_dim=4)
        adj = learner()
        loss = adj.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# STGNN
# ---------------------------------------------------------------------------

class TestSTGNN:
    def test_forward_output(self):
        from graphoracle.models.stgnn import STGNN

        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = STGNN(schema, horizons=[1, 3])
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert "sensor" in out
        assert out["sensor"][1].shape[0] == 5  # N nodes


# ---------------------------------------------------------------------------
# GraphOracleModel
# ---------------------------------------------------------------------------

class TestGraphOracleModel:
    def test_forward_output(self):
        from graphoracle.models.graphoracle_model import GraphOracleModel

        schema = _simple_schema()
        graph = _simple_graph(schema)
        model = GraphOracleModel(
            schema=schema,
            horizons=[1],
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
        )
        model.eval()
        nf, ei, te = _get_tensors(graph)
        out = model(graph, nf, ei, te)
        assert "sensor" in out


# ---------------------------------------------------------------------------
# BaseForecastModel interface
# ---------------------------------------------------------------------------

class TestBaseForecastModelInterface:
    def test_custom_model_can_subclass(self):
        schema = _simple_schema()

        class _MyModel(BaseForecastModel):
            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                return {
                    nt.name: {h: torch.zeros(node_features[nt.name].shape[0], nt.target_dim or 1)
                              for h in self.horizons}
                    for nt in graph.schema.forecast_node_types
                }

            def required_history_steps(self):
                return 1

        model = _MyModel(schema, horizons=[1])
        assert model.horizons == [1]
        assert not model.supports_missing_nodes()

    def test_hooks_are_no_ops_by_default(self):
        schema = _simple_schema()
        graph = _simple_graph(schema)

        class _SimpleModel(BaseForecastModel):
            def forward(self, graph, node_features, edge_index, temporal_encoding, memory=None):
                return {}

            def required_history_steps(self):
                return 1

        m = _SimpleModel(schema, horizons=[1])
        m.on_fit_start(graph)    # no raise
        m.on_predict_start(graph)  # no raise
        m.reset_memory()          # no raise
