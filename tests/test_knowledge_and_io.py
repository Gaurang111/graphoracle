"""Tests for EdgeDiscovery, EventInjector, and I/O utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType
from graphoracle.knowledge.edge_discovery import EdgeDiscovery
from graphoracle.knowledge.external_events import EventInjector, EventSpec
from graphoracle.utils.exceptions import EdgeDiscoveryError
from graphoracle.utils.io import load_graph, load_model, save_graph, save_model


# ---------------------------------------------------------------------------
# EdgeDiscovery
# ---------------------------------------------------------------------------

class TestEdgeDiscovery:
    def _ts(self, N=8, T=20):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((N, T)).astype(np.float64)
        # Make first two correlated
        ts[1] = ts[0] + 0.1 * rng.standard_normal(T)
        return ts

    def test_correlation_discovers_edges(self):
        disc = EdgeDiscovery(method="correlation", threshold=0.5)
        ids = [f"n{i}" for i in range(8)]
        ts = self._ts()
        edges = disc.discover(ids, ts)
        assert isinstance(edges, list)
        # The two correlated nodes should be connected
        assert ("n0", "n1") in edges or ("n1", "n0") in edges

    def test_correlation_returns_tuples(self):
        disc = EdgeDiscovery(method="correlation", threshold=0.0)
        ids = ["a", "b", "c"]
        ts = np.eye(3)
        edges = disc.discover(ids, ts)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 2

    def test_mutual_info_fallback(self):
        disc = EdgeDiscovery(method="mutual_info", threshold=0.0)
        ids = [f"n{i}" for i in range(4)]
        ts = np.random.randn(4, 30)
        edges = disc.discover(ids, ts)
        assert isinstance(edges, list)

    def test_spatial_knn(self):
        disc = EdgeDiscovery(method="spatial", threshold=0.0)
        ids = [f"n{i}" for i in range(6)]
        ts = np.random.randn(6, 10)
        edges = disc.discover(ids, ts, )
        assert len(edges) > 0

    def test_unsupported_method_raises(self):
        with pytest.raises(EdgeDiscoveryError):
            EdgeDiscovery(method="unsupported_xyz")

    def test_mismatched_timeseries_raises(self):
        disc = EdgeDiscovery(method="correlation")
        ids = ["a", "b"]
        ts = np.zeros((5, 10))  # 5 rows but 2 ids
        with pytest.raises(EdgeDiscoveryError):
            disc.discover(ids, ts)

    def test_granger_falls_back(self):
        disc = EdgeDiscovery(method="granger", threshold=0.0)
        ids = ["a", "b", "c"]
        ts = np.random.randn(3, 20)
        edges = disc.discover(ids, ts)
        assert isinstance(edges, list)

    def test_high_threshold_returns_few_edges(self):
        disc = EdgeDiscovery(method="correlation", threshold=0.99)
        ids = [f"n{i}" for i in range(6)]
        ts = np.eye(6)
        edges = disc.discover(ids, ts)
        # Diagonal of identity has no cross correlations above 0.99
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# EventSpec and EventInjector
# ---------------------------------------------------------------------------

class TestEventSpec:
    def test_fields(self):
        spec = EventSpec(
            event_type="storm",
            affected_nodes=["n0", "n1"],
            features={"severity": 0.8},
            start="2024-01-01",
            end="2024-01-02",
        )
        assert spec.event_type == "storm"
        assert len(spec.affected_nodes) == 2
        assert spec.features["severity"] == 0.8


class TestEventInjector:
    def _graph(self):
        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value", "severity"], targets=["value"])],
            edge_types=[EdgeType("link", "sensor", "sensor")],
        )
        feat = np.ones((3, 5, 2), dtype=np.float32)
        return (
            GraphBuilder(schema)
            .add_nodes("sensor", ["s0", "s1", "s2"], feat)
            .add_edges("link", ["s0"], ["s1"])
            .build()
        )

    def test_apply_injects_events(self):
        injector = EventInjector()
        spec = EventSpec("flood", ["s0", "s1"], {"severity": 1.0}, "t0", "t1")
        injector.add_event(spec)
        graph = self._graph()
        injector.apply(graph)
        assert len(graph.events) == 1

    def test_multiple_events(self):
        injector = EventInjector([
            EventSpec("storm", ["s0"], {"severity": 0.5}, "t0", "t1"),
            EventSpec("outage", ["s1", "s2"], {"severity": 0.9}, "t1", "t2"),
        ])
        graph = self._graph()
        injector.apply(graph)
        assert len(graph.events) == 2

    def test_build_event_tensor_shape(self):
        spec = EventSpec("flood", ["s0", "s1"], {"severity": 1.0}, "t0", "t1")
        injector = EventInjector([spec])
        graph = self._graph()
        injector.apply(graph)
        tensor = injector.build_event_tensor(graph, "sensor", ["value", "severity"])
        # (N=3, T=5, F=2)
        assert tensor.shape == (3, 5, 2)

    def test_build_event_tensor_values(self):
        spec = EventSpec("flood", ["s0"], {"severity": 0.75}, "t0", "t1")
        injector = EventInjector([spec])
        graph = self._graph()
        injector.apply(graph)
        tensor = injector.build_event_tensor(graph, "sensor", ["severity"])
        # s0 (index 0) should have severity=0.75
        assert abs(float(tensor[0, 0, 0]) - 0.75) < 1e-5

    def test_add_event_method(self):
        injector = EventInjector()
        injector.add_event(EventSpec("test", ["s0"], {}, "t0", "t1"))
        assert len(injector.event_specs) == 1

    def test_apply_returns_graph(self):
        injector = EventInjector()
        graph = self._graph()
        result = injector.apply(graph)
        assert result is graph


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

class TestIOUtils:
    def test_save_and_load_model(self, tmp_path):
        model = nn.Linear(4, 2)
        path = tmp_path / "model.pt"
        save_model(model, path)
        assert path.exists()

        model2 = nn.Linear(4, 2)
        model2.weight.data.fill_(0.0)
        load_model(model2, path)
        assert not (model2.weight.data == 0.0).all()

    def test_load_model_missing_raises(self, tmp_path):
        from graphoracle.utils.exceptions import GraphOracleError
        model = nn.Linear(2, 1)
        with pytest.raises(GraphOracleError):
            load_model(model, tmp_path / "nonexistent.pt")

    def test_save_and_load_graph(self, tmp_path):
        from graphoracle.graph.schema import GraphSchema, NodeType

        schema = GraphSchema(
            node_types=[NodeType("x", features=["v"], targets=["v"])],
            edge_types=[],
        )
        feat = np.zeros((2, 3, 1), dtype=np.float32)
        graph = GraphBuilder(schema).add_nodes("x", ["a", "b"], feat).build()

        path = tmp_path / "graph.pkl"
        save_graph(graph, path)
        assert path.exists()

        loaded = load_graph(path)
        assert loaded.num_nodes("x") == 2

    def test_load_graph_missing_raises(self, tmp_path):
        from graphoracle.utils.exceptions import GraphOracleError
        with pytest.raises(GraphOracleError):
            load_graph(tmp_path / "nonexistent.pkl")

    def test_save_model_creates_parent_dirs(self, tmp_path):
        model = nn.Linear(2, 1)
        path = tmp_path / "nested" / "dir" / "model.pt"
        save_model(model, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# JSON connector
# ---------------------------------------------------------------------------

class TestJSONConnector:
    def test_load_from_json(self, tmp_path):
        from graphoracle.connectors.json_connector import JSONConnector

        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value"], targets=["value"])],
            edge_types=[EdgeType("link", "sensor", "sensor")],
        )
        data = {
            "nodes": {
                "sensor": [
                    {"id": "s0", "value": 1.0},
                    {"id": "s1", "value": 2.0},
                ]
            },
            "edges": {
                "link": [{"src": "s0", "dst": "s1"}]
            },
        }
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        connector = JSONConnector(schema)
        graph = connector.load(path)
        assert graph.num_nodes("sensor") == 2

    def test_load_with_timeseries(self, tmp_path):
        from graphoracle.connectors.json_connector import JSONConnector

        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value"], targets=["value"])],
            edge_types=[],
        )
        data = {
            "nodes": {"sensor": [{"id": "s0", "value": 1.0}, {"id": "s1", "value": 2.0}]},
            "edges": {},
            "timeseries": {
                "s0": [{"timestamp": "t0", "value": 1.0}, {"timestamp": "t1", "value": 1.1}],
                "s1": [{"timestamp": "t0", "value": 2.0}, {"timestamp": "t1", "value": 2.1}],
            },
        }
        path = tmp_path / "ts_data.json"
        path.write_text(json.dumps(data))

        connector = JSONConnector(schema)
        graph = connector.load(path)
        assert graph.num_timesteps == 2


# ---------------------------------------------------------------------------
# DataFrame connector
# ---------------------------------------------------------------------------

class TestDataFrameConnector:
    def test_load_from_dataframes(self):
        pd = pytest.importorskip("pandas")
        import pandas as pd
        from graphoracle.connectors.dataframe_connector import DataFrameConnector

        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value", "temp"], targets=["value"])],
            edge_types=[EdgeType("link", "sensor", "sensor")],
        )

        node_df = pd.DataFrame({
            "node_id": ["s0", "s1", "s2"],
            "value": [1.0, 2.0, 3.0],
            "temp": [10.0, 20.0, 30.0],
        })
        edge_df = pd.DataFrame({
            "src_id": ["s0", "s1"],
            "dst_id": ["s1", "s2"],
        })

        connector = DataFrameConnector(schema)
        graph = connector.load(
            node_dfs={"sensor": node_df},
            edge_dfs={"link": edge_df},
            id_col="node_id",
        )
        assert graph.num_nodes("sensor") == 3
