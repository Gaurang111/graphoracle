"""Tests for graph schema, heterogeneous graph, builder, temporal encoding, and validators."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph, TemporalEvent
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType
from graphoracle.graph.temporal import (
    TEMPORAL_FEATURE_DIM,
    build_temporal_tensor,
    sinusoidal_encoding,
    time_delta_encoding,
)
from graphoracle.graph.validators import (
    check_no_inf,
    check_no_nan,
    validate_edge_index,
    validate_feature_tensor,
    validate_graph_has_targets,
)
from graphoracle.utils.exceptions import (
    GraphOracleError,
    GraphSchemaError,
    MissingTargetFeatureError,
)


# ---------------------------------------------------------------------------
# NodeType
# ---------------------------------------------------------------------------

class TestNodeType:
    def test_feature_dim(self):
        nt = NodeType("x", features=["a", "b", "c"])
        assert nt.feature_dim == 3

    def test_target_dim(self):
        nt = NodeType("x", targets=["y"])
        assert nt.target_dim == 1

    def test_defaults_empty(self):
        nt = NodeType("x")
        assert nt.features == []
        assert nt.targets == []
        assert nt.spatial_features == []
        assert nt.metadata == {}

    def test_repr(self):
        nt = NodeType("sensor", features=["v"])
        assert "sensor" in repr(nt)


# ---------------------------------------------------------------------------
# EdgeType
# ---------------------------------------------------------------------------

class TestEdgeType:
    def test_fields(self):
        et = EdgeType("link", "a", "b")
        assert et.name == "link"
        assert et.src_type == "a"
        assert et.dst_type == "b"

    def test_repr(self):
        et = EdgeType("road", "sensor", "sensor")
        assert "road" in repr(et)


# ---------------------------------------------------------------------------
# GraphSchema
# ---------------------------------------------------------------------------

class TestGraphSchema:
    def _schema(self):
        return GraphSchema(
            node_types=[
                NodeType("sensor", features=["v", "t"], targets=["v"]),
                NodeType("station", features=["temp"]),
            ],
            edge_types=[
                EdgeType("link", "sensor", "sensor"),
                EdgeType("influence", "station", "sensor"),
            ],
        )

    def test_node_type_names(self):
        s = self._schema()
        assert "sensor" in s.node_type_names
        assert "station" in s.node_type_names

    def test_forecast_node_types_only_those_with_targets(self):
        s = self._schema()
        names = [nt.name for nt in s.forecast_node_types]
        assert "sensor" in names
        assert "station" not in names

    def test_get_node_type(self):
        s = self._schema()
        nt = s.get_node_type("sensor")
        assert nt.name == "sensor"

    def test_get_node_type_missing_raises(self):
        s = self._schema()
        with pytest.raises(GraphSchemaError):
            s.get_node_type("nonexistent")

    def test_node_dim(self):
        s = self._schema()
        assert s.node_dim("sensor") == 2

    def test_target_dim(self):
        s = self._schema()
        assert s.target_dim("sensor") == 1

    def test_edge_triplets(self):
        s = self._schema()
        triplets = s.edge_triplets()
        assert ("sensor", "link", "sensor") in triplets
        assert ("station", "influence", "sensor") in triplets

    def test_invalid_edge_src_raises(self):
        with pytest.raises(GraphSchemaError):
            GraphSchema(
                node_types=[NodeType("a")],
                edge_types=[EdgeType("e", "nonexistent", "a")],
            )

    def test_invalid_edge_dst_raises(self):
        with pytest.raises(GraphSchemaError):
            GraphSchema(
                node_types=[NodeType("a")],
                edge_types=[EdgeType("e", "a", "nonexistent")],
            )

    def test_repr(self):
        s = self._schema()
        r = repr(s)
        assert "GraphSchema" in r


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def _simple_schema(self):
        return GraphSchema(
            node_types=[NodeType("s", features=["v"], targets=["v"])],
            edge_types=[EdgeType("link", "s", "s")],
        )

    def test_build_returns_graph(self):
        schema = self._simple_schema()
        feat = np.ones((3, 5, 1), dtype=np.float32)
        graph = (
            GraphBuilder(schema)
            .add_nodes("s", ["a", "b", "c"], feat)
            .add_edges("link", ["a", "b"], ["b", "c"])
            .build()
        )
        assert isinstance(graph, HeterogeneousTemporalGraph)
        assert graph.num_nodes("s") == 3

    def test_add_nodes_2d_features(self):
        schema = self._simple_schema()
        feat = np.ones((3, 1), dtype=np.float32)
        graph = GraphBuilder(schema).add_nodes("s", ["a", "b", "c"], feat).build()
        # Should be expanded to (N, 1, F)
        assert graph.get_node_features("s").ndim == 3

    def test_add_edges_builds_correct_index(self):
        schema = self._simple_schema()
        feat = np.ones((3, 1, 1), dtype=np.float32)
        graph = (
            GraphBuilder(schema)
            .add_nodes("s", ["a", "b", "c"], feat)
            .add_edges("link", ["a"], ["b"])
            .build()
        )
        ei = graph.get_edge_index("link")
        assert ei.shape == (2, 1)

    def test_unknown_node_type_raises(self):
        schema = self._simple_schema()
        with pytest.raises(GraphSchemaError):
            GraphBuilder(schema).add_nodes("nonexistent", ["x"], np.zeros((1, 1)))

    def test_unknown_edge_src_raises(self):
        schema = self._simple_schema()
        feat = np.ones((2, 1, 1), dtype=np.float32)
        builder = GraphBuilder(schema).add_nodes("s", ["a", "b"], feat)
        with pytest.raises(ValueError):
            builder.add_edges("link", ["MISSING"], ["a"])

    def test_inject_event(self):
        schema = self._simple_schema()
        feat = np.ones((2, 1, 1), dtype=np.float32)
        graph = (
            GraphBuilder(schema)
            .add_nodes("s", ["a", "b"], feat)
            .inject_event("storm", ["a"], {"severity": 0.9}, "t0", "t1")
            .build()
        )
        assert len(graph.events) == 1

    def test_timestamps_stored(self):
        schema = self._simple_schema()
        feat = np.ones((2, 3, 1), dtype=np.float32)
        ts = [0, 1, 2]
        graph = (
            GraphBuilder(schema)
            .add_nodes("s", ["a", "b"], feat, timestamps=ts)
            .build()
        )
        assert graph.timestamps == ts


# ---------------------------------------------------------------------------
# HeterogeneousTemporalGraph
# ---------------------------------------------------------------------------

class TestHeterogeneousTemporalGraph:
    def _build(self):
        schema = GraphSchema(
            node_types=[NodeType("s", features=["v"], targets=["v"])],
            edge_types=[EdgeType("link", "s", "s")],
        )
        feat = np.random.randn(4, 6, 1).astype(np.float32)
        return (
            GraphBuilder(schema)
            .add_nodes("s", [f"n{i}" for i in range(4)], feat)
            .add_edges("link", ["n0", "n1"], ["n1", "n2"])
            .build()
        )

    def test_num_nodes(self):
        g = self._build()
        assert g.num_nodes("s") == 4

    def test_num_nodes_missing_type_returns_zero(self):
        g = self._build()
        assert g.num_nodes("nonexistent") == 0

    def test_num_timesteps(self):
        g = self._build()
        assert g.num_timesteps == 6

    def test_all_node_ids(self):
        g = self._build()
        ids = g.all_node_ids("s")
        assert ids == ["n0", "n1", "n2", "n3"]

    def test_get_node_features_shape(self):
        g = self._build()
        feat = g.get_node_features("s")
        assert feat.shape == (4, 6, 1)

    def test_get_edge_index_shape(self):
        g = self._build()
        ei = g.get_edge_index("link")
        assert ei.shape[0] == 2
        assert ei.shape[1] == 2

    def test_get_edge_index_missing_returns_empty(self):
        g = self._build()
        ei = g.get_edge_index("nonexistent")
        assert ei.shape == (2, 0)

    def test_inject_event(self):
        g = self._build()
        g.inject_event("flood", ["n0"], {"severity": 1.0}, "2024-01", "2024-02")
        assert len(g.events) == 1
        assert g.events[0].event_type == "flood"

    def test_clone_is_independent(self):
        g = self._build()
        g2 = g.clone()
        g2.inject_event("test", ["n0"], {}, "t0", "t1")
        assert len(g.events) == 0
        assert len(g2.events) == 1

    def test_to_device(self):
        g = self._build()
        g2 = g.to("cpu")
        assert g2 is g  # same object, moved in place

    def test_summary_contains_node_type(self):
        g = self._build()
        s = g.summary()
        assert "s" in s

    def test_repr(self):
        g = self._build()
        assert "HeterogeneousTemporalGraph" in repr(g)

    def test_timestamps_none_by_default(self):
        g = self._build()
        assert g.timestamps is None


# ---------------------------------------------------------------------------
# Temporal encoding
# ---------------------------------------------------------------------------

class TestTemporalEncoding:
    def test_sinusoidal_encoding_shape(self):
        positions = torch.arange(10, dtype=torch.float32)
        enc = sinusoidal_encoding(positions, dim=16)
        assert enc.shape == (10, 16)

    def test_sinusoidal_encoding_finite(self):
        positions = torch.arange(5, dtype=torch.float32)
        enc = sinusoidal_encoding(positions)
        assert torch.isfinite(enc).all()

    def test_build_temporal_tensor_integer_timestamps(self):
        ts = [0, 1, 2, 3, 4]
        enc = build_temporal_tensor(ts, dim=16)
        assert enc.shape == (5, 16)

    def test_build_temporal_tensor_none(self):
        enc = build_temporal_tensor(None)
        assert enc.shape == (1, TEMPORAL_FEATURE_DIM)

    def test_build_temporal_tensor_empty(self):
        enc = build_temporal_tensor([])
        assert enc.shape == (1, TEMPORAL_FEATURE_DIM)

    def test_time_delta_encoding_shape(self):
        enc = time_delta_encoding([0, 1, 3, 6], dim=16)
        assert enc.shape == (4, 16)

    def test_time_delta_encoding_none(self):
        enc = time_delta_encoding(None, dim=16)
        assert enc.shape == (1, 16)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class TestValidators:
    def test_check_no_nan_raises(self):
        t = torch.tensor([1.0, float("nan")])
        with pytest.raises(GraphOracleError):
            check_no_nan(t)

    def test_check_no_nan_ok(self):
        check_no_nan(torch.tensor([1.0, 2.0]))  # no raise

    def test_check_no_inf_raises(self):
        t = torch.tensor([float("inf")])
        with pytest.raises(GraphOracleError):
            check_no_inf(t)

    def test_validate_edge_index_ok(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        validate_edge_index(ei, num_nodes=3)  # no raise

    def test_validate_edge_index_wrong_shape(self):
        ei = torch.tensor([0, 1, 2])
        with pytest.raises(GraphOracleError):
            validate_edge_index(ei, num_nodes=3)

    def test_validate_edge_index_out_of_range(self):
        ei = torch.tensor([[0], [5]], dtype=torch.long)
        with pytest.raises(GraphOracleError):
            validate_edge_index(ei, num_nodes=3)

    def test_validate_feature_tensor_ok(self):
        t = torch.zeros(3, 4)
        validate_feature_tensor(t, (3, 4))  # no raise

    def test_validate_feature_tensor_mismatch(self):
        t = torch.zeros(3, 4)
        with pytest.raises(GraphOracleError):
            validate_feature_tensor(t, (3, 5))

    def test_validate_graph_has_targets_raises_when_no_targets(self):
        schema = GraphSchema(
            node_types=[NodeType("x")],
            edge_types=[],
        )
        feat = np.zeros((2, 1, 1), dtype=np.float32)
        graph = GraphBuilder(schema).add_nodes("x", ["a", "b"], feat).build()
        with pytest.raises(MissingTargetFeatureError):
            validate_graph_has_targets(graph)

    def test_validate_graph_has_targets_ok(self):
        schema = GraphSchema(
            node_types=[NodeType("x", features=["v"], targets=["v"])],
            edge_types=[],
        )
        feat = np.zeros((2, 1, 1), dtype=np.float32)
        graph = GraphBuilder(schema).add_nodes("x", ["a", "b"], feat).build()
        validate_graph_has_targets(graph)  # no raise
