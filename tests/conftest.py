"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


@pytest.fixture
def simple_schema() -> GraphSchema:
    return GraphSchema(
        node_types=[
            NodeType(
                "sensor",
                features=["value", "temp"],
                targets=["value"],
            )
        ],
        edge_types=[EdgeType("link", "sensor", "sensor")],
    )


@pytest.fixture
def simple_graph(simple_schema):
    rng = np.random.default_rng(42)
    features = rng.random((5, 10, 2)).astype(np.float32)
    builder = GraphBuilder(simple_schema)
    builder.add_nodes("sensor", [f"s{i}" for i in range(5)], features)
    builder.add_edges("link", ["s0", "s1", "s2"], ["s1", "s2", "s0"])
    return builder.build()
