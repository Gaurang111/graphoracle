"""
graphoracle — Universal Interdependent Knowledge Graph Forecasting
==================================================================

Quick start
-----------
>>> from graphoracle import GraphOracle
>>> from graphoracle.domains import ElectricGridDomain
>>> domain = ElectricGridDomain()
>>> graph = domain.build_synthetic_graph(num_timesteps=48)
>>> oracle = GraphOracle(model="graphoracle", horizons=[1, 6, 24])
>>> oracle.fit(graph, epochs=5)
>>> forecast = oracle.predict(graph)
>>> print(forecast.summary())
"""

from __future__ import annotations

from typing import Any

from graphoracle._version import __version__
from graphoracle.oracle import GraphOracle
from graphoracle.graph import (
    EdgeType,
    GraphBuilder,
    GraphSchema,
    HeterogeneousTemporalGraph,
    NodeType,
    TemporalEvent,
)
from graphoracle.models import (
    BaseForecastModel,
    ModelRegistry,
)
from graphoracle.training import TrainingConfig
from graphoracle.utils import configure_logging

__all__ = [
    "__version__",
    "GraphOracle",
    "HeterogeneousTemporalGraph",
    "TemporalEvent",
    "NodeType",
    "EdgeType",
    "GraphSchema",
    "GraphBuilder",
    "BaseForecastModel",
    "ModelRegistry",
    "TrainingConfig",
    "configure_logging",
]
