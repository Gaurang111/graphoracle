"""Abstract base for data connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema


class DataConnector(ABC):
    """
    Abstract base class for data connectors.

    Connectors adapt external data sources (CSV, JSON, DataFrames, APIs)
    into HeterogeneousTemporalGraph objects matching a given schema.
    """

    def __init__(self, schema: GraphSchema) -> None:
        self.schema = schema

    @abstractmethod
    def load(self, *args: Any, **kwargs: Any) -> HeterogeneousTemporalGraph:
        """Load data and return a graph."""

    def validate_schema(self, graph: HeterogeneousTemporalGraph) -> bool:
        """
        Basic schema conformance check: every registered node type has data.
        Returns True if valid.
        """
        for nt in self.schema.forecast_node_types:
            if graph.num_nodes(nt.name) == 0:
                return False
        return True
