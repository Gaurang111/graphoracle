"""Abstract base class for domain definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


class BaseDomain(ABC):
    """
    Abstract base for a domain.

    Subclass this to define a new domain in ~30 lines:

    1. Implement ``schema`` property → return a ``GraphSchema``
    2. Optionally override ``build_graph_from_csv(...)``
    3. Optionally override ``default_horizons``
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def schema(self) -> GraphSchema:
        """Return the GraphSchema for this domain."""

    # ------------------------------------------------------------------
    # Optional defaults
    # ------------------------------------------------------------------

    @property
    def default_horizons(self) -> list[int]:
        return [1, 6, 24]

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def build_graph_from_csv(
        self,
        nodes: str | Path | dict[str, str | Path],
        edges: str | Path | dict[str, str | Path],
        timeseries: str | Path | dict[str, str | Path] | None = None,
        **kwargs: Any,
    ) -> HeterogeneousTemporalGraph:
        """
        Load a graph from CSV files.

        Subclasses may override for domain-specific parsing logic.
        The default implementation uses a generic CSV loader.

        Parameters
        ----------
        nodes      : path to a node CSV (columns: node_id, node_type, feature1, ...)
                     OR dict {node_type: path}
        edges      : path to an edge CSV (columns: src_id, dst_id, edge_type, ...)
                     OR dict {edge_type: path}
        timeseries : path to a long-format time series CSV
                     (columns: node_id, timestamp, feature1, ...)
        """
        from graphoracle.connectors.csv_connector import CSVConnector
        connector = CSVConnector(self.schema)
        return connector.load(nodes, edges, timeseries, **kwargs)

    def build_graph_from_dataframes(
        self,
        node_dfs: dict[str, Any],
        edge_dfs: dict[str, Any],
        feature_dfs: dict[str, Any] | None = None,
        timestamps: list[Any] | None = None,
    ) -> HeterogeneousTemporalGraph:
        """Build a graph from pandas DataFrames."""
        return GraphBuilder.from_dataframes(
            self.schema, node_dfs, edge_dfs, feature_dfs, timestamps
        )

    def build_synthetic_graph(
        self,
        nodes_per_type: dict[str, int] | None = None,
        num_timesteps: int = 48,
        seed: int = 42,
    ) -> HeterogeneousTemporalGraph:
        """Build a synthetic graph for prototyping and testing."""
        from graphoracle.connectors.synthetic import SyntheticGenerator
        gen = SyntheticGenerator(self.schema, seed=seed)
        return gen.generate(nodes_per_type, num_timesteps)

    def __repr__(self) -> str:
        return f"{self.name}(schema={self.schema!r})"
