"""Graph schema: NodeType, EdgeType, GraphSchema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graphoracle.utils.exceptions import GraphSchemaError


@dataclass
class NodeType:
    """Describes one node type in the heterogeneous graph."""

    name: str
    features: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    spatial_features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return len(self.features)

    @property
    def target_dim(self) -> int:
        return len(self.targets)

    def __repr__(self) -> str:
        return f"NodeType(name={self.name!r}, features={self.features}, targets={self.targets})"


@dataclass
class EdgeType:
    """Describes one edge type in the heterogeneous graph."""

    name: str
    src_type: str
    dst_type: str
    features: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"EdgeType(name={self.name!r}, {self.src_type} → {self.dst_type})"


@dataclass
class GraphSchema:
    """
    Schema for a heterogeneous temporal graph.

    Validates that all edge src_type/dst_type references exist in node_types
    at construction time.
    """

    node_types: list[NodeType]
    edge_types: list[EdgeType]

    def __post_init__(self) -> None:
        node_names = {nt.name for nt in self.node_types}
        for et in self.edge_types:
            if et.src_type not in node_names:
                raise GraphSchemaError(
                    f"Edge '{et.name}' src_type '{et.src_type}' not in node_types {node_names}"
                )
            if et.dst_type not in node_names:
                raise GraphSchemaError(
                    f"Edge '{et.name}' dst_type '{et.dst_type}' not in node_types {node_names}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_type_names(self) -> list[str]:
        return [nt.name for nt in self.node_types]

    @property
    def forecast_node_types(self) -> list[NodeType]:
        """Node types that have at least one target feature."""
        return [nt for nt in self.node_types if nt.targets]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_node_type(self, name: str) -> NodeType:
        for nt in self.node_types:
            if nt.name == name:
                return nt
        raise GraphSchemaError(f"Node type '{name}' not found in schema.")

    def get_edge_type(self, name: str) -> EdgeType:
        for et in self.edge_types:
            if et.name == name:
                return et
        raise GraphSchemaError(f"Edge type '{name}' not found in schema.")

    def node_dim(self, name: str) -> int:
        return self.get_node_type(name).feature_dim

    def target_dim(self, name: str) -> int:
        return self.get_node_type(name).target_dim

    def edge_triplets(self) -> list[tuple[str, str, str]]:
        """Return list of (src_type, edge_name, dst_type) tuples."""
        return [(et.src_type, et.name, et.dst_type) for et in self.edge_types]

    def __repr__(self) -> str:
        return (
            f"GraphSchema(node_types={self.node_type_names}, "
            f"edge_types={[et.name for et in self.edge_types]})"
        )
