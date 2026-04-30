"""Graph validation utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from graphoracle.utils.exceptions import GraphOracleError, MissingTargetFeatureError


def check_no_nan(tensor: Tensor, name: str = "tensor") -> None:
    if torch.isnan(tensor).any():
        raise GraphOracleError(f"NaN values found in {name}.")


def check_no_inf(tensor: Tensor, name: str = "tensor") -> None:
    if torch.isinf(tensor).any():
        raise GraphOracleError(f"Inf values found in {name}.")


def validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GraphOracleError(
            f"edge_index must be (2, E), got shape {tuple(edge_index.shape)}."
        )
    if edge_index.numel() > 0:
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise GraphOracleError(
                f"edge_index contains out-of-range indices for num_nodes={num_nodes}."
            )


def validate_feature_tensor(tensor: Tensor, expected_shape: tuple) -> None:
    if tensor.shape != torch.Size(expected_shape):
        raise GraphOracleError(
            f"Expected tensor shape {expected_shape}, got {tuple(tensor.shape)}."
        )


def validate_graph_has_targets(graph: object) -> None:
    schema = getattr(graph, "schema", None)
    if schema is None:
        raise GraphOracleError("Graph has no schema.")
    if not schema.forecast_node_types:
        raise MissingTargetFeatureError(
            "Schema has no node types with target features. "
            "Add at least one NodeType with non-empty 'targets'."
        )
