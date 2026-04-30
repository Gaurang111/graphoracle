"""Custom exception hierarchy for GraphOracle."""

from __future__ import annotations


class GraphOracleError(Exception):
    """Base exception for all GraphOracle errors."""


class GraphSchemaError(GraphOracleError):
    """Raised when a GraphSchema is invalid or inconsistent."""


class ModelNotRegisteredError(GraphOracleError):
    """Raised when a model name is not found in the ModelRegistry."""


class IncompatibleNodeTypeError(GraphOracleError):
    """Raised when a node type is not recognised by the current schema."""


class InsufficientHistoryError(GraphOracleError):
    """Raised when the graph has fewer timesteps than the model requires."""


class ForecastHorizonError(GraphOracleError):
    """Raised for invalid forecast horizon configurations."""


class MissingTargetFeatureError(GraphOracleError):
    """Raised when a target feature is not present in the node's feature list."""


class ConnectorError(GraphOracleError):
    """Raised by data connectors on load failure."""


class DomainError(GraphOracleError):
    """Raised for domain configuration or schema definition errors."""


class EdgeDiscoveryError(GraphOracleError):
    """Raised when automatic edge discovery fails."""


class CheckpointError(GraphOracleError):
    """Raised on checkpoint save or load failures."""
