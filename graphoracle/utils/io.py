"""Model and graph serialisation helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from graphoracle.utils.exceptions import GraphOracleError


def save_model(model: nn.Module, path: str | Path) -> None:
    """Save model state dict to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str | Path) -> None:
    """Load model weights from *path* into *model* in-place."""
    path = Path(path)
    if not path.exists():
        raise GraphOracleError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)


def save_graph(graph: Any, path: str | Path) -> None:
    """Pickle a graph object to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str | Path) -> Any:
    """Load and return a pickled graph from *path*."""
    path = Path(path)
    if not path.exists():
        raise GraphOracleError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
