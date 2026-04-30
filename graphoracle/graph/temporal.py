"""Temporal feature encoding utilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

TEMPORAL_FEATURE_DIM: int = 16


def sinusoidal_encoding(positions: Tensor, dim: int = TEMPORAL_FEATURE_DIM) -> Tensor:
    """
    Compute sinusoidal positional encoding for arbitrary float positions.

    Parameters
    ----------
    positions : (T,) float tensor
    dim       : output feature dimension (must be even)

    Returns
    -------
    (T, dim) float tensor
    """
    T = positions.shape[0]
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
    )
    args = positions.float().unsqueeze(1) * freqs.unsqueeze(0)  # (T, half)
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (T, 2*half)
    if enc.shape[1] < dim:
        enc = torch.cat([enc, torch.zeros(T, dim - enc.shape[1])], dim=-1)
    return enc[:, :dim]


def build_temporal_tensor(
    timestamps: list[Any] | None,
    dim: int = TEMPORAL_FEATURE_DIM,
) -> Tensor:
    """
    Build a (T, dim) temporal encoding tensor from a list of timestamps.

    Timestamps may be datetime objects or numeric values.
    Falls back to position-based encoding if timestamps is None or empty.
    """
    if not timestamps:
        return torch.zeros(1, dim)

    T = len(timestamps)
    try:
        t0 = timestamps[0]
        positions = torch.tensor(
            [(t - t0).total_seconds() / 3600.0 for t in timestamps],
            dtype=torch.float32,
        )
    except (AttributeError, TypeError):
        positions = torch.arange(T, dtype=torch.float32)

    return sinusoidal_encoding(positions, dim)


def time_delta_encoding(
    timestamps: list[Any] | None,
    dim: int = TEMPORAL_FEATURE_DIM,
) -> Tensor:
    """
    Encode consecutive time deltas (gap between adjacent steps) instead of
    absolute positions.  Useful for irregular sampling.

    Returns (T, dim).
    """
    if not timestamps:
        return torch.zeros(1, dim)

    T = len(timestamps)
    deltas: list[float] = [0.0]
    for i in range(1, T):
        try:
            d = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600.0
        except (AttributeError, TypeError):
            d = 1.0
        deltas.append(d)

    positions = torch.tensor(deltas, dtype=torch.float32)
    return sinusoidal_encoding(positions, dim)
