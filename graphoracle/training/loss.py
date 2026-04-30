"""Loss functions for multi-horizon heterogeneous graph forecasting."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Elementwise losses
# ---------------------------------------------------------------------------

def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()


def rmse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(((pred - target) ** 2).mean())


def mape_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    return ((pred - target).abs() / (target.abs() + eps)).mean()


def pinball_loss(pred: Tensor, target: Tensor, quantile: float = 0.5) -> Tensor:
    errors = target - pred
    return torch.max(quantile * errors, (quantile - 1) * errors).mean()


def quantile_loss(
    pred: Tensor,
    target: Tensor,
    quantiles: list[float] | None = None,
    target_dim: int = 1,
) -> Tensor:
    """
    Pinball / quantile loss.

    Parameters
    ----------
    pred       : (N, target_dim * n_quantiles)
    target     : (N, target_dim)
    quantiles  : list of quantile levels, default [0.1, 0.5, 0.9]
    target_dim : number of forecast targets per node
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]
    n_q = len(quantiles)
    N = pred.shape[0]
    pred_r = pred.reshape(N, target_dim, n_q)          # (N, D, Q)
    target_r = target.reshape(N, target_dim, 1).expand(N, target_dim, n_q)  # (N, D, Q)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    errors = target_r - pred_r
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


def get_loss_fn(name: str):
    mapping = {
        "mae": mae_loss,
        "rmse": rmse_loss,
        "mape": mape_loss,
        "quantile": lambda p, t: quantile_loss(p, t),
        "pinball": pinball_loss,
    }
    if name not in mapping:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(mapping)}")
    return mapping[name]


# ---------------------------------------------------------------------------
# Multi-horizon aggregator
# ---------------------------------------------------------------------------

class MultiHorizonLoss:
    """
    Aggregate loss across all (node_type, horizon) pairs.

    Parameters
    ----------
    loss_name : 'mae' | 'rmse' | 'mape' | 'quantile' | 'pinball'
    """

    def __init__(self, loss_name: str = "mae") -> None:
        self.loss_name = loss_name
        self.base_fn = get_loss_fn(loss_name)

    def __call__(
        self,
        preds: dict[str, dict[int, Tensor]],
        targets: dict[str, dict[int, Tensor]],
    ) -> Tensor:
        losses: list[Tensor] = []
        for nt_name, h_preds in preds.items():
            tgt_nt = targets.get(nt_name)
            if tgt_nt is None:
                continue
            for h, pred in h_preds.items():
                tgt = tgt_nt.get(h)
                if tgt is None:
                    continue
                target_dim = tgt.shape[-1]
                if pred.shape[-1] > target_dim:
                    n_q = pred.shape[-1] // target_dim
                    q = [0.1, 0.5, 0.9][:n_q]
                    losses.append(quantile_loss(pred, tgt, q, target_dim))
                else:
                    losses.append(self.base_fn(pred, tgt))

        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()
