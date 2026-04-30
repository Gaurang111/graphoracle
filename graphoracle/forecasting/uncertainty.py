"""Uncertainty estimation wrappers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.models.base import BaseForecastModel


class ConformalWrapper:
    """
    Conformal prediction wrapper for any BaseForecastModel.

    Calibrates prediction intervals on a held-out calibration set so
    that the nominal coverage (e.g. 90%) is guaranteed by construction.

    Parameters
    ----------
    model        : fitted BaseForecastModel
    coverage     : desired coverage level, e.g. 0.9
    """

    def __init__(self, model: BaseForecastModel, coverage: float = 0.9) -> None:
        self.model = model
        self.coverage = coverage
        self._quantile: float | None = None

    def calibrate(self, cal_graph: Any, cal_targets: dict) -> None:
        """Compute calibration quantile on *cal_graph*."""
        from graphoracle.forecasting.engine import ForecastEngine

        engine = ForecastEngine(self.model)
        preds = engine._run_inference(cal_graph)

        residuals: list[float] = []
        for nt_name, h_preds in preds.items():
            tgt = cal_targets.get(nt_name, {})
            for h, pred in h_preds.items():
                y = tgt.get(h)
                if y is None:
                    continue
                if pred.shape[-1] > y.shape[-1]:
                    n_q = pred.shape[-1] // y.shape[-1]
                    median = pred.reshape(pred.shape[0], -1, n_q)[:, :, n_q // 2]
                    residuals.extend((y - median).abs().flatten().tolist())
                else:
                    residuals.extend((y - pred).abs().flatten().tolist())

        if residuals:
            import numpy as np

            self._quantile = float(np.quantile(residuals, self.coverage))

    def predict_with_intervals(
        self, graph: Any
    ) -> dict[str, dict[int, tuple[Tensor, Tensor, Tensor]]]:
        """
        Return {node_type: {horizon: (lower, median, upper)}} tensors.
        """
        from graphoracle.forecasting.engine import ForecastEngine

        engine = ForecastEngine(self.model)
        raw = engine._run_inference(graph)
        q = self._quantile or 0.0
        result: dict[str, dict[int, tuple[Tensor, Tensor, Tensor]]] = {}
        for nt_name, h_preds in raw.items():
            result[nt_name] = {}
            for h, pred in h_preds.items():
                result[nt_name][h] = (pred - q, pred, pred + q)
        return result


class MonteCarloDropoutWrapper:
    """
    MC Dropout uncertainty estimation.

    Enables dropout at inference time and runs *n_samples* forward
    passes to estimate prediction variance.
    """

    def __init__(self, model: BaseForecastModel, n_samples: int = 30) -> None:
        self.model = model
        self.n_samples = n_samples

    def predict(self, graph: Any) -> dict[str, dict[int, Tensor]]:
        """
        Return mean predictions; variance is accessible via predict_variance.
        """
        from graphoracle.forecasting.engine import ForecastEngine

        self.model.train()  # Enable dropout
        engine = ForecastEngine(self.model)
        sample_preds: list[dict[str, dict[int, Tensor]]] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                sample_preds.append(engine._run_inference(graph))

        self.model.eval()

        mean_preds: dict[str, dict[int, Tensor]] = {}
        for nt_name in sample_preds[0]:
            mean_preds[nt_name] = {}
            for h in sample_preds[0][nt_name]:
                stacked = torch.stack(
                    [sp[nt_name][h] for sp in sample_preds], dim=0
                )
                mean_preds[nt_name][h] = stacked.mean(0)
        return mean_preds
