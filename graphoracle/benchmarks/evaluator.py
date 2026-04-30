"""Benchmark evaluation: MAE, RMSE, MAPE, CRPS scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _to_np(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.abs(pred - actual).mean())


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(((pred - actual) ** 2).mean()))


def mape(pred: np.ndarray, actual: np.ndarray, eps: float = 1e-8) -> float:
    return float((np.abs(pred - actual) / (np.abs(actual) + eps)).mean()) * 100


def crps(
    pred_samples: np.ndarray, actual: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score (Energy form).
    pred_samples : (N, S) — S ensemble samples per observation
    actual       : (N,)
    """
    N, S = pred_samples.shape
    mae_term = np.abs(pred_samples - actual[:, None]).mean()
    spread = 0.0
    for i in range(S):
        for j in range(i + 1, S):
            spread += np.abs(pred_samples[:, i] - pred_samples[:, j]).mean()
    spread /= S * (S - 1) / 2 if S > 1 else 1.0
    return float(mae_term - 0.5 * spread)


@dataclass
class EvalResult:
    """Per-model, per-node-type, per-horizon evaluation result."""

    model_name: str
    metrics: dict[str, dict[str, dict[int, float]]] = field(default_factory=dict)
    # metrics[metric_name][node_type][horizon] = value

    def summary_table(self) -> str:
        lines = [f"{'Metric':10s} {'NodeType':20s} {'Horizon':8s} {'Value':>10s}"]
        lines.append("-" * 52)
        for m, nt_dict in self.metrics.items():
            for nt, h_dict in nt_dict.items():
                for h, v in sorted(h_dict.items()):
                    lines.append(f"{m:10s} {nt:20s} {h:8d} {v:10.4f}")
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required") from exc
        rows = []
        for m, nt_dict in self.metrics.items():
            for nt, h_dict in nt_dict.items():
                for h, v in h_dict.items():
                    rows.append(
                        {"metric": m, "node_type": nt, "horizon": h, "value": v}
                    )
        return pd.DataFrame(rows)

    def plot_error_distribution(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        df = self.to_dataframe()
        mae_df = df[df["metric"] == "MAE"]
        if mae_df.empty:
            return
        fig, ax = plt.subplots()
        for nt, grp in mae_df.groupby("node_type"):
            ax.plot(grp["horizon"], grp["value"], marker="o", label=nt)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("MAE")
        ax.set_title("Error by horizon")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return f"EvalResult(model={self.model_name})\n{self.summary_table()}"


class Evaluator:
    """
    Evaluate a trained model on a test graph.

    Usage
    -----
    evaluator = Evaluator(oracle, metrics=["MAE", "RMSE", "MAPE"])
    result = evaluator.run(test_graph)
    result.to_dataframe()
    """

    SUPPORTED = {"MAE", "RMSE", "MAPE", "CRPS"}

    def __init__(
        self,
        oracle: Any,
        metrics: list[str] | None = None,
        per_node_type: bool = True,
        per_horizon: bool = True,
    ) -> None:
        self.oracle = oracle
        self.metrics = metrics or ["MAE", "RMSE", "MAPE"]
        self.per_node_type = per_node_type
        self.per_horizon = per_horizon

        unknown = set(self.metrics) - self.SUPPORTED
        if unknown:
            raise ValueError(f"Unsupported metrics: {unknown}. Choose from {self.SUPPORTED}")

    @torch.no_grad()
    def run(
        self,
        graph: Any,
        model_name: str = "graphoracle",
    ) -> EvalResult:
        from graphoracle.forecasting.engine import ForecastEngine

        engine = ForecastEngine(self.oracle.model, device=self.oracle.device)
        raw_preds = engine._run_inference(graph)

        result = EvalResult(model_name=model_name)

        # Build ground-truth targets
        schema = graph.schema
        T = graph.num_timesteps

        for nt_name, h_preds in raw_preds.items():
            nt = schema.get_node_type(nt_name)
            if not nt.targets:
                continue
            feat = graph.get_node_features(nt_name)
            tgt_indices = [
                nt.features.index(t) for t in nt.targets if t in nt.features
            ]
            if not tgt_indices:
                continue

            for h, pred_tensor in h_preds.items():
                t_idx = min(T - 1, T - 1 + h)
                actual_t = feat[:, t_idx, :][:, tgt_indices].cpu().numpy()
                pred_np = _to_np(pred_tensor)

                # Use median quantile if quantile output
                if pred_np.shape[-1] > len(tgt_indices):
                    n_q = pred_np.shape[-1] // len(tgt_indices)
                    pred_np = pred_np.reshape(-1, len(tgt_indices), n_q)[:, :, n_q // 2]

                pred_flat = pred_np.flatten()
                actual_flat = actual_t.flatten()

                for metric in self.metrics:
                    if metric not in result.metrics:
                        result.metrics[metric] = {}
                    if nt_name not in result.metrics[metric]:
                        result.metrics[metric][nt_name] = {}
                    if metric == "MAE":
                        val = mae(pred_flat, actual_flat)
                    elif metric == "RMSE":
                        val = rmse(pred_flat, actual_flat)
                    elif metric == "MAPE":
                        val = mape(pred_flat, actual_flat)
                    elif metric == "CRPS":
                        samples = pred_np.reshape(pred_np.shape[0], -1)
                        val = crps(samples, actual_flat[: samples.shape[0]])
                    else:
                        val = 0.0
                    result.metrics[metric][nt_name][h] = val

        return result
