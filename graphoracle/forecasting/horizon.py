"""ForecastResult and NodeForecast containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class NodeForecast:
    """Per-node, per-horizon forecast with optional quantile bands."""

    node_id: str
    node_type: str
    horizon: int
    point: Tensor        # (target_dim,) median / point forecast
    lower: Tensor | None = None  # (target_dim,) 10th-percentile
    upper: Tensor | None = None  # (target_dim,) 90th-percentile

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "horizon": self.horizon,
            "point": self.point.tolist(),
        }
        if self.lower is not None:
            d["lower"] = self.lower.tolist()
        if self.upper is not None:
            d["upper"] = self.upper.tolist()
        return d


class ForecastResult:
    """
    Result of ``oracle.predict(graph)``.

    Stores per-node-type, per-horizon raw tensors and exposes
    structured query methods.
    """

    def __init__(
        self,
        raw_predictions: dict[str, dict[int, Tensor]],
        graph: Any,
        reference_time: Any = None,
        horizon_unit_hours: int = 1,
    ) -> None:
        self._preds = raw_predictions
        self._graph = graph
        self.reference_time = reference_time
        self.horizon_unit_hours = horizon_unit_hours

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, node_id: str, feature: str | None = None) -> list[NodeForecast]:
        """Return NodeForecast objects for *node_id* across all horizons."""
        results: list[NodeForecast] = []
        schema = self._graph.schema
        for nt in schema.forecast_node_types:
            ids = self._graph.all_node_ids(nt.name)
            if node_id not in ids:
                continue
            node_idx = ids.index(node_id)
            h_preds = self._preds.get(nt.name, {})
            for h, pred_tensor in sorted(h_preds.items()):
                row = pred_tensor[node_idx].detach().cpu()
                target_dim = len(nt.targets) or 1
                if row.shape[-1] > target_dim:
                    n_q = row.shape[-1] // target_dim
                    row_r = row.reshape(target_dim, n_q)
                    point = row_r[:, n_q // 2]
                    lower = row_r[:, 0]
                    upper = row_r[:, -1]
                else:
                    point, lower, upper = row, None, None
                results.append(
                    NodeForecast(node_id, nt.name, h, point, lower, upper)
                )
        return results

    def all_nodes(self) -> list[str]:
        """Return all node IDs that have forecast outputs."""
        ids: list[str] = []
        for nt in self._graph.schema.forecast_node_types:
            ids.extend(self._graph.all_node_ids(nt.name))
        return ids

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        rows: list[dict[str, Any]] = []
        schema = self._graph.schema
        for nt in schema.forecast_node_types:
            ids = self._graph.all_node_ids(nt.name)
            h_preds = self._preds.get(nt.name, {})
            for h, pred_tensor in sorted(h_preds.items()):
                target_dim = len(nt.targets) or 1
                pred_np = pred_tensor.detach().cpu()
                if pred_np.shape[-1] > target_dim:
                    n_q = pred_np.shape[-1] // target_dim
                    point = pred_np.reshape(-1, target_dim, n_q)[:, :, n_q // 2]
                else:
                    point = pred_np
                for i, nid in enumerate(ids):
                    for j, feat_name in enumerate(nt.targets):
                        rows.append(
                            {
                                "node_id": nid,
                                "node_type": nt.name,
                                "horizon": h,
                                "feature": feat_name,
                                "value": float(point[i, j]) if point.ndim > 1 else float(point[i]),
                            }
                        )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = ["ForecastResult"]
        for nt_name, h_preds in self._preds.items():
            for h, pred in sorted(h_preds.items()):
                lines.append(f"  {nt_name} | h={h}: shape={tuple(pred.shape)}")
        return "\n".join(lines)

    def plot_gantt(self, node: str, feature: str) -> None:
        forecasts = self.get(node, feature)
        try:
            import matplotlib.pyplot as plt

            horizons = [f.horizon for f in forecasts]
            values = [float(f.point[0]) if f.point.ndim > 0 else float(f.point) for f in forecasts]
            plt.bar(horizons, values, label=node)
            plt.xlabel("Horizon")
            plt.ylabel(feature)
            plt.title(f"Forecast — {node} / {feature}")
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    def __repr__(self) -> str:
        return self.summary()
