"""Anomaly detection on forecast residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class AnomalyResult:
    """Detected anomalies per node type and horizon."""

    anomalies: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    # anomalies[node_type][horizon] = list of anomalous node IDs
    scores: dict[str, dict[int, Tensor]] = field(default_factory=dict)
    # scores[node_type][horizon] = (N,) anomaly score per node

    def summary(self) -> str:
        lines = ["AnomalyResult"]
        for nt, h_dict in self.anomalies.items():
            for h, nodes in sorted(h_dict.items()):
                lines.append(f"  {nt} | h={h}: {len(nodes)} anomalous nodes")
        return "\n".join(lines)


class AnomalyDetector:
    """
    Detect anomalies by comparing forecast residuals to a learned threshold.

    Parameters
    ----------
    oracle    : fitted GraphOracle
    threshold : residual magnitude above which a node is flagged
    """

    def __init__(self, oracle: Any, threshold: float = 2.0) -> None:
        self.oracle = oracle
        self.threshold = threshold

    @torch.no_grad()
    def detect(self, graph: Any) -> AnomalyResult:
        """Flag nodes whose forecast residuals exceed the threshold."""
        from graphoracle.forecasting.engine import ForecastEngine

        engine = ForecastEngine(self.oracle.model, device=self.oracle.device)
        raw_preds = engine._run_inference(graph)

        result = AnomalyResult()
        schema = graph.schema

        for nt in schema.forecast_node_types:
            ids = graph.all_node_ids(nt.name)
            feat = graph.get_node_features(nt.name).cpu()
            tgt_idx = [nt.features.index(t) for t in nt.targets if t in nt.features]
            if not tgt_idx:
                continue

            h_preds = raw_preds.get(nt.name, {})
            result.anomalies[nt.name] = {}
            result.scores[nt.name] = {}

            for h, pred_tensor in sorted(h_preds.items()):
                T = feat.shape[1]
                t = min(T - 1, h)
                actual = feat[:, t, :][:, tgt_idx]
                pred = pred_tensor.cpu()
                if pred.shape[-1] > actual.shape[-1]:
                    n_q = pred.shape[-1] // actual.shape[-1]
                    pred = pred.reshape(pred.shape[0], -1, n_q)[:, :, n_q // 2]

                residuals = (pred - actual).abs().mean(-1)  # (N,)
                result.scores[nt.name][h] = residuals
                flagged = [ids[i] for i in range(len(ids)) if float(residuals[i]) > self.threshold]
                result.anomalies[nt.name][h] = flagged

        return result
