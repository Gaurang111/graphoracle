"""Cascade shock simulation through the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.temporal import TEMPORAL_FEATURE_DIM, build_temporal_tensor
from graphoracle.models.base import BaseForecastModel


@dataclass
class Shock:
    """A perturbation applied to a single node feature."""

    node: str
    feature: str
    change: float
    type: str = "absolute"  # "absolute" | "percent"


@dataclass
class CascadeResult:
    """Result of a cascade simulation."""

    steps: int
    impact_by_step: list[dict[str, dict[str, float]]] = field(default_factory=list)
    # impact_by_step[t][node_type][node_id] = impact_magnitude

    def highest_risk_nodes(self, k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k nodes by total impact across all steps."""
        totals: dict[str, float] = {}
        for step_impacts in self.impact_by_step:
            for nt_impacts in step_impacts.values():
                for node_id, impact in nt_impacts.items():
                    totals[node_id] = totals.get(node_id, 0.0) + abs(impact)
        sorted_nodes = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:k]

    def estimated_recovery_hours(self, threshold: float = 0.1) -> float:
        """Estimate recovery time as first step where all impacts drop below threshold."""
        for t, step_impacts in enumerate(self.impact_by_step):
            all_low = all(
                abs(impact) < threshold
                for nt_impacts in step_impacts.values()
                for impact in nt_impacts.values()
            )
            if all_low:
                return float(t)
        return float(len(self.impact_by_step))

    def impact_delta(self, step: int) -> dict[str, dict[str, float]]:
        if step < len(self.impact_by_step):
            return self.impact_by_step[step]
        return {}

    def plot_impact_over_time(self) -> None:
        try:
            import matplotlib.pyplot as plt

            totals = [
                sum(abs(v) for nt_d in step.values() for v in nt_d.values())
                for step in self.impact_by_step
            ]
            plt.plot(range(len(totals)), totals, marker="o")
            plt.xlabel("Step")
            plt.ylabel("Total impact magnitude")
            plt.title("Cascade impact over time")
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass


class CascadeSimulator:
    """
    Simulate shock propagation through learned GNN embeddings.

    The approach:
    1. Apply shocks to the base graph node features.
    2. Run the model on the shocked graph to get a perturbed forecast.
    3. Compare with the unshocked baseline forecast.
    4. Repeat for *steps* propagation steps.

    Note: this is an approximation via GNN embeddings, not a mechanistic model.
    """

    def __init__(
        self,
        model: BaseForecastModel,
        steps: int = 24,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.steps = steps
        self.device = device

    @torch.no_grad()
    def simulate(
        self,
        graph: HeterogeneousTemporalGraph,
        shocks: list[dict[str, Any]],
    ) -> CascadeResult:
        parsed_shocks = [
            Shock(
                node=s["node"],
                feature=s.get("feature", ""),
                change=float(s.get("change", 0.0)),
                type=str(s.get("type", "absolute")),
            )
            for s in shocks
        ]

        device = torch.device(self.device)
        model = self.model
        model.eval()
        model.to(device)

        # Build base node features
        node_features_base: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = graph.get_node_features(nt.name).to(device)
            node_features_base[nt.name] = feat

        edge_index: dict[str, Tensor] = {
            et.name: graph.get_edge_index(et.name).to(device)
            for et in graph.schema.edge_types
        }

        ts = graph.timestamps
        if ts:
            temporal_enc = build_temporal_tensor(ts, TEMPORAL_FEATURE_DIM).to(device)
        else:
            T = max((v.shape[1] for v in node_features_base.values()), default=1)
            temporal_enc = torch.zeros(T, TEMPORAL_FEATURE_DIM, device=device)

        # Baseline predictions
        base_preds = model(graph, node_features_base, edge_index, temporal_enc)

        impact_by_step: list[dict[str, dict[str, float]]] = []

        # Apply shocks and compute impact at each propagation step
        node_features_shocked = {k: v.clone() for k, v in node_features_base.items()}

        for shock in parsed_shocks:
            for nt in graph.schema.node_types:
                ids = graph.all_node_ids(nt.name)
                if shock.node in ids:
                    idx = ids.index(shock.node)
                    feat_idx = nt.features.index(shock.feature) if shock.feature in nt.features else 0
                    feat = node_features_shocked[nt.name]
                    if shock.type == "percent":
                        feat[idx, :, feat_idx] *= (1 + shock.change)
                    else:
                        feat[idx, :, feat_idx] += shock.change

        for _ in range(self.steps):
            shocked_preds = model(graph, node_features_shocked, edge_index, temporal_enc)

            step_impact: dict[str, dict[str, float]] = {}
            for nt in graph.schema.forecast_node_types:
                ids = graph.all_node_ids(nt.name)
                base = base_preds.get(nt.name, {})
                shock_p = shocked_preds.get(nt.name, {})
                nt_impact: dict[str, float] = {}
                for nid in ids:
                    node_idx = ids.index(nid)
                    total_delta = 0.0
                    for h in model.horizons:
                        b = base.get(h)
                        s = shock_p.get(h)
                        if b is not None and s is not None:
                            total_delta += float((s[node_idx] - b[node_idx]).abs().mean())
                    nt_impact[nid] = total_delta
                step_impact[nt.name] = nt_impact

            impact_by_step.append(step_impact)

        return CascadeResult(steps=self.steps, impact_by_step=impact_by_step)
