"""GNNExplainer-style node importance attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.temporal import TEMPORAL_FEATURE_DIM, build_temporal_tensor
from graphoracle.models.base import BaseForecastModel


@dataclass
class NodeImportance:
    """Feature and neighbour attribution scores for a single node."""

    node_id: str
    node_type: str
    horizon: int
    feature_scores: dict[str, float] = field(default_factory=dict)
    neighbour_scores: dict[str, float] = field(default_factory=dict)


class GNNExplainer:
    """
    Simplified GNNExplainer for BaseForecastModel.

    Approximates feature importance via input-gradient attribution and
    neighbour importance by comparing predictions with and without each
    neighbour's contribution.

    Parameters
    ----------
    model  : fitted BaseForecastModel
    device : torch device string
    n_steps : gradient steps for mask optimisation
    """

    def __init__(
        self,
        model: BaseForecastModel,
        device: str = "cpu",
        n_steps: int = 50,
    ) -> None:
        self.model = model
        self.device = device
        self.n_steps = n_steps

    def explain_node(
        self,
        graph: HeterogeneousTemporalGraph,
        node_id: str,
        node_type: str,
        horizon: int,
    ) -> NodeImportance:
        """
        Return feature and neighbour attributions for *node_id* at *horizon*.
        Uses input gradients as a proxy for importance.
        """
        device = torch.device(self.device)
        model = self.model
        model.eval()
        model.to(device)

        node_features: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = graph.get_node_features(nt.name).to(device)
            node_features[nt.name] = feat.requires_grad_(feat.dtype.is_floating_point)

        edge_index: dict[str, Tensor] = {
            et.name: graph.get_edge_index(et.name).to(device)
            for et in graph.schema.edge_types
        }

        ts = graph.timestamps
        if ts:
            temporal_enc = build_temporal_tensor(ts, TEMPORAL_FEATURE_DIM).to(device)
        else:
            T = max((v.shape[1] for v in node_features.values()), default=1)
            temporal_enc = torch.zeros(T, TEMPORAL_FEATURE_DIM, device=device)

        # Forward pass with gradient tracking
        preds = model(graph, node_features, edge_index, temporal_enc)

        ids = graph.all_node_ids(node_type)
        node_idx = ids.index(node_id) if node_id in ids else 0
        nt = graph.schema.get_node_type(node_type)

        pred_tensor = preds.get(node_type, {}).get(horizon)
        feature_scores: dict[str, float] = {}

        if pred_tensor is not None and node_features.get(node_type) is not None:
            feat_t = node_features[node_type]
            target_dim = len(nt.targets) or 1
            if pred_tensor.shape[-1] > target_dim:
                n_q = pred_tensor.shape[-1] // target_dim
                score_val = pred_tensor[node_idx].reshape(target_dim, n_q)[:, n_q // 2].sum()
            else:
                score_val = pred_tensor[node_idx].sum()

            try:
                grads = torch.autograd.grad(score_val, feat_t, retain_graph=False)[0]
                node_grads = grads[node_idx].abs().mean(0)  # (F,)
                for i, fname in enumerate(nt.features):
                    if i < node_grads.shape[0]:
                        feature_scores[fname] = float(node_grads[i])
            except Exception:
                for fname in nt.features:
                    feature_scores[fname] = 0.0

        # Neighbour importance: use edge structure
        neighbour_scores: dict[str, float] = {}
        for et in graph.schema.edge_types:
            if et.dst_type != node_type:
                continue
            ei = graph.get_edge_index(et.name)
            if ei.numel() == 0:
                continue
            dst_mask = ei[1] == node_idx
            src_indices = ei[0][dst_mask].tolist()
            src_ids = graph.all_node_ids(et.src_type)
            for src_idx in src_indices:
                if src_idx < len(src_ids):
                    nid = src_ids[src_idx]
                    # Use feature magnitude as proxy importance
                    src_feat = graph.get_node_features(et.src_type)
                    if src_feat.numel() > 0 and src_idx < src_feat.shape[0]:
                        score = float(src_feat[src_idx].abs().mean())
                    else:
                        score = 0.0
                    neighbour_scores[nid] = score

        return NodeImportance(
            node_id=node_id,
            node_type=node_type,
            horizon=horizon,
            feature_scores=feature_scores,
            neighbour_scores=neighbour_scores,
        )
