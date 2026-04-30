"""Full built-in GraphOracle model — the flagship architecture."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.graph.temporal import TEMPORAL_FEATURE_DIM, build_temporal_tensor
from graphoracle.models.adaptive_graph import AdaptiveGraphLearner
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.hgt import HGT
from graphoracle.models.registry import ModelRegistry
from graphoracle.models.tgn import TGNMemory


class NodeEncoder(nn.Module):
    """Type-specific encoder: linear projection + spatial + temporal fusion."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        temporal_dim: int = TEMPORAL_FEATURE_DIM,
        use_spatial: bool = False,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(in_dim, embed_dim)
        self.temp_proj = nn.Linear(temporal_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, temporal_enc: Tensor) -> Tensor:
        """
        x           : (N, T, F) or (N, F)
        temporal_enc: (T, temporal_dim) or (1, temporal_dim)
        Returns     : (N, embed_dim)
        """
        if x.ndim == 3:
            N, T, _F = x.shape  # _F avoids shadowing torch.nn.functional alias F
            # Encode each timestep and pool
            h_feat = self.feat_proj(x)                          # (N, T, E)
            t_enc = self.temp_proj(temporal_enc)                 # (T, E)
            h = h_feat + t_enc.unsqueeze(0)                     # (N, T, E)
            h = h.mean(dim=1)                                   # (N, E) temporal pool
            # Re-encode last step for recency bias
            h_last = self.feat_proj(x[:, -1, :])               # (N, E)
        else:
            h = self.feat_proj(x)
            h_last = h

        fused = self.out_proj(torch.cat([h, h_last], dim=-1))
        return self.norm(F.relu(fused))


class GlobalContextModule(nn.Module):
    """Graph-level pooling → global state → cross-attention injection."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        node_embeds: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Aggregate all node embeddings → global vector, then inject into each type.
        node_embeds: list of (Ni, D) tensors
        Returns global_vec (1, D) and list of updated (Ni, D)
        """
        all_nodes = torch.cat(node_embeds, dim=0)              # (N_total, D)
        global_vec = self.pool_proj(all_nodes.mean(0, keepdim=True))  # (1, D)

        updated = []
        for h in node_embeds:
            q = h.unsqueeze(0)                                 # (1, N, D) batch
            kv = global_vec.unsqueeze(0)                       # (1, 1, D)
            attn_out, _ = self.cross_attn(q, kv, kv)
            updated.append(self.norm(h + attn_out.squeeze(0)))

        return global_vec, updated


class ForecastHead(nn.Module):
    """Per-horizon quantile forecast head."""

    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(self, embed_dim: int, target_dim: int, horizons: list[int]) -> None:
        super().__init__()
        self.horizons = horizons
        n_q = len(self.QUANTILES)
        self.heads = nn.ModuleDict(
            {
                f"h{h}": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim // 2, target_dim * n_q),
                )
                for h in horizons
            }
        )
        self.target_dim = target_dim
        self.n_q = n_q

    def forward(self, h: Tensor) -> dict[int, Tensor]:
        """h: (N, embed_dim) → {horizon: (N, target_dim * n_quantiles)}"""
        return {
            horizon: self.heads[f"h{horizon}"](h)
            for horizon in self.horizons
        }


@ModelRegistry.register("graphoracle")
class GraphOracleModel(BaseForecastModel):
    """
    Full GraphOracle architecture.

    Layers
    ------
    1. Type-specific NodeEncoder (temporal + spatial encoding)
    2. TGN memory injection
    3. 4-layer Heterogeneous Graph Transformer
    4. Adaptive graph blend (schema + learned adjacency)
    5. GlobalContextModule (cross-attention on global pooling)
    6. Quantile ForecastHead per target node type
    """

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        embed_dim: int = 128,
        num_hgt_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_memory: bool = True,
        memory_dim: int = 64,
        use_adaptive_graph: bool = True,
        adaptive_top_k: int = 10,
        history_steps: int = 24,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self.embed_dim = embed_dim
        self._history_steps = history_steps
        self.use_memory = use_memory
        self.use_adaptive_graph = use_adaptive_graph

        # 1. Node encoders
        self.node_encoders = nn.ModuleDict(
            {
                nt.name: NodeEncoder(
                    in_dim=max(nt.feature_dim, 1),
                    embed_dim=embed_dim,
                )
                for nt in schema.node_types
            }
        )

        # 2. TGN memory (initialised lazily per node type)
        self._memory_modules: dict[str, TGNMemory] = {}
        if use_memory:
            self.memory_modules = nn.ModuleDict()
            for nt in schema.node_types:
                # num_nodes unknown until fit; use placeholder 1 and re-init
                self.memory_modules[nt.name] = TGNMemory(1, memory_dim)
            self.mem_proj = nn.ModuleDict(
                {
                    nt.name: nn.Linear(memory_dim, embed_dim)
                    for nt in schema.node_types
                }
            )

        # 3. HGT
        self.hgt = HGT(
            node_types=schema.node_type_names,
            edge_triplets=schema.edge_triplets(),
            in_dim=embed_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_hgt_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 4. Adaptive graph learners (one per node type, used intra-type)
        if use_adaptive_graph:
            self.adaptive_learners: dict[str, AdaptiveGraphLearner] = {}

        # 5. Global context
        self.global_ctx = GlobalContextModule(embed_dim, num_heads=num_heads)

        # 6. Forecast heads
        self.forecast_heads = nn.ModuleDict(
            {
                nt.name: ForecastHead(embed_dim, nt.target_dim, horizons)
                for nt in schema.forecast_node_types
            }
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, graph: HeterogeneousTemporalGraph) -> None:
        if self.use_memory:
            self._reinit_memory(graph)
        if self.use_adaptive_graph:
            self._reinit_adaptive(graph)

    def on_predict_start(self, graph: HeterogeneousTemporalGraph) -> None:
        if self.use_memory:
            self._reinit_memory(graph)
        if self.use_adaptive_graph:
            self._reinit_adaptive(graph)

    def _reinit_memory(self, graph: HeterogeneousTemporalGraph) -> None:
        device = next(self.parameters()).device
        for nt in graph.schema.node_types:
            n = graph.num_nodes(nt.name)
            if n == 0:
                continue
            mem = TGNMemory(n, self.memory_modules[nt.name].memory_dim).to(device)
            self.memory_modules[nt.name] = mem

    def _reinit_adaptive(self, graph: HeterogeneousTemporalGraph) -> None:
        device = next(self.parameters()).device
        for nt in graph.schema.node_types:
            n = graph.num_nodes(nt.name)
            if n < 2:
                continue
            self.adaptive_learners[nt.name] = AdaptiveGraphLearner(
                n, embed_dim=32, top_k=min(10, n)
            ).to(device)

    def reset_memory(self) -> None:
        if self.use_memory:
            for mem in self.memory_modules.values():
                mem.reset_memory()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        temporal_encoding: Tensor,
        memory: dict[str, Tensor] | None = None,
    ) -> dict[str, dict[int, Tensor]]:
        # 1. Encode node features
        x_dict: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            h = self.node_encoders[nt.name](feat, temporal_encoding)  # (N, E)

            # Inject TGN memory
            if self.use_memory and nt.name in self.memory_modules:
                mem_module = self.memory_modules[nt.name]
                if mem_module.num_nodes == h.shape[0]:
                    ids = torch.arange(h.shape[0], device=h.device)
                    mem_h = mem_module(ids)
                    h = h + self.mem_proj[nt.name](mem_h)

            x_dict[nt.name] = h

        # 2. Build HGT edge_index_dict
        hgt_edges: dict[str, tuple[str, str, Tensor]] = {}
        for et in graph.schema.edge_types:
            ei = edge_index.get(et.name)
            if ei is not None and ei.numel() > 0:
                hgt_edges[et.name] = (et.src_type, et.dst_type, ei)

        # 3. HGT message passing
        x_dict = self.hgt(x_dict, hgt_edges)

        # 4. Adaptive graph augmentation (intra-type, optional)
        if self.use_adaptive_graph:
            for nt_name, learner in self.adaptive_learners.items():
                if nt_name not in x_dict:
                    continue
                h = x_dict[nt_name]
                adj = learner()
                # Simple graph convolution with learned adj
                x_dict[nt_name] = h + torch.softmax(adj, dim=-1) @ h * 0.1

        # 5. Global context injection
        node_types_present = [nt for nt in graph.schema.node_type_names if nt in x_dict]
        embeds = [x_dict[nt] for nt in node_types_present]
        if embeds:
            _, updated = self.global_ctx(embeds)
            for nt, upd in zip(node_types_present, updated):
                x_dict[nt] = upd

        # 6. Forecast heads
        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            if nt.name not in x_dict:
                continue
            out[nt.name] = self.forecast_heads[nt.name](x_dict[nt.name])

        return out

    def required_history_steps(self) -> int:
        return self._history_steps

    def supports_missing_nodes(self) -> bool:
        return True


@ModelRegistry.register("hgt")
class HGTModel(BaseForecastModel):
    """Standalone HGT model — lighter than full GraphOracle."""

    def __init__(
        self,
        schema: GraphSchema,
        horizons: list[int],
        embed_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        history_steps: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(schema, horizons, **kwargs)
        self._history_steps = history_steps

        self.input_proj = nn.ModuleDict(
            {
                nt.name: nn.Linear(max(nt.feature_dim, 1), embed_dim)
                for nt in schema.node_types
            }
        )
        self.hgt = HGT(
            node_types=schema.node_type_names,
            edge_triplets=schema.edge_triplets(),
            in_dim=embed_dim,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {
                nt.name: nn.ModuleDict(
                    {f"h{h}": nn.Linear(embed_dim, nt.target_dim) for h in horizons}
                )
                for nt in schema.forecast_node_types
            }
        )

    def forward(
        self,
        graph: HeterogeneousTemporalGraph,
        node_features: dict[str, Tensor],
        edge_index: dict[str, Tensor],
        temporal_encoding: Tensor,
        memory: dict[str, Tensor] | None = None,
    ) -> dict[str, dict[int, Tensor]]:
        x_dict: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = node_features.get(nt.name)
            if feat is None:
                continue
            if feat.ndim == 3:
                feat = feat.mean(dim=1)   # temporal pool
            x_dict[nt.name] = self.input_proj[nt.name](feat)

        hgt_edges = {
            et.name: (et.src_type, et.dst_type, edge_index[et.name])
            for et in graph.schema.edge_types
            if et.name in edge_index and edge_index[et.name].numel() > 0
        }
        x_dict = self.hgt(x_dict, hgt_edges)

        out: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            if nt.name not in x_dict:
                continue
            h = x_dict[nt.name]
            out[nt.name] = {
                horizon: self.heads[nt.name][f"h{horizon}"](h)
                for horizon in self.horizons
            }
        return out

    def required_history_steps(self) -> int:
        return self._history_steps
