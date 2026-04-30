"""Heterogeneous Graph Transformer (HGT) — Hu et al., 2020."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HGTLayer(nn.Module):
    """
    Single HGT message-passing layer.

    Uses type-specific Q/K/V projections and per-edge-triplet attention
    priors (μ) to handle heterogeneous node and edge types.
    """

    def __init__(
        self,
        node_types: list[str],
        edge_triplets: list[tuple[str, str, str]],
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(out_dim // num_heads, 1)
        self.out_dim = out_dim
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.ModuleDict({t: nn.Linear(in_dim, out_dim, bias=False) for t in node_types})
        self.W_K = nn.ModuleDict({t: nn.Linear(in_dim, out_dim, bias=False) for t in node_types})
        self.W_V = nn.ModuleDict({t: nn.Linear(in_dim, out_dim, bias=False) for t in node_types})
        self.W_skip = nn.ModuleDict({t: nn.Linear(in_dim, out_dim) for t in node_types})
        self.norms = nn.ModuleDict({t: nn.LayerNorm(out_dim) for t in node_types})

        mu_params: dict[str, nn.Parameter] = {}
        msg_layers: dict[str, nn.Module] = {}
        for src_t, e_name, dst_t in edge_triplets:
            key = f"{src_t}__{e_name}__{dst_t}"
            mu_params[key] = nn.Parameter(torch.zeros(num_heads))
            msg_layers[key] = nn.Linear(out_dim, out_dim, bias=False)

        self.mu = nn.ParameterDict(mu_params)
        self.W_msg = nn.ModuleDict(msg_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, tuple[str, str, Tensor]],
    ) -> dict[str, Tensor]:
        H, D_h = self.num_heads, self.head_dim

        Q: dict[str, Tensor] = {}
        K: dict[str, Tensor] = {}
        V: dict[str, Tensor] = {}
        for t, x in x_dict.items():
            N = x.shape[0]
            Q[t] = self.W_Q[t](x).reshape(N, H, D_h)
            K[t] = self.W_K[t](x).reshape(N, H, D_h)
            V[t] = self.W_V[t](x).reshape(N, H, D_h)

        agg: dict[str, Tensor] = {
            t: torch.zeros(x.shape[0], self.out_dim, device=x.device)
            for t, x in x_dict.items()
        }

        for edge_name, (src_type, dst_type, edge_index) in edge_index_dict.items():
            if edge_index.numel() == 0:
                continue
            if src_type not in x_dict or dst_type not in x_dict:
                continue

            key = f"{src_type}__{edge_name}__{dst_type}"
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            E = src_idx.shape[0]
            N_dst = x_dict[dst_type].shape[0]

            q = Q[dst_type][dst_idx]  # (E, H, D_h)
            k = K[src_type][src_idx]
            v = V[src_type][src_idx]

            mu = self.mu.get(key)
            if mu is None:
                mu = torch.zeros(H, device=q.device)
            attn_scale = torch.sigmoid(mu)  # (H,)

            attn = (q * k).sum(-1) * self.scale * attn_scale.unsqueeze(0)  # (E, H)
            attn = self._scatter_softmax(attn, dst_idx, N_dst)
            attn = self.dropout(attn)

            msg = (attn.unsqueeze(-1) * v).reshape(E, self.out_dim)

            if key in self.W_msg:
                msg = self.W_msg[key](msg)

            agg[dst_type].scatter_add_(
                0, dst_idx.unsqueeze(1).expand_as(msg), msg
            )

        out: dict[str, Tensor] = {}
        for t, x in x_dict.items():
            h = agg[t] + self.W_skip[t](x)
            out[t] = self.norms[t](F.relu(h))

        return out

    @staticmethod
    def _scatter_softmax(attn: Tensor, dst: Tensor, N_dst: int) -> Tensor:
        """Per-destination softmax over incoming attention scores (E, H) → (E, H)."""
        out = torch.zeros_like(attn)
        for j in range(N_dst):
            mask = dst == j
            if mask.any():
                out[mask] = F.softmax(attn[mask], dim=0)
        return out


class HGT(nn.Module):
    """
    Multi-layer Heterogeneous Graph Transformer.

    Parameters
    ----------
    node_types     : list of node type names
    edge_triplets  : list of (src_type, edge_name, dst_type)
    in_dim         : input feature dimension
    hidden_dim     : hidden dimension for intermediate layers
    out_dim        : output dimension
    num_layers     : number of HGT layers
    num_heads      : number of attention heads (out_dim must be divisible)
    dropout        : attention dropout probability
    """

    def __init__(
        self,
        node_types: list[str],
        edge_triplets: list[tuple[str, str, str]],
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(
                HGTLayer(node_types, edge_triplets, d_in, d_out, num_heads, dropout)
            )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[str, tuple[str, str, Tensor]],
    ) -> dict[str, Tensor]:
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        return x_dict
