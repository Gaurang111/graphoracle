"""Temporal Graph Network (TGN) memory module."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TGNMemory(nn.Module):
    """
    Per-node GRU-based memory module from TGN (Rossi et al., 2020).

    Each node maintains a compressed memory state that is updated on every
    interaction and can persist across inference calls.

    Attributes
    ----------
    memory     : (N, memory_dim) current memory per node
    last_update: (N,) float — last time each node was updated
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 64,
        message_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim or memory_dim

        self.gru_cell = nn.GRUCell(self.message_dim, memory_dim)
        self.msg_aggregator = nn.Linear(memory_dim * 2, self.message_dim)
        self.time_encoder = nn.Linear(1, memory_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

    def reset_memory(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()

    def get_memory(self, node_ids: Tensor) -> Tensor:
        """Return memory slice for *node_ids* → (len(node_ids), memory_dim)."""
        return self.memory[node_ids]

    def update_memory(
        self,
        node_ids: Tensor,
        messages: Tensor,
    ) -> None:
        """
        Update memory for *node_ids* using *messages* (|node_ids|, message_dim).
        Uniquify node_ids first to handle duplicates.
        """
        unique_ids, inverse = torch.unique(node_ids, return_inverse=True)
        # Aggregate messages for duplicate ids (mean)
        agg = torch.zeros(unique_ids.shape[0], messages.shape[1], device=messages.device)
        agg.scatter_add_(0, inverse.unsqueeze(1).expand_as(messages), messages)
        counts = torch.zeros(unique_ids.shape[0], device=messages.device)
        counts.scatter_add_(0, inverse, torch.ones(node_ids.shape[0], device=messages.device))
        agg = agg / counts.unsqueeze(1).clamp(min=1)

        old_mem = self.memory[unique_ids]
        new_mem = self.gru_cell(agg, old_mem)
        self.memory[unique_ids] = new_mem.detach()

    def compute_messages(
        self,
        src_ids: Tensor,
        dst_ids: Tensor,
        timestamps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute source and destination messages from current memory.
        Returns (src_messages, dst_messages), each (E, message_dim).
        """
        src_mem = self.memory[src_ids]   # (E, D)
        dst_mem = self.memory[dst_ids]   # (E, D)
        combined = torch.cat([src_mem, dst_mem], dim=-1)
        msg = self.msg_aggregator(combined)
        return msg, msg

    def forward(self, node_ids: Tensor) -> Tensor:
        """Return current memory for *node_ids*."""
        return self.get_memory(node_ids)


class TemporalGraphNetwork(nn.Module):
    """
    Simplified TGN backbone that wraps TGNMemory with a GNN layer.

    The memory is updated per-batch and provides additional context
    to the message-passing step.
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int = 128,
        memory_dim: int = 64,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.memory = TGNMemory(num_nodes, memory_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        layers = []
        for _ in range(num_gnn_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.gnn_layers = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_ids: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x          : (N, in_dim) node features
        edge_index : (2, E)
        node_ids   : (N,) integer IDs for memory lookup

        Returns
        -------
        (N, hidden_dim) enriched node representations
        """
        h = self.input_proj(x)

        if node_ids is not None:
            mem = self.memory(node_ids)
            h = h + self.memory_proj(mem)

        h = self.gnn_layers(h)

        # Simple graph aggregation (mean of neighbours)
        if edge_index.numel() > 0:
            src, dst = edge_index
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]), h[src])
            deg = torch.zeros(h.shape[0], device=h.device)
            deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=h.device))
            deg = deg.clamp(min=1).unsqueeze(1)
            h = h + agg / deg

        return h

    def reset_memory(self) -> None:
        self.memory.reset_memory()
