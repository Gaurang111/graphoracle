"""Adaptive graph learner — learns intra-type edges from node embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaptiveGraphLearner(nn.Module):
    """
    Learns a soft adjacency matrix for *num_nodes* nodes of one type.

    The adjacency is computed as softmax(E1 @ E2^T) where E1 and E2 are
    per-node learnable embeddings.  The result is a dense (N, N) matrix
    that is blended into the fixed schema adjacency:
        A_blend = alpha * A_schema + (1 - alpha) * A_learned

    Only A_learned is produced here; blending is done in the model forward.

    Parameters
    ----------
    num_nodes : number of nodes of this type
    embed_dim : dimension of node embedding for adjacency estimation
    top_k     : number of nearest neighbours to keep (unused in soft version)
    """

    def __init__(self, num_nodes: int, embed_dim: int = 32, top_k: int = 10) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.emb1 = nn.Embedding(num_nodes, embed_dim)
        self.emb2 = nn.Embedding(num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.emb1.weight)
        nn.init.xavier_uniform_(self.emb2.weight)

    def forward(self) -> Tensor:
        """Return (N, N) soft adjacency matrix."""
        idx = torch.arange(self.num_nodes, device=self.emb1.weight.device)
        e1 = self.emb1(idx)  # (N, D)
        e2 = self.emb2(idx)  # (N, D)
        adj = e1 @ e2.t()    # (N, N)
        return adj
