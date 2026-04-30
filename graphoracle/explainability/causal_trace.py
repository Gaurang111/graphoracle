"""Causal chain tracing through the graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalChain:
    """A single causal path ending at an anomalous node."""

    target_node: str
    target_type: str
    chain: list[tuple[str, str]]  # list of (node_id, node_type)
    edge_types: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __repr__(self) -> str:
        path = " → ".join(f"{nid}({nt})" for nid, nt in self.chain)
        return f"CausalChain({path} → {self.target_node}, conf={self.confidence:.2f})"


class CausalTracer:
    """
    Trace causal paths leading to an anomalous node.

    Performs a backward graph walk from the anomaly node, following
    incoming edges up to *max_hops* steps.  Confidence is approximated
    by feature magnitude along the path (a heuristic, not a structural
    causal model).

    Parameters
    ----------
    max_hops : maximum number of hops to trace back
    """

    def __init__(self, max_hops: int = 4) -> None:
        self.max_hops = max_hops

    def trace(
        self,
        graph: Any,
        anomaly_node_id: str,
        anomaly_node_type: str,
        anomaly_timestamp: str | None = None,
    ) -> list[CausalChain]:
        """
        Return causal chains leading to *anomaly_node_id*.

        Parameters
        ----------
        graph              : HeterogeneousTemporalGraph
        anomaly_node_id    : node to explain
        anomaly_node_type  : type of the anomalous node
        anomaly_timestamp  : optional timestamp string (not used in heuristic tracing)
        """
        chains: list[CausalChain] = []
        visited: set[str] = set()

        ids = graph.all_node_ids(anomaly_node_type)
        if anomaly_node_id not in ids:
            return chains

        target_idx = ids.index(anomaly_node_id)

        self._trace_recursive(
            graph=graph,
            node_id=anomaly_node_id,
            node_type=anomaly_node_type,
            node_idx=target_idx,
            current_chain=[],
            current_edges=[],
            depth=0,
            visited=visited,
            chains=chains,
            target_node=anomaly_node_id,
            target_type=anomaly_node_type,
        )

        return chains

    def _trace_recursive(
        self,
        graph: Any,
        node_id: str,
        node_type: str,
        node_idx: int,
        current_chain: list[tuple[str, str]],
        current_edges: list[str],
        depth: int,
        visited: set[str],
        chains: list[CausalChain],
        target_node: str,
        target_type: str,
    ) -> None:
        key = f"{node_id}/{node_type}"
        if key in visited or depth >= self.max_hops:
            if current_chain:
                confidence = max(0.0, 1.0 - depth * 0.2)
                chains.append(
                    CausalChain(
                        target_node=target_node,
                        target_type=target_type,
                        chain=list(current_chain),
                        edge_types=list(current_edges),
                        confidence=confidence,
                    )
                )
            return

        visited.add(key)
        current_chain.append((node_id, node_type))

        found_parents = False
        for et in graph.schema.edge_types:
            if et.dst_type != node_type:
                continue
            ei = graph.get_edge_index(et.name)
            if ei.numel() == 0:
                continue
            dst_mask = ei[1] == node_idx
            src_indices = ei[0][dst_mask].tolist()
            src_ids = graph.all_node_ids(et.src_type)
            for src_idx in src_indices[:3]:  # limit branching
                if src_idx < len(src_ids):
                    src_id = src_ids[src_idx]
                    found_parents = True
                    self._trace_recursive(
                        graph=graph,
                        node_id=src_id,
                        node_type=et.src_type,
                        node_idx=src_idx,
                        current_chain=list(current_chain),
                        current_edges=list(current_edges) + [et.name],
                        depth=depth + 1,
                        visited=set(visited),
                        chains=chains,
                        target_node=target_node,
                        target_type=target_type,
                    )

        if not found_parents and current_chain:
            confidence = max(0.0, 1.0 - depth * 0.2)
            chains.append(
                CausalChain(
                    target_node=target_node,
                    target_type=target_type,
                    chain=list(current_chain),
                    edge_types=list(current_edges),
                    confidence=confidence,
                )
            )
