"""Synthetic data generator for any GraphSchema."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_NODES_PER_TYPE = 10


class SyntheticGenerator:
    """
    Generate a realistic synthetic HeterogeneousTemporalGraph for any schema.

    The generator creates:
    - Random node features with realistic temporal dynamics (AR(1) + noise)
    - Random sparse edges (~30% density, at least min_edges_per_type)
    - A consistent timestamp sequence

    Parameters
    ----------
    schema          : GraphSchema
    seed            : random seed for reproducibility
    ar_coeff        : AR(1) auto-regression coefficient (0.8 = smooth, 0.0 = pure noise)
    noise_std       : standard deviation of white noise
    min_edges_per_type : minimum edges per edge type
    """

    def __init__(
        self,
        schema: GraphSchema,
        seed: int = 42,
        ar_coeff: float = 0.85,
        noise_std: float = 0.1,
        min_edges_per_type: int = 2,
    ) -> None:
        self.schema = schema
        self.rng = np.random.default_rng(seed)
        self.ar_coeff = ar_coeff
        self.noise_std = noise_std
        self.min_edges_per_type = min_edges_per_type

    def generate(
        self,
        nodes_per_type: dict[str, int] | None = None,
        num_timesteps: int = 48,
        start_time: datetime | None = None,
        freq_minutes: int = 60,
    ) -> HeterogeneousTemporalGraph:
        """Generate and return a synthetic graph."""
        if nodes_per_type is None:
            nodes_per_type = {
                nt.name: DEFAULT_NODES_PER_TYPE for nt in self.schema.node_types
            }

        start_time = start_time or datetime(2024, 1, 1)
        timestamps = [
            start_time + timedelta(minutes=freq_minutes * t)
            for t in range(num_timesteps)
        ]

        builder = GraphBuilder(self.schema)

        # Generate nodes
        all_ids: dict[str, list[str]] = {}
        for nt in self.schema.node_types:
            n = nodes_per_type.get(nt.name, DEFAULT_NODES_PER_TYPE)
            ids = [f"{nt.name}_{i}" for i in range(n)]
            all_ids[nt.name] = ids

            F = max(len(nt.features), 1)
            features = self._generate_timeseries(n, num_timesteps, F)
            builder.add_nodes(nt.name, ids, features, timestamps)

        # Generate edges
        for et in self.schema.edge_types:
            src_ids = all_ids.get(et.src_type, [])
            dst_ids = all_ids.get(et.dst_type, [])
            if not src_ids or not dst_ids:
                continue
            e_src, e_dst = self._generate_edges(src_ids, dst_ids)
            ef = None
            if et.features:
                ef = self.rng.standard_normal((len(e_src), len(et.features))).astype(
                    np.float32
                )
            if e_src:
                try:
                    builder.add_edges(et.name, e_src, e_dst, ef)
                except Exception as exc:
                    log.debug(f"Skipped edge type '{et.name}': {exc}")

        graph = builder.build()
        log.info(f"Generated synthetic graph: {graph}")
        return graph

    # ------------------------------------------------------------------

    def _generate_timeseries(self, N: int, T: int, F: int) -> np.ndarray:
        """AR(1) process per node per feature → (N, T, F)."""
        data = np.zeros((N, T, F), dtype=np.float32)
        x = self.rng.standard_normal((N, F)).astype(np.float32)
        for t in range(T):
            noise = (self.rng.standard_normal((N, F)) * self.noise_std).astype(
                np.float32
            )
            x = self.ar_coeff * x + noise
            data[:, t, :] = x
        return data

    def _generate_edges(
        self,
        src_ids: list[str],
        dst_ids: list[str],
        density: float = 0.3,
    ) -> tuple[list[str], list[str]]:
        n_src, n_dst = len(src_ids), len(dst_ids)
        n_edges = max(
            self.min_edges_per_type, int(n_src * n_dst * density)
        )
        n_edges = min(n_edges, n_src * n_dst)

        pairs = set()
        attempts = 0
        while len(pairs) < n_edges and attempts < n_edges * 10:
            s = int(self.rng.integers(0, n_src))
            d = int(self.rng.integers(0, n_dst))
            if src_ids[s] != dst_ids[d]:  # avoid self-loops when src==dst type
                pairs.add((s, d))
            attempts += 1

        e_src = [src_ids[s] for s, _ in pairs]
        e_dst = [dst_ids[d] for _, d in pairs]
        return e_src, e_dst
