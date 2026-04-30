"""External event injection utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EventSpec:
    """Specification for an external event."""

    event_type: str
    affected_nodes: list[str]
    features: dict[str, float]
    start: Any
    end: Any


class EventInjector:
    """
    Apply external events to a graph's node features.

    Events are stored inside the graph and can optionally modify
    node feature tensors to reflect the event (e.g. setting a severity
    flag to 1.0 for the affected timesteps).
    """

    def __init__(self, event_specs: list[EventSpec] | None = None) -> None:
        self.event_specs: list[EventSpec] = event_specs or []

    def add_event(self, spec: EventSpec) -> None:
        self.event_specs.append(spec)

    def apply(self, graph: HeterogeneousTemporalGraph) -> HeterogeneousTemporalGraph:
        """
        Inject all registered events into *graph*.

        Each event is stored via graph.inject_event().  Feature tensors
        are not mutated by default — subclass and override _apply_feature_mask
        for that behaviour.
        """
        for spec in self.event_specs:
            graph.inject_event(
                event_type=spec.event_type,
                affected_nodes=spec.affected_nodes,
                features=spec.features,
                start=spec.start,
                end=spec.end,
            )
            log.info(
                f"Injected event '{spec.event_type}' affecting "
                f"{len(spec.affected_nodes)} nodes."
            )
        return graph

    def build_event_tensor(
        self,
        graph: HeterogeneousTemporalGraph,
        node_type: str,
        feature_names: list[str],
    ) -> Tensor:
        """
        Build an (N, T, len(feature_names)) event feature tensor for *node_type*.

        For each event that affects nodes of *node_type*, set the
        corresponding feature to the event value for all affected timesteps.
        """
        base_feat = graph.get_node_features(node_type)
        N, T, _ = base_feat.shape
        ids = graph.all_node_ids(node_type)
        id_to_idx = {nid: i for i, nid in enumerate(ids)}

        event_feat = torch.zeros(N, T, len(feature_names))

        for event in graph.events:
            for nid in event.affected_nodes:
                idx = id_to_idx.get(nid)
                if idx is None:
                    continue
                for fi, fname in enumerate(feature_names):
                    if fname in event.features:
                        event_feat[idx, :, fi] = float(event.features[fname])

        return event_feat
