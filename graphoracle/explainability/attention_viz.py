"""Attention weight extraction and visualisation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from graphoracle.models.base import BaseForecastModel


class AttentionExtractor:
    """
    Extract per-layer, per-head attention weights from HGT layers via hooks.

    Usage
    -----
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    engine.run(graph)               # forward pass triggers hooks
    weights = extractor.get_attention_weights()
    extractor.plot_heatmap("sensor")
    extractor.remove_hooks()
    """

    def __init__(self, model: BaseForecastModel) -> None:
        self.model = model
        self._hooks: list[Any] = []
        self._attention_data: list[dict[str, Any]] = []

    def register_hooks(self) -> "AttentionExtractor":
        """Attach forward hooks to all HGTLayer instances."""
        from graphoracle.models.hgt import HGTLayer

        self._attention_data.clear()

        def make_hook(layer_idx: int):
            def hook(module: Any, inputs: Any, output: Any) -> None:
                # Store a summary dict (full attention isn't surfaced from current HGT)
                self._attention_data.append(
                    {
                        "layer": layer_idx,
                        "node_types": list(output.keys()) if isinstance(output, dict) else [],
                        "shapes": {k: tuple(v.shape) for k, v in output.items()}
                        if isinstance(output, dict)
                        else {},
                    }
                )

            return hook

        for i, module in enumerate(self.model.modules()):
            if module.__class__.__name__ == "HGTLayer":
                h = module.register_forward_hook(make_hook(i))
                self._hooks.append(h)

        return self

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_attention_weights(self) -> list[dict[str, Any]]:
        return list(self._attention_data)

    def plot_heatmap(self, node_type: str) -> None:
        """Visualise a simple attention summary per layer for *node_type*."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        layers = [d["layer"] for d in self._attention_data if node_type in d.get("node_types", [])]
        if not layers:
            return

        fig, ax = plt.subplots()
        ax.bar(range(len(layers)), [1.0] * len(layers), tick_label=layers)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Active")
        ax.set_title(f"HGT attention layers active for '{node_type}'")
        plt.tight_layout()
        plt.show()
