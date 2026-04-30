"""GraphOracle — the main user-facing facade class."""

from __future__ import annotations

from typing import Any

import torch

from graphoracle.benchmarks.evaluator import EvalResult, Evaluator
from graphoracle.explainability.attention_viz import AttentionExtractor
from graphoracle.explainability.causal_trace import CausalChain, CausalTracer
from graphoracle.explainability.node_importance import GNNExplainer, NodeImportance
from graphoracle.forecasting.cascade import CascadeResult, CascadeSimulator
from graphoracle.forecasting.engine import ForecastEngine
from graphoracle.forecasting.horizon import ForecastResult
from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.schema import GraphSchema
from graphoracle.models.base import BaseForecastModel
from graphoracle.models.registry import ModelRegistry
from graphoracle.training.trainer import TrainingConfig, TrainingHistory, Trainer
from graphoracle.utils.exceptions import ModelNotRegisteredError
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


class _ExplainProxy:
    """
    Returned by ``oracle.explain()``.  Wraps GNNExplainer, AttentionExtractor,
    and CausalTracer for a specific node.
    """

    def __init__(
        self,
        model: BaseForecastModel,
        graph: HeterogeneousTemporalGraph,
        node_id: str,
        node_type: str,
        horizon: int,
        device: str | torch.device,
    ) -> None:
        self._model = model
        self._graph = graph
        self._node_id = node_id
        self._node_type = node_type
        self._horizon = horizon
        self._device = device
        self._importance: NodeImportance | None = None

    def _get_importance(self) -> NodeImportance:
        if self._importance is None:
            explainer = GNNExplainer(self._model, device=self._device)
            self._importance = explainer.explain_node(
                self._graph, self._node_id, self._node_type, self._horizon
            )
        return self._importance

    def top_influencers(self, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k neighbour nodes by attribution score."""
        imp = self._get_importance()
        sorted_neighbours = sorted(
            imp.neighbour_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_neighbours[:k]

    def feature_importances(self) -> dict[str, float]:
        """Return feature-level importance scores."""
        return self._get_importance().feature_scores

    def plot_attention_heatmap(self) -> None:
        extractor = AttentionExtractor(self._model)
        extractor.register_hooks()
        engine = ForecastEngine(self._model, device=self._device)
        engine.run(self._graph)
        extractor.plot_heatmap(self._node_type)
        extractor.remove_hooks()

    def causal_trace(
        self,
        anomaly_timestamp: str | None = None,
    ) -> list[CausalChain]:
        tracer = CausalTracer()
        return tracer.trace(
            self._graph,
            anomaly_node_id=self._node_id,
            anomaly_node_type=self._node_type,
            anomaly_timestamp=anomaly_timestamp,
        )


class GraphOracle:
    """
    Main user-facing API for graph-based time series forecasting.

    Supports three levels of model customisation:

    Level 1 — Swap backbone
    -----------------------
    oracle = GraphOracle(backbone=MyGNN, backbone_kwargs={...})

    Level 2 — Full custom model
    ---------------------------
    oracle = GraphOracle(model="my_model")  # after registering

    Level 3 — Config dict
    ---------------------
    oracle = GraphOracle.from_config({...})

    Parameters
    ----------
    model           : model name in ModelRegistry, or "graphoracle" (default)
    horizons        : list of forecast horizons (steps)
    backbone        : optional custom nn.Module backbone (Level 1)
    backbone_kwargs : kwargs forwarded to backbone constructor
    device          : "cpu" | "cuda" | "mps"
    model_kwargs    : extra kwargs for the model constructor
    """

    def __init__(
        self,
        model: str = "graphoracle",
        horizons: list[int] | None = None,
        backbone: type | None = None,
        backbone_kwargs: dict[str, Any] | None = None,
        device: str = "cpu",
        **model_kwargs: Any,
    ) -> None:
        self._model_name = model
        self.horizons = horizons or [1, 6, 24]
        self._backbone = backbone
        self._backbone_kwargs = backbone_kwargs or {}
        self.device = device
        self._model_kwargs = model_kwargs
        self._model: BaseForecastModel | None = None
        self._trainer: Trainer | None = None
        self._engine: ForecastEngine | None = None
        self._schema: GraphSchema | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        graph: HeterogeneousTemporalGraph,
        epochs: int = 100,
        config: TrainingConfig | None = None,
        val_graph: HeterogeneousTemporalGraph | None = None,
    ) -> TrainingHistory:
        """Train the model on *graph*."""
        self._schema = graph.schema
        self._model = self._build_model(graph.schema)
        cfg = config or TrainingConfig(epochs=epochs, device=self.device)
        self._trainer = Trainer(self._model, cfg)
        history = self._trainer.fit(graph, val_graph)
        self._engine = ForecastEngine(self._model, device=self.device)
        log.info(f"Training complete. Best val loss: {history.best_val_loss():.4f}")
        return history

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        graph: HeterogeneousTemporalGraph,
        reference_time: Any = None,
        horizon_unit_hours: int = 1,
    ) -> ForecastResult:
        """Run inference on *graph* and return a ForecastResult."""
        self._ensure_fitted()
        return self._engine.run(graph, reference_time, horizon_unit_hours)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        graph: HeterogeneousTemporalGraph,
        metrics: list[str] | None = None,
        per_node_type: bool = True,
        per_horizon: bool = True,
        split: str = "test",
    ) -> EvalResult:
        """Evaluate model on *graph* with standard metrics."""
        self._ensure_fitted()
        evaluator = Evaluator(self, metrics or ["MAE", "RMSE", "MAPE"])
        return evaluator.run(graph, model_name=self._model_name)

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(
        self,
        graph: HeterogeneousTemporalGraph,
        node_id: str,
        horizon: int,
        node_type: str | None = None,
    ) -> _ExplainProxy:
        """Return an explanation proxy for *node_id* at *horizon*."""
        self._ensure_fitted()
        if node_type is None:
            node_type = self._infer_node_type(graph, node_id)
        return _ExplainProxy(
            self._model, graph, node_id, node_type, horizon, self.device
        )

    # ------------------------------------------------------------------
    # Cascade simulation
    # ------------------------------------------------------------------

    def simulate_cascade(
        self,
        graph: HeterogeneousTemporalGraph,
        shocks: list[dict[str, Any]],
        steps: int = 24,
    ) -> CascadeResult:
        """Simulate shock propagation through the graph."""
        self._ensure_fitted()
        simulator = CascadeSimulator(self._model, steps=steps, device=self.device)
        return simulator.simulate(graph, shocks)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        from graphoracle.utils.io import save_model
        self._ensure_fitted()
        save_model(self._model, path)

    def load(self, path: str, schema: GraphSchema) -> "GraphOracle":
        from graphoracle.utils.io import load_model
        self._schema = schema
        self._model = self._build_model(schema)
        load_model(self._model, path)
        self._engine = ForecastEngine(self._model, device=self.device)
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GraphOracle":
        """Instantiate from a config dict (useful for experiments)."""
        return cls(
            model=config.get("model", "graphoracle"),
            horizons=config.get("horizons", [1, 6, 24]),
            device=config.get("device", "cpu"),
            **config.get("model_kwargs", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, schema: GraphSchema) -> BaseForecastModel:
        if self._backbone is not None:
            return self._build_backbone_model(schema)

        model_cls = ModelRegistry.get(self._model_name)
        return model_cls(
            schema=schema,
            horizons=self.horizons,
            **self._model_kwargs,
        )

    def _build_backbone_model(self, schema: GraphSchema) -> BaseForecastModel:
        """Wrap a custom backbone (Level 1 API) in a BaseForecastModel shell.

        Adds a per-type Linear projector from each node type's feature dim to
        backbone ``in_channels`` so the user never has to match schema dims
        manually.  ``in_channels`` in backbone_kwargs becomes the *projected*
        dimension fed into the backbone; the raw schema feature dim is handled
        transparently.
        """
        import torch.nn as nn

        backbone_cls = self._backbone
        backbone_kwargs = self._backbone_kwargs
        horizons = self.horizons
        in_channels = backbone_kwargs.get("in_channels", 64)

        class _BackboneModel(BaseForecastModel):
            def __init__(self, schema: GraphSchema, horizons: list[int], **kw: Any) -> None:
                super().__init__(schema, horizons)
                self.backbone = backbone_cls(**backbone_kwargs)
                out_channels = backbone_kwargs.get("out_channels", 64)
                # Project each node type's raw feature dim → in_channels before backbone
                self.projectors = nn.ModuleDict(
                    {
                        nt.name: nn.Linear(max(nt.feature_dim, 1), in_channels)
                        for nt in schema.node_types
                    }
                )
                self.heads = nn.ModuleDict(
                    {
                        nt.name: nn.ModuleDict(
                            {
                                f"h{h}": nn.Linear(out_channels, max(nt.target_dim, 1))
                                for h in horizons
                            }
                        )
                        for nt in schema.forecast_node_types
                    }
                )

            def forward(
                self,
                graph: HeterogeneousTemporalGraph,
                node_features: dict[str, torch.Tensor],
                edge_index: dict[str, torch.Tensor],
                temporal_encoding: torch.Tensor,
                memory: dict[str, torch.Tensor] | None = None,
            ) -> dict[str, dict[int, torch.Tensor]]:
                out = {}
                # Use first available edge_index as the backbone edge_index
                ei = next(iter(edge_index.values()), torch.empty((2, 0), dtype=torch.long))
                for nt in graph.schema.forecast_node_types:
                    feat = node_features.get(nt.name)
                    if feat is None:
                        continue
                    if feat.ndim == 3:
                        feat = feat.mean(1)  # (N, T, F) → (N, F) temporal pool
                    projected = self.projectors[nt.name](feat)  # (N, in_channels)
                    h = self.backbone(projected, ei)
                    out[nt.name] = {
                        horizon: self.heads[nt.name][f"h{horizon}"](h)
                        for horizon in self.horizons
                    }
                return out

            def required_history_steps(self) -> int:
                return 12

        return _BackboneModel(schema=schema, horizons=horizons)

    def _ensure_fitted(self) -> None:
        if self._model is None or self._engine is None:
            raise RuntimeError(
                "Model is not fitted. Call oracle.fit(graph) first."
            )

    def _infer_node_type(
        self, graph: HeterogeneousTemporalGraph, node_id: str
    ) -> str:
        for nt in graph.schema.node_types:
            if node_id in graph.all_node_ids(nt.name):
                return nt.name
        raise ValueError(f"Node '{node_id}' not found in any node type.")

    @property
    def model(self) -> BaseForecastModel:
        self._ensure_fitted()
        return self._model  # type: ignore[return-value]

    def __repr__(self) -> str:
        fitted = self._model is not None
        return (
            f"GraphOracle(model={self._model_name!r}, "
            f"horizons={self.horizons}, fitted={fitted})"
        )
