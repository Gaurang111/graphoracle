"""Tests for ForecastEngine, ForecastResult, CascadeSimulator, and AnomalyDetector."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphoracle.forecasting.cascade import CascadeResult, CascadeSimulator, Shock
from graphoracle.forecasting.engine import ForecastEngine
from graphoracle.forecasting.horizon import ForecastResult, NodeForecast
from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType
from graphoracle.models.baselines import LSTMBaseline
from graphoracle.training.trainer import TrainingConfig, Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema():
    return GraphSchema(
        node_types=[NodeType("sensor", features=["v", "t"], targets=["v"])],
        edge_types=[EdgeType("link", "sensor", "sensor")],
    )


def _graph(schema, N=4, T=8):
    feat = np.random.randn(N, T, 2).astype(np.float32)
    ids = [f"s{i}" for i in range(N)]
    return (
        GraphBuilder(schema)
        .add_nodes("sensor", ids, feat)
        .add_edges("link", [ids[0], ids[1]], [ids[1], ids[2]])
        .build()
    )


def _trained_model(schema, graph):
    model = LSTMBaseline(schema, horizons=[1, 3], hidden=16, num_layers=1)
    trainer = Trainer(model, TrainingConfig(epochs=2))
    trainer.fit(graph)
    return model


# ---------------------------------------------------------------------------
# ForecastEngine
# ---------------------------------------------------------------------------

class TestForecastEngine:
    def test_run_returns_forecast_result(self):
        schema = _schema()
        graph = _graph(schema)
        model = _trained_model(schema, graph)
        engine = ForecastEngine(model)
        result = engine.run(graph)
        assert isinstance(result, ForecastResult)

    def test_run_inference_returns_dict(self):
        schema = _schema()
        graph = _graph(schema)
        model = _trained_model(schema, graph)
        engine = ForecastEngine(model)
        raw = engine._run_inference(graph)
        assert "sensor" in raw
        assert 1 in raw["sensor"]

    def test_predictions_correct_shape(self):
        schema = _schema()
        graph = _graph(schema, N=4)
        model = _trained_model(schema, graph)
        engine = ForecastEngine(model)
        raw = engine._run_inference(graph)
        pred = raw["sensor"][1]
        assert pred.shape[0] == 4  # N nodes


# ---------------------------------------------------------------------------
# ForecastResult
# ---------------------------------------------------------------------------

class TestForecastResult:
    def _make_result(self):
        schema = _schema()
        graph = _graph(schema, N=3)
        model = _trained_model(schema, graph)
        engine = ForecastEngine(model)
        return engine.run(graph), graph

    def test_all_nodes(self):
        result, graph = self._make_result()
        nodes = result.all_nodes()
        assert len(nodes) > 0

    def test_get_node_forecasts(self):
        result, graph = self._make_result()
        node_id = graph.all_node_ids("sensor")[0]
        forecasts = result.get(node_id)
        assert isinstance(forecasts, list)
        assert len(forecasts) > 0

    def test_summary_is_string(self):
        result, _ = self._make_result()
        s = result.summary()
        assert isinstance(s, str)

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        result, _ = self._make_result()
        df = result.to_dataframe()
        assert len(df) > 0
        assert "node_id" in df.columns


# ---------------------------------------------------------------------------
# NodeForecast
# ---------------------------------------------------------------------------

class TestNodeForecast:
    def test_construction(self):
        nf = NodeForecast(
            node_id="s0",
            node_type="sensor",
            horizon=1,
            point=torch.tensor([1.0]),
            lower=None,
            upper=None,
        )
        assert nf.node_id == "s0"
        assert nf.horizon == 1


# ---------------------------------------------------------------------------
# CascadeResult
# ---------------------------------------------------------------------------

class TestCascadeResult:
    def _make_result(self):
        step_impacts = [
            {"sensor": {"s0": 0.5, "s1": 0.1}},
            {"sensor": {"s0": 0.3, "s1": 0.05}},
        ]
        return CascadeResult(steps=2, impact_by_step=step_impacts)

    def test_highest_risk_nodes(self):
        r = self._make_result()
        top = r.highest_risk_nodes(k=2)
        assert len(top) == 2
        # s0 has highest total impact
        assert top[0][0] == "s0"

    def test_impact_delta(self):
        r = self._make_result()
        d = r.impact_delta(0)
        assert "sensor" in d
        assert "s0" in d["sensor"]

    def test_impact_delta_out_of_range(self):
        r = self._make_result()
        assert r.impact_delta(100) == {}

    def test_estimated_recovery_hours(self):
        step_impacts = [
            {"sensor": {"s0": 10.0}},
            {"sensor": {"s0": 0.05}},  # below threshold
        ]
        r = CascadeResult(steps=2, impact_by_step=step_impacts)
        hours = r.estimated_recovery_hours(threshold=0.1)
        assert hours == 1.0


# ---------------------------------------------------------------------------
# CascadeSimulator
# ---------------------------------------------------------------------------

class TestCascadeSimulator:
    def test_simulate_returns_cascade_result(self):
        schema = _schema()
        graph = _graph(schema, N=4)
        model = _trained_model(schema, graph)
        sim = CascadeSimulator(model, steps=2)
        result = sim.simulate(graph, shocks=[{"node": "s0", "feature": "v", "change": 2.0}])
        assert isinstance(result, CascadeResult)
        assert result.steps == 2

    def test_highest_risk_nodes_returned(self):
        schema = _schema()
        graph = _graph(schema, N=4)
        model = _trained_model(schema, graph)
        sim = CascadeSimulator(model, steps=2)
        result = sim.simulate(graph, shocks=[{"node": "s0", "feature": "v", "change": 5.0}])
        top = result.highest_risk_nodes(k=4)
        assert isinstance(top, list)


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def test_detect_returns_anomaly_result(self):
        from graphoracle.forecasting.anomaly import AnomalyDetector, AnomalyResult
        from graphoracle.oracle import GraphOracle

        schema = _schema()
        graph = _graph(schema)
        oracle = GraphOracle(model="lstm", horizons=[1])
        oracle.fit(graph, config=TrainingConfig(epochs=1))

        detector = AnomalyDetector(oracle, threshold=0.0)
        result = detector.detect(graph)
        assert isinstance(result, AnomalyResult)
        assert "sensor" in result.anomalies

    def test_summary_is_string(self):
        from graphoracle.forecasting.anomaly import AnomalyResult

        r = AnomalyResult(
            anomalies={"sensor": {1: ["s0", "s1"]}},
            scores={"sensor": {1: torch.tensor([0.5, 0.5])}},
        )
        s = r.summary()
        assert "sensor" in s
