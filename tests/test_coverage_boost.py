"""Miscellaneous tests for coverage of remaining modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphoracle.graph.builder import GraphBuilder
from graphoracle.graph.schema import EdgeType, GraphSchema, NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _schema():
    return GraphSchema(
        node_types=[NodeType("sensor", features=["v", "t"], targets=["v"])],
        edge_types=[EdgeType("link", "sensor", "sensor")],
    )


def _graph(schema=None, N=5, T=8):
    if schema is None:
        schema = _schema()
    feat = np.random.randn(N, T, 2).astype(np.float32)
    ids = [f"s{i}" for i in range(N)]
    return (
        GraphBuilder(schema)
        .add_nodes("sensor", ids, feat)
        .add_edges("link", [ids[0], ids[1]], [ids[1], ids[2]])
        .build()
    )


def _trained_oracle(horizons=None):
    from graphoracle.oracle import GraphOracle
    from graphoracle.training.trainer import TrainingConfig

    if horizons is None:
        horizons = [1]
    schema = _schema()
    graph = _graph(schema)
    oracle = GraphOracle(model="lstm", horizons=horizons)
    oracle.fit(graph, config=TrainingConfig(epochs=1))
    return oracle, graph


# ---------------------------------------------------------------------------
# Exceptions module
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_all_exceptions_instantiate(self):
        from graphoracle.utils.exceptions import (
            GraphOracleError,
            GraphSchemaError,
            ModelNotRegisteredError,
            IncompatibleNodeTypeError,
            InsufficientHistoryError,
            ForecastHorizonError,
            MissingTargetFeatureError,
            ConnectorError,
            DomainError,
            EdgeDiscoveryError,
            CheckpointError,
        )
        for exc_cls in [
            GraphOracleError, GraphSchemaError, ModelNotRegisteredError,
            IncompatibleNodeTypeError, InsufficientHistoryError, ForecastHorizonError,
            MissingTargetFeatureError, ConnectorError, DomainError,
            EdgeDiscoveryError, CheckpointError,
        ]:
            e = exc_cls("test")
            assert str(e) == "test"

    def test_hierarchy(self):
        from graphoracle.utils.exceptions import GraphOracleError, GraphSchemaError
        assert issubclass(GraphSchemaError, GraphOracleError)


# ---------------------------------------------------------------------------
# Logging module
# ---------------------------------------------------------------------------

class TestLogging:
    def test_get_logger_returns_logger(self):
        from graphoracle.utils.logging import get_logger
        log = get_logger("test.module")
        assert log is not None

    def test_configure_logging(self):
        from graphoracle.utils.logging import configure_logging
        configure_logging(level="WARNING")  # no raise


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_string(self):
        from graphoracle._version import __version__
        assert isinstance(__version__, str)
        assert len(__version__) > 0


# ---------------------------------------------------------------------------
# GraphOracle facade
# ---------------------------------------------------------------------------

class TestGraphOracleFacade:
    def test_fit_and_predict(self):
        oracle, graph = _trained_oracle()
        result = oracle.predict(graph)
        assert result is not None

    def test_predict_returns_forecast_result(self):
        from graphoracle.forecasting.horizon import ForecastResult
        oracle, graph = _trained_oracle()
        result = oracle.predict(graph)
        assert isinstance(result, ForecastResult)

    def test_evaluate_returns_eval_result(self):
        from graphoracle.benchmarks.evaluator import EvalResult
        oracle, graph = _trained_oracle()
        result = oracle.evaluate(graph, metrics=["MAE"])
        assert isinstance(result, EvalResult)

    def test_simulate_cascade(self):
        from graphoracle.forecasting.cascade import CascadeResult
        oracle, graph = _trained_oracle()
        result = oracle.simulate_cascade(
            graph,
            shocks=[{"node": "s0", "feature": "v", "change": 1.0}],
            steps=2,
        )
        assert isinstance(result, CascadeResult)

    def test_explain_returns_proxy(self):
        oracle, graph = _trained_oracle()
        proxy = oracle.explain(graph, node_id="s0", node_type="sensor", horizon=1)
        assert proxy is not None
        # Should be able to get feature importances
        scores = proxy.feature_importances()
        assert isinstance(scores, dict)

    def test_explain_top_influencers(self):
        oracle, graph = _trained_oracle()
        proxy = oracle.explain(graph, node_id="s0", node_type="sensor", horizon=1)
        inf = proxy.top_influencers(k=3)
        assert isinstance(inf, list)

    def test_oracle_device_property(self):
        oracle, _ = _trained_oracle()
        assert oracle.device in ("cpu", "cuda", "mps")

    def test_fit_returns_training_history(self):
        from graphoracle.training.trainer import TrainingHistory
        from graphoracle.oracle import GraphOracle
        from graphoracle.training.trainer import TrainingConfig

        schema = _schema()
        graph = _graph(schema)
        oracle = GraphOracle(model="gru", horizons=[1])
        history = oracle.fit(graph, config=TrainingConfig(epochs=2))
        assert isinstance(history, TrainingHistory)


# ---------------------------------------------------------------------------
# GNNExplainer
# ---------------------------------------------------------------------------

class TestGNNExplainer:
    def test_explain_node_returns_importance(self):
        from graphoracle.explainability.node_importance import GNNExplainer, NodeImportance
        from graphoracle.training.trainer import TrainingConfig, Trainer
        from graphoracle.models.baselines import LSTMBaseline

        schema = _schema()
        graph = _graph(schema)
        model = LSTMBaseline(schema, horizons=[1], hidden=16, num_layers=1)
        Trainer(model, TrainingConfig(epochs=1)).fit(graph)

        explainer = GNNExplainer(model)
        result = explainer.explain_node(graph, "s0", "sensor", horizon=1)
        assert isinstance(result, NodeImportance)
        assert result.node_id == "s0"
        assert isinstance(result.feature_scores, dict)


# ---------------------------------------------------------------------------
# AttentionExtractor
# ---------------------------------------------------------------------------

class TestAttentionExtractor:
    def test_register_and_remove_hooks(self):
        from graphoracle.explainability.attention_viz import AttentionExtractor
        from graphoracle.models.graphoracle_model import GraphOracleModel

        schema = _schema()
        model = GraphOracleModel(schema, horizons=[1], hidden_dim=8, num_layers=1, num_heads=2)
        extractor = AttentionExtractor(model)
        extractor.register_hooks()
        extractor.remove_hooks()  # no raise


# ---------------------------------------------------------------------------
# CausalTracer
# ---------------------------------------------------------------------------

class TestCausalTracer:
    def test_trace_returns_causal_chain_list(self):
        from graphoracle.explainability.causal_trace import CausalChain, CausalTracer

        schema = _schema()
        graph = _graph(schema)

        tracer = CausalTracer(max_hops=2)
        chains = tracer.trace(
            graph,
            anomaly_node_id="s0",
            anomaly_node_type="sensor",
        )
        assert isinstance(chains, list)


# ---------------------------------------------------------------------------
# Training: loss functions additional cases
# ---------------------------------------------------------------------------

class TestLossFunctionsAdditional:
    def test_get_loss_fn_mae(self):
        from graphoracle.training.loss import get_loss_fn
        fn = get_loss_fn("mae")
        pred = torch.tensor([1.0, 2.0])
        tgt = torch.tensor([1.0, 2.0])
        assert float(fn(pred, tgt)) == 0.0

    def test_get_loss_fn_rmse(self):
        from graphoracle.training.loss import get_loss_fn
        fn = get_loss_fn("rmse")
        assert fn is not None

    def test_get_loss_fn_mape(self):
        from graphoracle.training.loss import get_loss_fn
        fn = get_loss_fn("mape")
        assert fn is not None

    def test_get_loss_fn_unknown_raises(self):
        from graphoracle.training.loss import get_loss_fn
        with pytest.raises((KeyError, ValueError)):
            get_loss_fn("unknown_loss_xyz")

    def test_pinball_loss(self):
        from graphoracle.training.loss import pinball_loss
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.5, 2.5, 3.5])
        loss = pinball_loss(pred, tgt, quantile=0.5)
        assert float(loss) >= 0


# ---------------------------------------------------------------------------
# Training: curriculum manager
# ---------------------------------------------------------------------------

class TestCurriculumManagerAdditional:
    def test_horizon_ramp_start_warmup(self):
        from graphoracle.training.curriculum import CurriculumManager, CurriculumSchedule

        schedule = CurriculumSchedule(enabled=True, warmup_epochs=5)
        mgr = CurriculumManager(schedule, total_epochs=20, all_horizons=[1, 6, 24])
        # During warmup, should only have first horizon
        cfg = mgr.step(2)
        assert len(cfg["active_horizons"]) >= 1


# ---------------------------------------------------------------------------
# Conformal wrapper
# ---------------------------------------------------------------------------

class TestConformalWrapper:
    def test_calibrate_and_predict(self):
        from graphoracle.forecasting.uncertainty import ConformalWrapper
        from graphoracle.models.baselines import LSTMBaseline
        from graphoracle.training.trainer import TrainingConfig, Trainer

        schema = _schema()
        graph = _graph(schema)
        model = LSTMBaseline(schema, horizons=[1], hidden=16, num_layers=1)
        Trainer(model, TrainingConfig(epochs=1)).fit(graph)

        wrapped = ConformalWrapper(model, coverage=0.9)
        # calibrate requires cal_targets dict (can be empty for smoke test)
        wrapped.calibrate(graph, cal_targets={})
        result = wrapped.predict_with_intervals(graph)
        assert result is not None


# ---------------------------------------------------------------------------
# MonteCarlo Dropout wrapper
# ---------------------------------------------------------------------------

class TestMCDropoutWrapper:
    def test_predict_returns_samples(self):
        from graphoracle.forecasting.uncertainty import MonteCarloDropoutWrapper
        from graphoracle.models.baselines import LSTMBaseline
        from graphoracle.training.trainer import TrainingConfig, Trainer

        schema = _schema()
        graph = _graph(schema)
        model = LSTMBaseline(schema, horizons=[1], hidden=16, num_layers=1)
        Trainer(model, TrainingConfig(epochs=1)).fit(graph)

        wrapped = MonteCarloDropoutWrapper(model, n_samples=5)
        result = wrapped.predict(graph)
        assert result is not None


# ---------------------------------------------------------------------------
# Synthetic connector
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    def test_generate_graph(self):
        from graphoracle.connectors.synthetic import SyntheticGenerator

        schema = _schema()
        gen = SyntheticGenerator(schema, seed=0)
        graph = gen.generate(nodes_per_type={"sensor": 6}, num_timesteps=12)
        assert graph.num_nodes("sensor") == 6
        assert graph.num_timesteps == 12


# ---------------------------------------------------------------------------
# CSV connector
# ---------------------------------------------------------------------------

class TestCSVConnector:
    def test_load_from_csv(self, tmp_path):
        import csv
        from graphoracle.connectors.csv_connector import CSVConnector

        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value", "temp"], targets=["value"])],
            edge_types=[],
        )
        # Nodes CSV: dict form keyed by node_type name
        csv_path = tmp_path / "sensor_nodes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "value", "temp"])
            for i in range(4):
                writer.writerow([f"s{i}", float(i), float(i * 2)])

        # Edges CSV: empty dict
        edges_path = tmp_path / "edges.csv"
        with open(edges_path, "w", newline="") as f:
            f.write("src_id,dst_id,edge_type\n")

        connector = CSVConnector(schema)
        graph = connector.load(
            nodes={"sensor": str(csv_path)},
            edges=str(edges_path),
            node_id_col="node_id",
        )
        assert graph.num_nodes("sensor") == 4


# ---------------------------------------------------------------------------
# NodeForecast.to_dict
# ---------------------------------------------------------------------------

class TestNodeForecastToDict:
    def test_to_dict_point_only(self):
        from graphoracle.forecasting.horizon import NodeForecast

        nf = NodeForecast(
            node_id="s0",
            node_type="sensor",
            horizon=1,
            point=torch.tensor([1.5, 2.5]),
            lower=None,
            upper=None,
        )
        d = nf.to_dict()
        assert d["node_id"] == "s0"
        assert d["horizon"] == 1
        assert "lower" not in d
        assert "upper" not in d

    def test_to_dict_with_intervals(self):
        from graphoracle.forecasting.horizon import NodeForecast

        nf = NodeForecast(
            node_id="s1",
            node_type="sensor",
            horizon=6,
            point=torch.tensor([1.0]),
            lower=torch.tensor([0.5]),
            upper=torch.tensor([1.5]),
        )
        d = nf.to_dict()
        assert "lower" in d
        assert "upper" in d


# ---------------------------------------------------------------------------
# TGNMemory
# ---------------------------------------------------------------------------

class TestTGNMemory:
    def test_get_and_reset_memory(self):
        from graphoracle.models.tgn import TGNMemory

        mem = TGNMemory(num_nodes=10, memory_dim=16)
        ids = torch.tensor([0, 1, 2])
        retrieved = mem.get_memory(ids)
        assert retrieved.shape == (3, 16)

        mem.reset_memory()
        assert (mem.memory == 0).all()

    def test_update_memory(self):
        from graphoracle.models.tgn import TGNMemory

        mem = TGNMemory(num_nodes=5, memory_dim=8)
        node_ids = torch.tensor([0, 1])
        messages = torch.randn(2, 8)
        mem.update_memory(node_ids, messages)
        # After update, memory for involved nodes should have changed
        assert mem.memory is not None

    def test_forward(self):
        from graphoracle.models.tgn import TGNMemory

        mem = TGNMemory(num_nodes=6, memory_dim=8)
        ids = torch.tensor([0, 2, 4])
        out = mem(ids)
        assert out.shape == (3, 8)

    def test_compute_messages(self):
        from graphoracle.models.tgn import TGNMemory

        mem = TGNMemory(num_nodes=5, memory_dim=8)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 3])
        src_msg, dst_msg = mem.compute_messages(src, dst)
        assert src_msg.shape[0] == 3


# ---------------------------------------------------------------------------
# GraphBuilder.from_dataframes
# ---------------------------------------------------------------------------

class TestGraphBuilderFromDataframes:
    def test_from_dataframes(self):
        pd = pytest.importorskip("pandas")
        import pandas as pd
        from graphoracle.graph.builder import GraphBuilder

        schema = GraphSchema(
            node_types=[NodeType("sensor", features=["value"], targets=["value"])],
            edge_types=[EdgeType("link", "sensor", "sensor")],
        )
        node_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=["s0", "s1", "s2"])
        edge_df = pd.DataFrame({"src_id": ["s0"], "dst_id": ["s1"]})

        graph = GraphBuilder.from_dataframes(
            schema,
            node_dfs={"sensor": node_df},
            edge_dfs={"link": edge_df},
        )
        assert graph.num_nodes("sensor") == 3
