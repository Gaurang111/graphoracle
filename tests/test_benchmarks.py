"""Tests for benchmarks and training utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphoracle.benchmarks.datasets import ETTDataset, MetrLA, NREL118, PemsBay
from graphoracle.benchmarks.evaluator import (
    EvalResult,
    Evaluator,
    mae,
    mape,
    rmse,
    crps,
)
from graphoracle.oracle import GraphOracle
from graphoracle.training.checkpointing import CheckpointManager, CheckpointState
from graphoracle.training.curriculum import CurriculumManager, CurriculumSchedule
from graphoracle.training.loss import (
    MultiHorizonLoss,
    mae_loss,
    mape_loss,
    quantile_loss,
    rmse_loss,
)
from graphoracle.training.trainer import TrainingConfig, TrainingHistory, Trainer


# ---------------------------------------------------------------------------
# Benchmark datasets (synthetic / placeholder)
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_metr_la_loads(self):
        graph = MetrLA.load(num_timesteps=48)
        assert graph.num_nodes("traffic_sensor") == MetrLA.N_SENSORS
        assert graph.num_timesteps == 48

    def test_pems_bay_loads(self):
        graph = PemsBay.load(num_timesteps=48)
        assert graph.num_nodes("traffic_sensor") == PemsBay.N_SENSORS

    def test_nrel118_loads(self):
        graph = NREL118.load(num_timesteps=48)
        assert graph.num_nodes("substation") == 118

    def test_ett_loads_synthetic(self):
        graph = ETTDataset.load(split="ETTh1", num_timesteps=48)
        assert graph.num_timesteps == 48


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mae(self):
        pred = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.1, 3.1])
        result = mae(pred, actual)
        assert abs(result - 0.1) < 1e-5

    def test_rmse(self):
        pred = np.array([0.0, 0.0])
        actual = np.array([1.0, 1.0])
        result = rmse(pred, actual)
        assert abs(result - 1.0) < 1e-5

    def test_mape_nonzero(self):
        pred = np.array([90.0, 110.0])
        actual = np.array([100.0, 100.0])
        result = mape(pred, actual)
        assert 0.0 < result < 20.0

    def test_crps(self):
        pred_samples = np.random.randn(50, 20)
        actual = np.random.randn(50)
        score = crps(pred_samples, actual)
        assert isinstance(score, float)
        assert score >= 0


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def _make_result(self) -> EvalResult:
        result = EvalResult(model_name="test")
        result.metrics = {
            "MAE": {"sensor": {1: 0.5, 6: 0.8}},
            "RMSE": {"sensor": {1: 0.6, 6: 0.9}},
        }
        return result

    def test_summary_table(self):
        result = self._make_result()
        s = result.summary_table()
        assert "MAE" in s
        assert "sensor" in s

    def test_to_dataframe(self):
        pd = pytest.importorskip("pandas")
        result = self._make_result()
        df = result.to_dataframe()
        assert len(df) == 4
        assert "metric" in df.columns
        assert "value" in df.columns

    def test_repr_contains_model_name(self):
        result = self._make_result()
        r = repr(result)
        assert "test" in r


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def test_mae_loss(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.1, 2.1, 3.1])
        loss = mae_loss(pred, tgt)
        assert abs(float(loss) - 0.1) < 1e-5

    def test_rmse_loss(self):
        pred = torch.zeros(4)
        tgt = torch.ones(4)
        loss = rmse_loss(pred, tgt)
        assert abs(float(loss) - 1.0) < 1e-5

    def test_mape_loss(self):
        pred = torch.tensor([90.0])
        tgt = torch.tensor([100.0])
        loss = mape_loss(pred, tgt)
        assert abs(float(loss) - 0.10) < 1e-3

    def test_quantile_loss_shape(self):
        pred = torch.randn(10, 3)     # 3 quantiles, 1 target dim
        tgt = torch.randn(10, 1)
        loss = quantile_loss(pred, tgt, quantiles=[0.1, 0.5, 0.9], target_dim=1)
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_multi_horizon_loss(self):
        loss_fn = MultiHorizonLoss(loss_name="mae")
        preds = {"sensor": {1: torch.ones(5, 1), 6: torch.ones(5, 1) * 2}}
        tgts = {"sensor": {1: torch.ones(5, 1), 6: torch.ones(5, 1) * 2}}
        loss = loss_fn(preds, tgts)
        assert float(loss) < 1e-5

    def test_multi_horizon_loss_missing_target(self):
        loss_fn = MultiHorizonLoss(loss_name="mae")
        preds = {"sensor": {1: torch.ones(5, 1)}}
        tgts = {}   # no targets
        loss = loss_fn(preds, tgts)
        assert float(loss) == 0.0


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------

class TestCurriculumManager:
    def test_initial_horizon_is_shortest(self):
        schedule = CurriculumSchedule(warmup_epochs=0, horizon_ramp_start=0)
        mgr = CurriculumManager(schedule, total_epochs=50, all_horizons=[1, 6, 24])
        cfg = mgr.step(0)
        assert cfg["active_horizons"][0] == 1

    def test_all_horizons_active_at_end(self):
        schedule = CurriculumSchedule(warmup_epochs=0, horizon_ramp_start=0)
        mgr = CurriculumManager(schedule, total_epochs=10, all_horizons=[1, 6, 24])
        cfg = mgr.step(9)
        assert 24 in cfg["active_horizons"]

    def test_disabled_curriculum(self):
        schedule = CurriculumSchedule(enabled=False)
        mgr = CurriculumManager(schedule, total_epochs=20, all_horizons=[1, 6])
        cfg = mgr.step(0)
        assert cfg["active_horizons"] == [1, 6]

    def test_mask_ratio_increases(self):
        schedule = CurriculumSchedule(
            warmup_epochs=0,
            mask_ratio_start=0.0,
            mask_ratio_end=0.3,
            enabled=True,
        )
        mgr = CurriculumManager(schedule, total_epochs=100, all_horizons=[1])
        early = mgr.step(0)["mask_ratio"]
        late = mgr.step(99)["mask_ratio"]
        assert late > early


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    def test_save_and_load_best(self, tmp_path):
        import torch.nn as nn
        model = nn.Linear(4, 2)
        mgr = CheckpointManager(str(tmp_path / "ckpts"), save_top_k=1)

        mgr.save(model, epoch=0, metrics={"val_loss": 0.5})
        mgr.save(model, epoch=1, metrics={"val_loss": 0.3})
        mgr.save(model, epoch=2, metrics={"val_loss": 0.4})

        assert mgr.best_score == 0.3

        # Corrupt weights, then reload
        for p in model.parameters():
            p.data.fill_(99.0)
        mgr.load_best(model)
        assert not (model.weight.data == 99.0).all()

    def test_last_checkpoint_always_saved(self, tmp_path):
        import torch.nn as nn
        model = nn.Linear(2, 1)
        mgr = CheckpointManager(str(tmp_path / "ckpts2"))
        mgr.save(model, epoch=0, metrics={"val_loss": 1.0})
        last = tmp_path / "ckpts2" / "last.pt"
        assert last.exists()

    def test_no_saved_raises_on_load(self, tmp_path):
        import torch.nn as nn
        model = nn.Linear(2, 1)
        mgr = CheckpointManager(str(tmp_path / "empty"))
        from graphoracle.utils.exceptions import CheckpointError
        with pytest.raises(CheckpointError):
            mgr.load_best(model)


# ---------------------------------------------------------------------------
# TrainingHistory
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def test_record(self):
        h = TrainingHistory()
        h.record(0, 1.0, 0.9)
        h.record(1, 0.8, 0.75)
        assert h.best_val_loss() == 0.75

    def test_repr(self):
        h = TrainingHistory()
        h.record(0, 1.0, 0.9)
        r = repr(h)
        assert "TrainingHistory" in r


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_unsupported_metric_raises(self):
        oracle = GraphOracle(model="lstm", horizons=[1])
        with pytest.raises(ValueError):
            Evaluator(oracle, metrics=["INVALID"])

    def test_run_returns_eval_result(self, simple_schema, simple_graph):
        oracle = GraphOracle(model="lstm", horizons=[1])
        oracle.fit(simple_graph, config=TrainingConfig(epochs=1))
        evaluator = Evaluator(oracle, metrics=["MAE", "RMSE"])
        result = evaluator.run(simple_graph)
        assert isinstance(result, EvalResult)
        assert "MAE" in result.metrics
        assert "sensor" in result.metrics["MAE"]
