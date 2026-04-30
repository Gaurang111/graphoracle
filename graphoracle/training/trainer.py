"""Training loop, config, and history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from graphoracle.graph.heterogeneous import HeterogeneousTemporalGraph
from graphoracle.graph.temporal import TEMPORAL_FEATURE_DIM, build_temporal_tensor
from graphoracle.models.base import BaseForecastModel
from graphoracle.training.checkpointing import CheckpointManager
from graphoracle.training.curriculum import CurriculumManager, CurriculumSchedule
from graphoracle.training.loss import MultiHorizonLoss
from graphoracle.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    epochs: int = 100
    learning_rate: float = 1e-3
    loss: str = "mae"
    scheduler: str = "cosine"
    early_stopping_patience: int = 15
    use_curriculum: bool = False
    checkpoint_dir: str = ""
    device: str = "cpu"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0


class TrainingHistory:
    """Records per-epoch training and validation losses."""

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self._epochs: list[int] = []

    def record(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self._epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def best_val_loss(self) -> float:
        if not self.val_losses:
            return float("inf")
        return min(self.val_losses)

    def plot(self) -> None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(self._epochs, self.train_losses, label="train")
            ax.plot(self._epochs, self.val_losses, label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    def __repr__(self) -> str:
        if self.val_losses:
            return (
                f"TrainingHistory(epochs={len(self._epochs)}, "
                f"best_val={self.best_val_loss():.4f})"
            )
        return f"TrainingHistory(epochs=0)"


class Trainer:
    """
    Full training loop for BaseForecastModel subclasses.

    Handles device placement, optimiser, learning-rate scheduling,
    curriculum learning, checkpointing, and early stopping.
    """

    def __init__(self, model: BaseForecastModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

    def fit(
        self,
        graph: HeterogeneousTemporalGraph,
        val_graph: HeterogeneousTemporalGraph | None = None,
    ) -> TrainingHistory:
        cfg = self.config
        model = self.model
        device = torch.device(cfg.device)
        model.to(device)
        model.on_fit_start(graph)

        optimiser = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        if cfg.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
            )
        else:
            sched = torch.optim.lr_scheduler.StepLR(optimiser, step_size=30, gamma=0.5)

        loss_fn = MultiHorizonLoss(cfg.loss)
        history = TrainingHistory()

        ckpt_mgr: CheckpointManager | None = None
        if cfg.checkpoint_dir:
            ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

        curriculum: CurriculumManager | None = None
        if cfg.use_curriculum:
            schedule = CurriculumSchedule(enabled=True)
            curriculum = CurriculumManager(schedule, cfg.epochs, model.horizons)

        no_improve = 0
        best_val = float("inf")

        # Pre-build static tensors once
        node_features, edge_index, temporal_enc, targets = self._prepare(graph, device)

        for epoch in range(cfg.epochs):
            # Select active horizons for this epoch
            active_horizons = model.horizons
            if curriculum is not None:
                c_cfg = curriculum.step(epoch)
                active_horizons = c_cfg["active_horizons"]

            # --- Training step ---
            model.train()
            optimiser.zero_grad()
            preds = model(graph, node_features, edge_index, temporal_enc)

            # Filter to active horizons
            preds_filtered = {
                nt: {h: t for h, t in h_dict.items() if h in active_horizons}
                for nt, h_dict in preds.items()
            }
            tgt_filtered = {
                nt: {h: t for h, t in h_dict.items() if h in active_horizons}
                for nt, h_dict in targets.items()
            }

            train_loss = loss_fn(preds_filtered, tgt_filtered)
            if train_loss.requires_grad:
                train_loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimiser.step()
            sched.step()

            # --- Validation step ---
            val_loss_val: float
            if val_graph is not None:
                val_loss_val = self._eval(val_graph, model, loss_fn, device)
            else:
                val_loss_val = float(train_loss.item())

            history.record(epoch, float(train_loss.item()), val_loss_val)

            if ckpt_mgr is not None:
                ckpt_mgr.save(model, epoch, {"val_loss": val_loss_val})

            # Early stopping
            if val_loss_val < best_val:
                best_val = val_loss_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                log.info(f"Early stopping at epoch {epoch}.")
                break

        return history

    # ------------------------------------------------------------------

    def _prepare(
        self,
        graph: HeterogeneousTemporalGraph,
        device: torch.device,
    ) -> tuple[dict, dict, Tensor, dict]:
        """Build tensors and targets from the graph (once per fit call)."""
        node_features: dict[str, Tensor] = {}
        for nt in graph.schema.node_types:
            feat = graph.get_node_features(nt.name)
            if feat.numel() > 0:
                node_features[nt.name] = feat.to(device)

        edge_index: dict[str, Tensor] = {}
        for et in graph.schema.edge_types:
            ei = graph.get_edge_index(et.name)
            edge_index[et.name] = ei.to(device)

        ts = graph.timestamps
        if ts:
            temporal_enc = build_temporal_tensor(ts, TEMPORAL_FEATURE_DIM).to(device)
        else:
            T = max((v.shape[1] for v in node_features.values()), default=1)
            temporal_enc = torch.zeros(T, TEMPORAL_FEATURE_DIM, device=device)

        targets: dict[str, dict[int, Tensor]] = {}
        for nt in graph.schema.forecast_node_types:
            if nt.name not in node_features:
                continue
            feat = node_features[nt.name]  # (N, T, F)
            T = feat.shape[1]
            tgt_idx = [nt.features.index(t) for t in nt.targets if t in nt.features]
            if not tgt_idx:
                continue
            idx_t = torch.tensor(tgt_idx, device=device)
            h_dict: dict[int, Tensor] = {}
            for h in self.model.horizons:
                t = min(T - 1, h)
                h_dict[h] = feat[:, t, :][:, idx_t]
            targets[nt.name] = h_dict

        return node_features, edge_index, temporal_enc, targets

    @torch.no_grad()
    def _eval(
        self,
        graph: HeterogeneousTemporalGraph,
        model: BaseForecastModel,
        loss_fn: MultiHorizonLoss,
        device: torch.device,
    ) -> float:
        model.eval()
        node_features, edge_index, temporal_enc, targets = self._prepare(graph, device)
        preds = model(graph, node_features, edge_index, temporal_enc)
        loss = loss_fn(preds, targets)
        model.train()
        return float(loss.item())
