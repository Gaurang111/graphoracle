"""Checkpoint manager: save top-k models during training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from graphoracle.utils.exceptions import CheckpointError


@dataclass
class CheckpointState:
    epoch: int
    metrics: dict[str, float]
    path: str


class CheckpointManager:
    """
    Save model checkpoints during training, keeping only the best *save_top_k*.

    Parameters
    ----------
    checkpoint_dir : directory to write checkpoint files
    save_top_k     : keep this many best checkpoints (based on val_loss)
    monitor        : metric key to optimise (lower = better)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_top_k: int = 3,
        monitor: str = "val_loss",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor

        self._checkpoints: list[CheckpointState] = []
        self.best_score: float = float("inf")

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save *model* for *epoch* with *metrics*."""
        score = metrics.get(self.monitor, float("inf"))
        ckpt_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # Always save a "last" checkpoint
        last_path = self.checkpoint_dir / "last.pt"
        shutil.copy2(ckpt_path, last_path)

        self._checkpoints.append(CheckpointState(epoch, metrics, str(ckpt_path)))

        if score < self.best_score:
            self.best_score = score
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy2(ckpt_path, best_path)

        # Prune excess checkpoints (keep best.pt, last.pt, and top-k by score)
        self._prune()

    def load_best(self, model: nn.Module) -> None:
        """Load the best checkpoint into *model* in-place."""
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            # Fall back to last
            last_path = self.checkpoint_dir / "last.pt"
            if not last_path.exists():
                raise CheckpointError(
                    f"No checkpoint found in '{self.checkpoint_dir}'. "
                    "Call save() at least once before load_best()."
                )
            best_path = last_path
        state = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove epoch_*.pt files beyond the top-k by monitored metric."""
        sorted_ckpts = sorted(
            self._checkpoints,
            key=lambda c: c.metrics.get(self.monitor, float("inf")),
        )
        keep = {c.path for c in sorted_ckpts[: self.save_top_k]}
        for c in self._checkpoints:
            p = Path(c.path)
            if c.path not in keep and p.exists():
                p.unlink(missing_ok=True)
