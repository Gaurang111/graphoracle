"""Curriculum learning schedule for multi-horizon training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumSchedule:
    """
    Configuration for curriculum learning.

    Attributes
    ----------
    enabled           : if False, all horizons are active from epoch 0
    warmup_epochs     : epochs before any curriculum ramp starts
    horizon_ramp_start: epoch at which horizon ramp begins
    mask_ratio_start  : initial fraction of node features to mask
    mask_ratio_end    : final fraction to mask at the end of training
    """

    enabled: bool = True
    warmup_epochs: int = 5
    horizon_ramp_start: int = 0
    mask_ratio_start: float = 0.0
    mask_ratio_end: float = 0.0


class CurriculumManager:
    """
    Manages the curriculum schedule over the course of training.

    Usage
    -----
    mgr = CurriculumManager(schedule, total_epochs=100, all_horizons=[1, 6, 24])
    for epoch in range(total_epochs):
        cfg = mgr.step(epoch)
        # cfg["active_horizons"] — which horizons to include this epoch
        # cfg["mask_ratio"]     — fraction of features to mask
    """

    def __init__(
        self,
        schedule: CurriculumSchedule,
        total_epochs: int,
        all_horizons: list[int],
    ) -> None:
        self.schedule = schedule
        self.total_epochs = max(total_epochs, 1)
        self.all_horizons = sorted(all_horizons)

    def step(self, epoch: int) -> dict[str, Any]:
        s = self.schedule

        if not s.enabled:
            return {
                "active_horizons": list(self.all_horizons),
                "mask_ratio": s.mask_ratio_end,
            }

        n = len(self.all_horizons)
        ramp_start = max(s.horizon_ramp_start, s.warmup_epochs)
        ramp_len = max(self.total_epochs - ramp_start, 1)
        progress = max(0.0, (epoch - ramp_start) / ramp_len)

        # Ramp from 1 horizon to all horizons
        active_count = max(1, round(1 + (n - 1) * progress))
        active_horizons = self.all_horizons[:active_count]

        # Mask ratio ramp
        mask_progress = epoch / max(self.total_epochs - 1, 1)
        mask_ratio = (
            s.mask_ratio_start
            + (s.mask_ratio_end - s.mask_ratio_start) * mask_progress
        )

        return {
            "active_horizons": active_horizons,
            "mask_ratio": float(mask_ratio),
        }
