"""MVQ training config: maskless, one unified training root.

>>> MVQTrainConfig().pseudo_weight
0.3
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class MVQTrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    total_steps: int = 30000
    batch_size: int = 32
    backbone_lr_mult: float = 0.1
    grad_clip: float = 1.0
    ema: float = 0.999
    seed: int = 0
    window_lengths: tuple = (1,)
    female_weight: float = 1.0
    balance_alpha: float = 0.5
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 1000
    num_workers: int = 16
    loader_workers: str = "threads"
    pretrained: bool = True
    smoke: bool = False
    val_cohorts: tuple = ("female", "two_fly")
    copy_paste_p: float = 0.0
    copy_paste_opposite_sex_p: float = 0.7
    copy_paste_contact_p: float = 0.3
    copy_paste_contact_sep: tuple = (8.0, 30.0)
    female_host_weight: float = 1.0
    sex_label_overrides: dict = dataclasses.field(default_factory=dict)
    warm_start: str | None = None
    jitter_units: float = 3.0
    pair_deltas: tuple = (1,)
    pseudo_weight: float = 0.3
    """Weight the config expects; ``run.py`` warns if the manifest disagrees (spec §7.1)."""
    negatives_frac: float = 0.05
    """Share of sampled mass reserved for negatives, not proportional to export count."""
    female_host_target: float | None = None
    """Female-host mass to solve for; falls back to ``female_host_weight`` if ``None``."""
    wing_kp_mult: float = 1.0
    """Per-keypoint loss multiplier applied to wing landmarks by name."""
