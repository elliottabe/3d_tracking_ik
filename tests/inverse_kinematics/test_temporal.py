"""Temporal IK: windowing never bridges a gap, and coupling beats per-frame on noise."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401
from tracking.inverse_kinematics.perframe import solve_bout
from tracking.inverse_kinematics.solver import temporal_windows
from tracking.run import _load_anatomy

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("window,overlap", [(5, 0), (5, 2), (8, 3), (40, 10)])
def test_windows_own_every_frame_once_and_stay_inside_runs(window, overlap):
    frame_idx = np.r_[0:23, 30:31, 40:52]  # runs of 23, 1, 12
    run_of = np.searchsorted([23, 24], np.arange(frame_idx.size), side="right")
    owned = np.zeros(frame_idx.size, int)
    for pos, keep in temporal_windows(frame_idx, window, overlap):
        assert pos.size <= window
        assert np.all(np.diff(frame_idx[pos]) == 1)  # contiguous frames only
        assert len(set(run_of[pos])) == 1
        owned[pos[keep]] += 1
    assert np.all(owned == 1)


@pytest.fixture(scope="module")
def anatomy():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="pipeline", overrides=["recording=session0"])
    return _load_anatomy(cfg)


def test_temporal_tracks_truth_better_than_per_frame_on_noisy_abdomen(anatomy):
    mj = anatomy.mj_model
    site_idxs = np.asarray(anatomy.site_idxs)
    offsets = np.asarray(mj.site_pos[site_idxs], np.float64)
    names = list(anatomy.names_qpos)
    abd = [i for i, n in enumerate(names) if n.startswith("abdomen")]

    # Smooth ground truth: slow abdomen bend, everything else at rest.
    T = 32
    q_true = np.tile(np.asarray(mj.qpos0, np.float64), (T, 1))
    q_true[:, abd] += 0.15 * np.sin(np.linspace(0, np.pi, T))[:, None]
    d = mujoco.MjData(mj)
    truth = np.empty((T, site_idxs.size, 3))
    for t in range(T):
        d.qpos[:] = q_true[t]
        mujoco.mj_kinematics(mj, d)
        truth[t] = d.site_xpos[site_idxs]
    rng = np.random.default_rng(0)
    kp = truth + rng.normal(0, 0.003, truth.shape)

    base = dict(dt=0.00125, n_iter=100, refine_wings=False, save_candidates=False)
    per = solve_bout(anatomy, kp, offsets=offsets, per_frame_cfg=base)
    tmp = solve_bout(
        anatomy,
        kp,
        offsets=offsets,
        per_frame_cfg={**base, "temporal": {"weight": 0.1, "window": 16, "overlap": 4}},
    )
    assert tmp.candidate_names == ("temporal",)

    ia = [anatomy.kp_order.index(n) for n in ("Abd_A4", "Abd_tip")]

    def err(r):
        return np.linalg.norm(r.marker_sites[:, ia] - truth[:, ia], axis=-1).mean()

    def jitter(r):
        return np.abs(np.diff(r.qpos[:, abd], axis=0)).sum(1).mean()

    assert err(tmp) < 0.9 * err(per)  # measured 0.85
    assert jitter(tmp) < 0.5 * jitter(per)
