"""A finished run root describes itself, whatever assay produced it."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from tracking.viz.runroot import (
    discover_bouts,
    discover_flies,
    load_bout_observed,
    load_bout_qpos,
    load_run_anatomy,
    run_kp_names,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# v1's KP_NAMES minus the six distal T1L markers -- an amputation run's order.
AMP_44 = [
    k
    for k in OmegaConf.load(CONFIGS / "anatomy/v1.yaml").model.KP_NAMES
    if k not in ("T1L_Tro", "T1L_FeTi", "T1L_TiTa", "T1L_TaT1", "T1L_TaT3", "T1L_TaTip")
]


def _write_h5(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)


def _run_root(tmp_path, *, flies, bouts, kp_names, nq=93, n_frames=7):
    """A synthetic run root: `flies` x `bouts` outputs.h5 plus offsets files."""
    root = tmp_path / "run"
    names = np.array(kp_names, dtype="S16")
    for fly in flies:
        _write_h5(
            root / f"offsets_fly{fly}.h5",
            offsets=np.zeros((len(kp_names), 3), np.float32),
            kp_names=names,
        )
        for b in bouts:
            _write_h5(
                root / "bouts" / f"bout_{b:05d}" / f"fly{fly}" / "outputs.h5",
                qpos=np.zeros((n_frames, nq), np.float32),
                kp_names=names,
            )
    return root


def test_single_animal_run_reports_one_fly(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[1, 2], kp_names=AMP_44)
    assert discover_flies(root) == [0]


def test_courtship_run_reports_two_flies(tmp_path):
    root = _run_root(tmp_path, flies=[0, 1], bouts=[1], kp_names=["Scutellum"] * 50)
    assert discover_flies(root) == [0, 1]


def test_a_run_with_no_bouts_refuses_rather_than_returning_empty(tmp_path):
    (tmp_path / "run" / "bouts").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no bouts"):
        discover_flies(tmp_path / "run")


def test_bouts_are_discovered_in_ascending_numeric_order(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[2, 10, 1], kp_names=AMP_44)
    assert discover_bouts(root) == [1, 2, 10]


def test_bouts_can_be_narrowed_to_one_fly(tmp_path):
    root = _run_root(tmp_path, flies=[0, 1], bouts=[1, 2], kp_names=AMP_44)
    # fly1 solved only bout 1
    (root / "bouts" / "bout_00002" / "fly1" / "outputs.h5").unlink()
    assert discover_bouts(root, fly=0) == [1, 2]
    assert discover_bouts(root, fly=1) == [1]


def test_kp_names_come_from_the_run_not_from_the_anatomy_default(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[1], kp_names=AMP_44)
    names = run_kp_names(root, fly=0)
    assert len(names) == 44
    assert "T1L_ThxCx" in names
    assert "T1L_TaTip" not in names


def test_load_bout_qpos_returns_the_solved_pose(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[3], kp_names=AMP_44, n_frames=11)
    q = load_bout_qpos(root, bout=3, fly=0)
    assert q.shape == (11, 93)
    assert q.dtype == np.float64


def test_load_bout_qpos_names_the_missing_file(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[1], kp_names=AMP_44)
    with pytest.raises(FileNotFoundError, match="bout_00099"):
        load_bout_qpos(root, bout=99, fly=0)


def test_run_anatomy_is_filtered_to_the_run_and_carries_its_offsets(tmp_path):
    """The load-bearing case: a 44-keypoint run must not get v1's 50."""
    root = _run_root(tmp_path, flies=[0], bouts=[1], kp_names=AMP_44)
    offsets = np.arange(44 * 3, dtype=np.float32).reshape(44, 3) * 1e-4
    _write_h5(
        root / "offsets_fly0.h5",
        offsets=offsets,
        kp_names=np.array(AMP_44, dtype="S16"),
    )

    anatomy = load_run_anatomy(root, fly=0, anatomy_cfg=CONFIGS / "anatomy/v1.yaml")

    assert len(anatomy.kp_order) == 44
    assert "T1L_TaTip" not in anatomy.kp_order
    # the offsets actually reached the model's marker sites
    np.testing.assert_allclose(
        anatomy.mj_model.site_pos[anatomy.site_idxs], offsets.astype(np.float64), rtol=1e-6
    )


def test_observed_keypoints_come_from_stac_ik_in_model_space(tmp_path):
    """The overlay must use stac_ik.h5:kp_data, never outputs.h5:kp3d_mm.

    kp3d_mm is millimetres and runs ~90x larger than the model; drawing it into
    a MuJoCo scene puts the cloud nowhere near the body and raises nothing.
    """
    root = _run_root(tmp_path, flies=[0], bouts=[4], kp_names=AMP_44, n_frames=5)
    model_space = np.linspace(0.0, 0.35, 5 * 44 * 3, dtype=np.float32).reshape(5, 44 * 3)
    _write_h5(
        root / "bouts" / "bout_00004" / "fly0" / "stac_ik.h5",
        kp_data=model_space,
        kp_names=np.array(AMP_44, dtype="S16"),
    )
    # outputs.h5 additionally carries a millimetre array; it must NOT be chosen.
    with h5py.File(root / "bouts" / "bout_00004" / "fly0" / "outputs.h5", "a") as f:
        f.create_dataset("kp3d_mm", data=model_space.reshape(5, 44, 3) * 93.0)

    obs = load_bout_observed(root, bout=4, fly=0)
    assert obs.shape == (5, 44, 3)
    assert obs.max() <= 0.4, "observed overlay is not in model space"


def test_observed_is_none_when_the_run_has_no_stac_file(tmp_path):
    root = _run_root(tmp_path, flies=[0], bouts=[4], kp_names=AMP_44)
    assert load_bout_observed(root, bout=4, fly=0) is None
