"""Scoring a bout that never had 2D to reproject against."""

import json

import numpy as np

from tracking.detector.coarse import FloorPlane
from tracking.io.names import Order
from tracking.qc.bout import bout_qc
from tracking.qc.session import _row_from_qc, write_scorecard

KPS = ["Scutellum", "WingL_base", "WingR_base", "Antenna_Base", "Abd_tip", "Abd_A4"]


def _pose(n=40):
    rng = np.random.default_rng(0)
    base = np.array([[0, 0, 5.0], [-2, 1, 5], [-2, -1, 5], [4, 0, 5], [-6, 0, 5], [-4, 0, 5]])
    return base[None] + rng.normal(0, 0.01, size=(n, len(KPS), 3))


def _floor():
    return FloorPlane(np.array([0.0, 0.0, 1.0]), 0.0, "level", 0.0)


def test_rigless_qc_omits_reproj_and_loo():
    kp = _pose()
    report = bout_qc(
        kp2d=None,
        conf2d=None,
        kp3d_raw=kp,
        conf3d=np.full(kp.shape[:2], 0.9),
        fitted_world=kp,
        kp_order=Order(KPS),
        rig=None,
        floor=_floor(),
        sex="female",
    )
    assert "reproj" not in report
    assert "loo" not in report
    assert report["structural_ok"] is True
    assert "posture" in report
    assert "coverage" in report
    assert "invariants" in report


def test_rigless_posture_flags_nothing_on_residuals():
    kp = _pose()
    report = bout_qc(
        kp2d=None,
        conf2d=None,
        kp3d_raw=kp,
        conf3d=np.full(kp.shape[:2], 0.9),
        fitted_world=kp,
        kp_order=Order(KPS),
        rig=None,
        floor=_floor(),
        sex="female",
    )
    assert report["posture"]["n_flagged"] == 0


def test_scorecard_tolerates_a_missing_reproj_ratio(tmp_path):
    """A REAL rigless report through `_row_from_qc`, not `_blank_row`'s own definition."""
    kp = _pose()
    report = bout_qc(
        kp2d=None,
        conf2d=None,
        kp3d_raw=kp,
        conf3d=np.full(kp.shape[:2], 0.9),
        fitted_world=kp,
        kp_order=Order(KPS),
        rig=None,
        floor=_floor(),
        sex="female",
    )

    row = _row_from_qc("bout_00001", 0, "female", report, "ok")
    assert row.reproj_ratio is None

    scorecard = write_scorecard(
        tmp_path, [row], json_path=tmp_path / "qc.json", md_path=tmp_path / "qc.md"
    )
    assert (tmp_path / "qc.json").exists()
    assert (tmp_path / "qc.md").exists()
    assert scorecard is not None

    on_disk = json.loads((tmp_path / "qc.json").read_text())
    assert on_disk["by_sex"]["female"]["median_reproj_ratio"] is None
