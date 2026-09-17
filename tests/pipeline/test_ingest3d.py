"""Slicing a whole-recording 3D CSV into the per-bout artifacts the pipeline reads."""

import numpy as np

from tracking.io.bouts import BoutSpec
from tracking.io.names import Order
from tracking.pipeline.ingest_stages import ingest3d_recording, sex_json_for
from tracking.pipeline.plan import resolve
from tracking.pipeline.stages import stage_by_name
from tracking.qc.posture import fly_sex_label

KPS = ["Scutellum", "Abd_tip", "T1R_FeTi"]


def _csv(path, n_frames):
    rows = [
        ",".join(k for k in KPS for _ in range(4)),
        ",".join(["x", "y", "z", "confidence"] * len(KPS)),
    ]
    for t in range(n_frames):
        row = []
        for k_i in range(len(KPS)):
            base = t * 100 + k_i * 10
            row += [f"{base}.0", f"{base + 1}.0", f"{base + 2}.0", "0.9"]
        rows.append(",".join(row))
    path.write_text("\n".join(rows) + "\n")
    return path


class _Spec:
    def __init__(self, kp3d_csv, index_csv=None, name="rec1"):
        self.kp3d_csv = kp3d_csv
        self.index_csv = index_csv
        self.name = name


def test_ingest_writes_one_kp3d_per_bout(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    bouts = [BoutSpec(1, 10, 12, 3, "summary"), BoutSpec(2, 40, 41, 2, "summary")]
    run_root = tmp_path / "run"

    ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)

    for idx, n in ((1, 3), (2, 2)):
        path = run_root / "bouts" / f"bout_{idx:05d}" / "fly0" / "kp3d.npz"
        assert path.exists()
        with np.load(path, allow_pickle=True) as z:
            assert z["kp3d"].shape == (n, 3, 3)
            assert z["conf3d"].shape == (n, 3)
            assert [str(s) for s in z["kp_names"]] == KPS


def test_ingest_is_resumable_and_skips_written_bouts(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    bouts = [BoutSpec(1, 10, 12, 3, "summary")]
    run_root = tmp_path / "run"

    first = ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)
    second = ingest3d_recording(run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts)
    assert first["n_written"] == 1
    assert second["n_written"] == 0
    assert second["n_skipped"] == 1

    forced = ingest3d_recording(
        run_root, spec=_Spec(csv), kp_order=Order(KPS), bouts=bouts, force=True
    )
    assert forced["n_written"] == 1


def test_sex_column_is_found_by_header_not_first_matching_cell(tmp_path):
    """A flyID of "F" must not be mistaken for the sex column."""
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\nF,m,recY,T1L\n")
    assert sex_json_for(index, "recY")["sex_by_fly"] == {"0": "male"}


def test_sex_json_read_from_the_cohort_index(tmp_path):
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\n1,f,rec1,T1L\n2,m,rec2,T1L\n")
    assert sex_json_for(index, "rec1")["sex_by_fly"] == {"0": "female"}
    assert sex_json_for(index, "rec2")["sex_by_fly"] == {"0": "male"}
    assert sex_json_for(index, "absent") is None


def test_fly_sex_label_understands_sex_by_fly():
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {"0": "female"}}, 0) == "female"
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {"0": "male"}}, 0) == "male"
    # the courtship form is untouched
    assert fly_sex_label({"identity": "sex", "male_fly": 1}, 1) == "male"
    assert fly_sex_label({"identity": "sex", "male_fly": 1}, 0) == "female"
    assert fly_sex_label({"identity": "sex", "sex_by_fly": {}}, 0) == "unknown"


def test_ingest_writes_sex_json_beside_each_bout(tmp_path):
    csv = _csv(tmp_path / "data3D.csv", 50)
    index = tmp_path / "index.csv"
    index.write_text("flyID,sex,,amp\n1,f,rec1,T1L\n")
    run_root = tmp_path / "run"

    ingest3d_recording(
        run_root,
        spec=_Spec(csv, index_csv=index),
        kp_order=Order(KPS),
        bouts=[BoutSpec(1, 10, 12, 3, "summary")],
    )
    import json

    written = json.loads((run_root / "bouts" / "bout_00001" / "sex.json").read_text())
    assert fly_sex_label(written, 0) == "female"


def test_ingest3d_is_registered_and_plannable(tmp_path):
    from omegaconf import OmegaConf

    assert stage_by_name("ingest3d").scope == "recording"
    cfg = OmegaConf.create(
        {"run": {"root": str(tmp_path)}, "recording": {"name": "rec1", "num_animals": 1}}
    )
    works = resolve(cfg, stages=["ingest3d"], bout_ids=[1])
    assert [w.stage for w in works] == ["ingest3d"]
    assert works[0].skip_reason is None
