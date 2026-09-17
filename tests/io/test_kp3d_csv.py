"""Reading a two-row-header whole-recording 3D keypoint CSV."""

import numpy as np
import pytest

from tracking.io.bouts import BoutSpec
from tracking.io.kp3d_csv import read_bout_kp3d, read_kp3d_header
from tracking.io.names import Order, OrderMismatch

KPS = ["Scutellum", "Abd_tip", "T1R_FeTi"]


def _write_csv(path, n_frames, kps=KPS):
    header1 = ",".join(k for k in kps for _ in range(4))
    header2 = ",".join(["x", "y", "z", "confidence"] * len(kps))
    lines = [header1, header2]
    for t in range(n_frames):
        row = []
        for k_i in range(len(kps)):
            base = t * 100 + k_i * 10
            row += [f"{base}.0", f"{base + 1}.0", f"{base + 2}.0", "0.9"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _bout(idx, start, end):
    return BoutSpec(
        idx=idx,
        start_frame=start,
        end_frame=end,
        n_frames=end - start + 1,
        source="summary",
    )


def test_header_names_are_read_in_column_order(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 5)
    assert read_kp3d_header(csv) == KPS


def test_header_rejects_unexpected_coordinate_row(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("A,A,A,A\nx,y,z,score\n1,2,3,4\n")
    with pytest.raises(ValueError, match="expected"):
        read_kp3d_header(csv)


def test_header_rejects_mismatched_name_block(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("A,A,B,A\nx,y,z,confidence\n1,2,3,4\n")
    with pytest.raises(ValueError, match="one keypoint name"):
        read_kp3d_header(csv)


def test_bout_slices_are_exact_and_permuted(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 50)
    order = Order(["Abd_tip", "T1R_FeTi", "Scutellum"])
    out = read_bout_kp3d(csv, [_bout(1, 10, 12), _bout(2, 40, 40)], kp_order=order, chunksize=7)

    kp3d, conf = out[1]
    assert kp3d.shape == (3, 3, 3)
    assert conf.shape == (3, 3)
    # frame 10, "Abd_tip" is kps index 1 -> base = 10*100 + 1*10 = 1010
    np.testing.assert_allclose(kp3d[0, 0], [1010.0, 1011.0, 1012.0])
    # frame 10, "Scutellum" is kps index 0 -> base = 1000
    np.testing.assert_allclose(kp3d[0, 2], [1000.0, 1001.0, 1002.0])
    np.testing.assert_allclose(conf, 0.9)

    assert out[2][0].shape == (1, 3, 3)
    np.testing.assert_allclose(out[2][0][0, 2], [4000.0, 4001.0, 4002.0])


def test_chunk_boundary_does_not_split_a_bout(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 50)
    order = Order(KPS)
    whole = read_bout_kp3d(csv, [_bout(1, 5, 30)], kp_order=order, chunksize=1000)[1][0]
    split = read_bout_kp3d(csv, [_bout(1, 5, 30)], kp_order=order, chunksize=4)[1][0]
    np.testing.assert_array_equal(whole, split)


def test_bout_past_end_of_file_refuses(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 20)
    with pytest.raises(ValueError, match="yielded"):
        read_bout_kp3d(csv, [_bout(1, 15, 25)], kp_order=Order(KPS))


def test_kp_order_naming_an_absent_keypoint_refuses(tmp_path):
    csv = _write_csv(tmp_path / "data3D.csv", 5)
    with pytest.raises(OrderMismatch):
        read_bout_kp3d(csv, [_bout(1, 0, 2)], kp_order=Order(["Scutellum", "T1L_TaTip"]))
