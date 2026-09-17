"""The amputation campaign freezes its recording list at submit time."""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "slurm" / "amputation_array.sh"


def _data_root(tmp_path, recordings, with_bouts=True):
    root = tmp_path / "amputation"
    for rec in recordings:
        d = root / rec
        d.mkdir(parents=True)
        (d / "data3D.csv").write_text("A,A,A,A\nx,y,z,confidence\n")
        if with_bouts:
            (d / "running_bouts_summary.csv").write_text(
                "bout,start_frame,end_frame,n_frames\n1,0,9,10\n"
            )
    return root


def _run(*args, cwd=None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd or REPO
    )


def test_dry_run_writes_a_manifest_of_every_usable_recording(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07", "2026_07_06_17_11_20"])
    out = _run(
        "--dry-run",
        "--run-name",
        "ik_v1",
        "--data-root",
        str(root),
        "--manifest-dir",
        str(tmp_path),
    )
    assert out.returncode == 0, out.stderr
    manifest = next(tmp_path.glob("amputation_*.manifest"))
    assert manifest.read_text().split() == [
        "2026_07_06_16_55_07",
        "2026_07_06_17_11_20",
    ]
    assert "--array=0-1" in out.stdout


def test_recording_without_a_bout_table_is_skipped_by_name(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    bare = root / "2026_07_06_17_11_20"
    bare.mkdir()
    (bare / "data3D.csv").write_text("A,A,A,A\nx,y,z,confidence\n")

    out = _run(
        "--dry-run",
        "--run-name",
        "ik_v1",
        "--data-root",
        str(root),
        "--manifest-dir",
        str(tmp_path),
    )
    assert out.returncode == 0, out.stderr
    assert "2026_07_06_17_11_20" in out.stderr
    manifest = next(tmp_path.glob("amputation_*.manifest"))
    assert manifest.read_text().split() == ["2026_07_06_16_55_07"]


def test_run_name_is_required(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    out = _run("--dry-run", "--data-root", str(root))
    assert out.returncode != 0
    assert "run-name" in out.stderr


def test_no_usable_recording_refuses(tmp_path):
    root = tmp_path / "amputation"
    root.mkdir()
    out = _run(
        "--dry-run",
        "--run-name",
        "ik_v1",
        "--data-root",
        str(root),
        "--manifest-dir",
        str(tmp_path),
    )
    assert out.returncode != 0
    assert "no recording" in out.stderr


def test_dry_run_command_quotes_the_recording_id(tmp_path):
    root = _data_root(tmp_path, ["2026_07_06_16_55_07"])
    out = _run(
        "--dry-run",
        "--run-name",
        "ik_v1",
        "--data-root",
        str(root),
        "--manifest-dir",
        str(tmp_path),
    )
    assert "recording.id='" in out.stdout
    assert "stages=[bouts,ingest3d,preprocess,ik,postprocess,collect]" in out.stdout
    assert "ik=amputation" in out.stdout
