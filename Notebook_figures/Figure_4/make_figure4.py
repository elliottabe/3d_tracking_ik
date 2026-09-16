#!/usr/bin/env python
"""Build Figure 4 from the courtship IK h5.

    python make_figure4.py          # all ten panels -> fig4.svg / fig4.png

Panels:

    A  video_0..3        camera crop + SAM3 mask tints + vector keypoints
    B  sine_phase        extended vs folded wing, in phase during sine song
    C  wing              both wing tips' z, shaded by song type, pulses marked
    D  wing_phase_polar  pooled L-R wing phase difference during sine
    E  angle_2d          wing-extension angle density, pulse vs sine
    F  pulse_class       Pslow / Pfast pooled waveforms and centroids
    G  zheight           body height: singing vs free-running walking
    H  render_0..3       MuJoCo render of the fitted pair, same frames as A
    I  pitch             male body pitch vs the pitch that would face the female
    J  align_violin      per-bout |body - target| pitch alignment, pooled

Stage 1 (`stage_analysis`) reads the combined h5 and runs the song and
locomotion analysis over every bout-fly. Stage 2 (`stage_assets`) reads
everything that is not in that h5: video, SAM3 masks, calibration, MuJoCo.
Stage 3 (`render_figure`) places the panels and saves. Stages 1 and 2 are
cached, so a cold run is minutes and restyling afterwards is seconds.

What the figure should look like if it is right, written down before it is
generated so that it can disagree:

* B, C and I are one exemplar bout: the two wing traces run anti-phase during
  pulse song and in phase during sine, and panel C's shaded segments line up
  with the pulse events drawn over them.
* H shows the pair on the arena floor with the wing extension panel C reports
  at those same four frames. Wings flat where C says extended, or a body sunk
  through the floor, means the qpos convention is not v1's. A female who
  fragments or vanishes is the arena mesh occluding her; see `RIG_OFFSET`.
* A's keypoints sit on the two flies. Dots on one fly and empty space beside
  the other is the camera-order or mask-slot trap, not a fit failure.
* D-G and J are pooled distributions. A collapsed, emptied or sign-flipped one
  means the fly0/fly1 pairing mismatched the sexes, which is why the sex
  cross-check in `stage_analysis` is fatal.

`fig4_analysis.py` and `fig4_assets.py` are AST extractions of the source
repo's analysis, bodies unchanged. An extraction can silently drop a branch
that fires on only some bouts, and no figure would reveal it, so verify either
file against a dataset whose answer is known rather than trusting the figure
to look right.

Needs a GPU node and MUJOCO_GL=egl for panel H. Never run on a login node.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (str(_HERE), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fig4_analysis as A                        # noqa: E402
import fig4_assets as X                          # noqa: E402
import fig4_layout as L                          # noqa: E402
import fig4_panels as P                          # noqa: E402

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

_DATA = Path("/gscratch/portia/eabe/data/Johnson_lab")

H5 = _DATA / "processed/courtship/ik_output_combined_v1_pose_v2_20260914_Session0+Session1.h5"
PROCESSED_ROOT = _DATA / "processed/courtship"
VIDEO_ROOT = _DATA / "Video_recordings/courtship"
FREE_RUN_H5 = _DATA / "processed/free_running/NewBouts/v1/ik_output_combined_v1_free_running.h5"

#: Pose tree under each recording. Must match the run `H5` was collected from.
POSE_DIR = "pose_v2_20260914"

MODEL_XML = _REPO / "model/fruitfly_v1/fruitfly_v1_free.xml"
FLOOR_XML = _HERE / "floor_happy_house.xml"
ANATOMY_YAML = _REPO / "configs/anatomy/v1.yaml"

#: Pinned by (recording, start frame), never by ordinal position.
EXEMPLAR_RECORDING = "Session0/2025_10_20_13_20_04"
EXEMPLAR_START_FRAME = 446306
EXEMPLAR_CAMERA = "Cam2012630"

FS = 800.0
#: kp3d world units -> DLT world units.
KP_SCALE = 0.1
#: Panel A/H timepoints in FRAMES: 0 / 644 / 1288 / 1931 ms at 800 fps.
VIDEO_FRAMES = (0, 515, 1030, 1545)
CROP_WH = (520, 448)
RENDER_SIZE = 1024
#: Arena nudge in world cm. At (0, 0) the wall occludes a female rearing
#: against it: she renders in fragments, or vanishes, while the male looks
#: perfect. Not cosmetic.
RIG_OFFSET = (0.2, 0.0)
#: Unit-sanity and male/female-slot guard on panel J; see
#: `compute_pitch_alignment_mvq` for the measured margin before retuning.
SCUT_COM_OFFSET_TOL = 15.0
#: `sex.json` authorities panel J accepts besides a human review. Safe only
#: because panel J also cross-checks every bout against `info/sex` and raises
#: on disagreement. Empty this for human-reviewed trees only.
SEX_AUTHORITIES = ("mvq_typed_slots",)

_PULSE_TYPES = ("Pslow", "Pfast")
SEGMENT_DTYPE = np.dtype([("start", "<i8"), ("end", "<i8"), ("type", "S8")])


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def segments_to_array(segments) -> np.ndarray:
    """Song segment dicts -> the structured array the cache stores."""
    segs = list(segments)
    out = np.zeros(len(segs), dtype=SEGMENT_DTYPE)
    for i, s in enumerate(segs):
        out[i] = (int(s["start"]), int(s["end"]),
                  str(s.get("type", "")).encode("utf-8"))
    return out


def write_bundle(path, panels: dict) -> None:
    """`{panel_id: {"data": {...}, "assets": {...}}}` -> one h5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_group("meta")
        pg = f.create_group("panels")
        for pid, pd in panels.items():
            g = pg.create_group(pid)
            g.attrs["type"] = pd.get("type", "")
            dg = g.create_group("data")
            for k, v in (pd.get("data") or {}).items():
                arr = np.asarray(v)
                if arr.dtype == object:
                    arr = np.asarray([str(x) for x in arr.ravel()],
                                     dtype=h5py.string_dtype())
                dg.create_dataset(k, data=arr)
            ag = g.create_group("assets")
            for k, v in (pd.get("assets") or {}).items():
                ag.create_dataset(k, data=np.asarray(v), compression="gzip")


def read_bundle(path) -> dict:
    """Inverse of `write_bundle`."""
    out = {}
    with h5py.File(Path(path), "r") as f:
        for pid, g in f.get("panels", {}).items():
            out[pid] = {
                "type": g.attrs.get("type", ""),
                "data": {k: np.asarray(v) for k, v in g["data"].items()},
                "assets": {k: np.asarray(v) for k, v in g["assets"].items()},
            }
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _as_str(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def model_kp_names() -> list[str]:
    """The 50 MODEL-order keypoint names from `configs/anatomy/v1.yaml`.

    The combined h5 carries no `info/kp_names`, so the order comes from the
    anatomy config and is then checked rather than trusted: every `kp3d.npz`
    ships its own `kp_names`, and `_load_kp3d` compares the two elementwise
    and refuses to load on a mismatch. A count-only match is not enough --
    the same 50 names in a different order reads as a real landmark with
    excellent jitter and confidence.
    """
    import yaml
    cfg = yaml.safe_load(ANATOMY_YAML.read_text())
    names = cfg["model"]["KP_NAMES"]
    if len(names) != 50:
        raise SystemExit(f"{ANATOMY_YAML}: expected 50 KP_NAMES, got {len(names)}")
    return [str(n) for n in names]


def load_combined(h5_path):
    """Load the combined h5 into `(data, info, kp_names, bout_keys, rows)`.

    `rows` is one dict per bout-fly:

        key          bout group name in `data` ('bout_000', ...)
        recording    'Session0/2025_10_20_13_20_04'
        start_frame  absolute recording frame of this bout's frame 0
        fly          0 or 1, the tracker slot
        bout_name    the bout's directory in the processed tree ('bout_00028')
        sex          'male' or 'female'

    Every field is read from a place the file states outright. None is inferred
    from ordinal position, because position here is a permutation of the
    processed tree's bout numbering: deriving the directory from it reads an
    unrelated bout's masks and kp3d while looking entirely plausible. A missing
    field raises rather than falling back to a guess.
    """
    data = A.load(str(h5_path), enable_jax=False)
    info = data.get("info", {}) or {}
    bout_keys = sorted(k for k in data
                       if k.startswith("bout_") and isinstance(data[k], dict))
    n = len(bout_keys)
    kp_names = model_kp_names()

    def _field(name, cast=_as_str):
        vals = info.get(name, [])
        if len(vals) != n:
            raise SystemExit(
                f"{h5_path.name}: info/{name} has {len(vals)} entries for {n} "
                f"bout-flies. Every per-bout field is read from the file "
                f"outright rather than inferred; it cannot proceed without "
                f"this one.")
        return [cast(v) for v in vals]

    start_frames = _field("start_frames", int)
    uid = _field("fly_uid")
    sex = _field("sex")
    grp_names = _field("bout_keys")

    rows = []
    for i, key in enumerate(bout_keys):
        recording, slot = uid[i].rsplit("/", 1)
        rows.append({"key": key, "recording": recording,
                     "start_frame": start_frames[i],
                     "fly": int(slot.replace("fly", "")),
                     "bout_name": grp_names[i],
                     "sex": sex[i]})

    kp = np.asarray(data[bout_keys[0]]["kp_data"])
    n_kp = kp.shape[1] // 3 if kp.ndim == 2 else kp.shape[1]
    if n_kp != len(kp_names):
        raise SystemExit(f"{h5_path.name}: kp_data has {n_kp} keypoints but "
                         f"kp_names has {len(kp_names)} -- indexing one by the "
                         f"other would measure the wrong body part")
    print(f"loaded {h5_path.name}: {n} bout-flies, "
          f"{len(set(r['recording'] for r in rows))} recordings, "
          f"{len(kp_names)} keypoints")
    return data, info, kp_names, bout_keys, rows


def build_pairs(rows, bout_keys):
    """(key_fly0, key_fly1) for every bout where both flies were reconstructed.

    Grouped by (recording, bout_name) with the slot taken from `fly_uid`, so
    pairing never depends on two rows happening to be adjacent.
    """
    seen = {}
    for r in rows:
        seen.setdefault((r["recording"], r["bout_name"]), {})[r["fly"]] = r["key"]
    pairs = [(s[0], s[1]) for s in seen.values() if 0 in s and 1 in s]
    pairs.sort(key=lambda p: bout_keys.index(p[0]))
    print(f"pairs: {len(pairs)} of {len(rows) // 2} possible")
    return pairs


def find_exemplar(results, rows):
    """The pinned exemplar pair, matched by (recording, start frame).

    There is no fallback to "some other bout": that is how the wrong exemplar
    shipped in the source repo.
    """
    by_key = {r["key"]: r for r in rows}
    for r in results:
        row = by_key.get(r["key0"])
        if (row and EXEMPLAR_RECORDING in row["recording"]
                and row["start_frame"] == EXEMPLAR_START_FRAME):
            return r, row
    avail = sorted({by_key[r["key0"]]["recording"] for r in results
                    if r["key0"] in by_key})
    raise SystemExit(
        f"exemplar not found: {EXEMPLAR_RECORDING!r} @ start_frame "
        f"{EXEMPLAR_START_FRAME}.\n  {len(results)} surviving pairs across "
        f"{len(avail)} recordings: {avail}")


def _sam3_bout_name(row) -> str:
    """The processed tree's bout directory for a row, checked against the CSV.

    The name comes from the h5; where the recording's bout summary is readable
    the two must agree, since a disagreement means masks and keypoints would be
    read from an unrelated bout.
    """
    name = row.get("bout_name") or ""
    rec_dir = PROCESSED_ROOT / row["recording"]
    for csv in sorted(rec_dir.glob("*/bouts*.csv")) + sorted(rec_dir.glob("*.csv")):
        try:
            import pandas as pd
            df = pd.read_csv(csv)
        except Exception:                          # noqa: BLE001
            continue
        if not {"start_frame", "bout_idx"} <= set(df.columns):
            continue
        hit = df[df["start_frame"] == int(row["start_frame"])]
        if len(hit) != 1:
            continue
        from_csv = f"bout_{int(hit.iloc[0]['bout_idx']):05d}"
        if name and from_csv != name:
            raise LookupError(
                f"{row['recording']} start_frame {row['start_frame']}: the h5 "
                f"names this bout {name!r} but {csv.name} says {from_csv!r}. "
                f"One of them points at a different bout's masks and kp3d.")
        return from_csv
    if name:
        return name
    raise LookupError(f"no bout name for start_frame {row['start_frame']} "
                      f"under {rec_dir}")


def _sync_plan(session_dir):
    """The recording's sync plan, or None when it has no dropped-frame record."""
    p = Path(session_dir) / "sync_plan.json"
    if not p.exists():
        return None
    import json
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# Stage 1: panels derivable from the combined h5
# --------------------------------------------------------------------------

def stage_analysis(h5_path, pose_dir, skipped):
    """Run the song/locomotion analysis and shape it into panel arrays."""
    from scipy.signal import hilbert

    data, info, kp_names, bout_keys, rows = load_combined(h5_path)
    pairs = build_pairs(rows, bout_keys)

    song = A.SongAnalysisConfig()
    song.pipeline = "both"
    results = A.analyze_all_pairs(
        data, pairs, kp_names, song_cfg=song, sex_cfg=A.SexIdConfig(),
        loc_cfg=A.LocomotionConfig(), pair_cfg=A.PairValidityConfig())
    if not results:
        raise SystemExit("no pairs survived filtering; nothing to plot")

    # `analyze_pair` puts the male in slot 0 from song and body length; the h5
    # states the sex outright. Two independent records, so a disagreement means
    # neither can be trusted -- and every pooled panel is conditioned on the
    # male.
    by_key = {r["key"]: r for r in rows}
    disagree = [r["key0"] for r in results
                if by_key.get(r["key0"], {}).get("sex") != "male"]
    if disagree:
        raise SystemExit(
            f"sex cross-check FAILED on {len(disagree)}/{len(results)} pairs: "
            f"the song/body-length sexing put a fly the h5 calls FEMALE in the "
            f"male slot (e.g. {disagree[:4]}). Every pooled panel is "
            f"conditioned on the male, so this is not cosmetic.")
    print(f"sex cross-check: song-based sexing agrees with info/sex on all "
          f"{len(results)} pairs")

    # Males whose partner could not be reconstructed contribute nothing to
    # `pairs`. Pool them into the single-fly panels only (D, E, F, G); the
    # exemplar, pitch traces and violin are pair quantities.
    singles = X.analyze_unpaired_males(data, bout_keys, info, pairs, kp_names,
                                       song_cfg=song)
    if singles:
        print(f"single-fly: pooling {len(singles)} unpaired male bout(s) into "
              f"the wing/pulse/z-height panels")
    pooled = list(results) + list(singles)

    ex, ex_row = find_exemplar(results, rows)
    T = int(ex["T"])
    print(f"exemplar {ex['key0']}/{ex['key1']} from {ex_row['recording']} "
          f"@ start_frame {ex_row['start_frame']} (T={T})")

    t_ms = (np.arange(T) / FS) * 1000.0
    s = ex["song0"]
    wd = s["wing_data"]
    zL = np.asarray(wd["WingL_V13"]["z"], float)
    zR = np.asarray(wd["WingR_V13"]["z"], float)
    seg_L = segments_to_array(s["sides"]["L"]["segments"])
    seg_R = segments_to_array(s["sides"]["R"]["segments"])
    ext_is_L = np.asarray(s["angle_L"], float) > np.asarray(s["angle_R"], float)
    ext_z = np.where(ext_is_L, zL, zR)[:T]
    fold_z = np.where(ext_is_L, zR, zL)[:T]

    phase_diffs = []
    for r in pooled:
        w = r["song0"]["wing_data"]
        a_all = np.asarray(w["WingL_V13"]["z"], float)
        b_all = np.asarray(w["WingR_V13"]["z"], float)
        for seg in r["song0"]["sides"]["L"]["segments"]:
            if seg.get("type") != "sine":
                continue
            i0, i1 = int(seg["start"]), int(seg["end"]) + 1
            if i1 - i0 < 16:
                continue
            a, b = a_all[i0:i1], b_all[i0:i1]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            R = np.mean(np.exp(1j * (np.angle(hilbert(a - a.mean()))
                                     - np.angle(hilbert(b - b.mean())))))
            if np.isfinite(R):
                phase_diffs.append(np.angle(R))

    ext_pulse, ext_sine = [], []
    for r in pooled:
        hL, hR = r["song0"].get("horiz_angle_L"), r["song0"].get("horiz_angle_R")
        if hL is None or hR is None:
            continue
        yL, yR = np.asarray(hL, float), np.asarray(hR, float)
        ext = np.abs(np.where(np.abs(yL) > np.abs(yR), yL, yR))
        lab = np.asarray(r["male_labels"])
        base = np.asarray(r["male_valid"], bool) & np.isfinite(ext)
        if (base & (lab == "pulse")).any():
            ext_pulse.append(ext[base & (lab == "pulse")])
        if (base & (lab == "sine")).any():
            ext_sine.append(ext[base & (lab == "sine")])

    ptr = A.get_pulse_type_labels(pooled, fs=FS)

    # Panel C's vertical pulse lines. Without these the trace shows song
    # shading but no individual pulse events.
    labels = ptr.get("labels", {}) or {}
    ex_pair = int(ex.get("pair_idx", -1))
    pulse_side = {}
    for side in ("L", "R"):
        pf = (ex["song0"]["sides"].get(side, {}) or {}).get("pulse_features", {}) or {}
        peaks = np.asarray(pf.get("peak_frames", np.zeros(0, dtype=int)))
        labs = np.asarray((labels.get(ex_pair, {}) or {}).get(side, np.array([])))
        if peaks.size and labs.size == peaks.size:
            pulse_side[side] = {"peak_frames": peaks, "labels": labs}
    print("pulse lines: " + (", ".join(f"{k}={len(v['peak_frames'])}"
                                       for k, v in pulse_side.items()) or "none"))

    # Panel G's walking arm must use the same floor-corrected estimator as the
    # courtship arms, or the panel compares two different quantities.
    walking_z = np.zeros(0)
    try:
        walking_z = X._free_running_com_z(FREE_RUN_H5)
        print(f"free-running baseline: {len(walking_z)} bouts")
    except Exception as e:                        # noqa: BLE001
        skipped.append(f"panel G free-running arm ({type(e).__name__}: {e})")

    # Frames whose Scutellum leaves the recording's own arena envelope are a
    # triangulation failure near a wall, not a real climb. Guarded per bout:
    # one unresolvable recording must not silently ungate the rest.
    arena_bad, gated, ungated = {}, 0, []
    for r in results:
        row = by_key.get(r["key0"])
        if row is None:
            continue
        try:
            bad = X._outside_arena(
                PROCESSED_ROOT / row["recording"], _sam3_bout_name(row),
                f"fly{row['fly']}", int(np.asarray(r["com_z"]).size),
                pose_dir=pose_dir)
        except Exception as e:                    # noqa: BLE001
            ungated.append(f"{row['recording']}/{row['bout_name']} "
                           f"({type(e).__name__}: {e})")
            continue
        if bad.any():
            arena_bad[r["key0"]] = bad
            gated += 1
    print(f"arena gate: {gated}/{len(results)} bouts have frames outside their "
          f"recording's Scutellum envelope")
    if ungated:
        skipped.append(f"arena gate: {len(ungated)} bout(s) ungated, "
                       f"e.g. {ungated[0]}")

    panels = {
        "wing": {"type": "courtship.wing_z", "data": {
            "t_ms": t_ms, "wingL_z": zL[:T], "wingR_z": zR[:T],
            "seg_L": seg_L, "seg_R": seg_R,
            **{f"pulse_{k}_{f}": np.asarray(v[f])
               for k, v in pulse_side.items() for f in ("peak_frames", "labels")}}},
        # scut shades the union of L+R, matching the wing panel above it.
        "scut": {"type": "courtship.scutellum_z", "data": {
            "t_ms": t_ms, "scutellum_z": np.asarray(ex["com_z"], float)[:T],
            "segments": seg_L, "segments_R": seg_R}},
        "sine_phase": {"type": "courtship.sine_inphase", "data": {
            "t_ms": t_ms - t_ms[0], "ext_z": ext_z, "fold_z": fold_z,
            "sine_segments": segments_to_array(
                [g for g in s["sides"]["L"]["segments"] if g.get("type") == "sine"])}},
        "wing_phase_polar": {"type": "courtship.wing_polar", "data": {
            "phase_diffs": np.asarray(phase_diffs, float)}},
        "angle_2d": {"type": "courtship.angle_density", "data": {
            "ext_pulse": np.concatenate(ext_pulse) if ext_pulse else np.zeros(0),
            "ext_sine": np.concatenate(ext_sine) if ext_sine else np.zeros(0)}},
        "pulse_class": {"type": "courtship.pulse_class", "data": {
            **{f"centroid_{t}": np.asarray(
                ptr.get("centroids", {}).get(t, np.zeros(0)), float)
               for t in _PULSE_TYPES},
            **{f"pooled_{t}": np.asarray(
                ptr.get("pooled_waveforms", {}).get(t, np.zeros((0, 0))), float)
               for t in _PULSE_TYPES}}},
        "zheight": {"type": "courtship.zheight", "data": {
            "pulse_z": X._mean_z_by_label(pooled, "pulse", arena_bad),
            "sine_z": X._mean_z_by_label(pooled, "sine", arena_bad),
            "walking_z": np.asarray(walking_z, float)}},
    }
    ctx = {"data": data, "info": info, "kp_names": kp_names, "rows": rows,
           "bout_keys": bout_keys, "results": results, "ex": ex,
           "ex_row": ex_row, "T": T}
    return panels, ctx


# --------------------------------------------------------------------------
# Stage 2: video, MuJoCo renders, pitch traces, pooled violin
# --------------------------------------------------------------------------

def stage_assets(ctx, pose_dir, skipped):
    """Panels A, H, I and J.

    Each panel is independently guarded: a missing SAM3 tree must not take the
    renders down with it, and a failure is reported at the end rather than
    aborting a figure whose other panels are fine.
    """
    import cv2

    data, kp_names = ctx["data"], ctx["kp_names"]
    ex, ex_row, T = ctx["ex"], ctx["ex_row"], ctx["T"]
    panels = {}

    recording = ex_row["recording"]
    rec_dir = PROCESSED_ROOT / recording
    session_dir = VIDEO_ROOT / recording
    calib_dir = session_dir / "calibration"
    sam3_root = rec_dir / "sam3_masks"
    vidx = np.asarray(VIDEO_FRAMES, dtype=int)
    vidx = vidx[vidx < T]

    sam3_bout = _sam3_bout_name(ex_row)
    sex_json = None
    sj = rec_dir / pose_dir / "bouts" / sam3_bout / "sex.json"
    if sj.exists():
        import json
        sex_json = json.loads(sj.read_text())
    male_slot, female_slot = A._resolve_male_female_slots(None, sex_json)
    print(f"exemplar -> {recording}/{sam3_bout}  male_slot={male_slot} "
          f"female_slot={female_slot}")

    # key0 is the male after `analyze_pair`'s reorder; resolve each fly's own
    # directory from its row rather than assuming key0 -> fly0.
    rows_by_key = {r["key"]: r for r in ctx["rows"]}
    bout_dir = rec_dir / pose_dir / "bouts" / sam3_bout
    male_kp3d = A._load_kp3d(
        bout_dir / f"fly{rows_by_key[ex['key0']]['fly']}" / "kp3d.npz",
        expected_n_kp=len(kp_names), expected_kp_names=kp_names)
    female_kp3d = A._load_kp3d(
        bout_dir / f"fly{rows_by_key[ex['key1']]['fly']}" / "kp3d.npz",
        expected_n_kp=len(kp_names), expected_kp_names=kp_names)

    npz_cameras = None

    # --- panel A ---------------------------------------------------------
    try:
        dlt = X._dlt_load(calib_dir / f"{EXEMPLAR_CAMERA}_dlt.csv")
        sam3_npz = sam3_root / sam3_bout / "sam3_masks.npz"
        with np.load(sam3_npz, allow_pickle=True) as z:
            npz_cameras = ([str(c) for c in z["cameras"]]
                           if "cameras" in z.files else None)
            shape = np.asarray(z["shape"]).ravel()[:2]
        cam_idx, how = X._resolve_sam3_camera_index(
            npz_cameras, str(calib_dir), EXEMPLAR_CAMERA)
        print(f"sam3 camera {EXEMPLAR_CAMERA} -> axis {cam_idx} ({how})")

        # Bout frame 0 is not video frame `start_frame` when the recording
        # dropped frames, so go through the sync plan.
        offset, realigned = X._slot_to_raw_frame(
            _sync_plan(session_dir), EXEMPLAR_CAMERA, int(ex_row["start_frame"]))
        if realigned:
            print(f"sync plan: slot {ex_row['start_frame']} -> raw frame {offset}")

        H_, W_ = int(shape[0]), int(shape[1])
        cw, ch = min(CROP_WH[0], W_), min(CROP_WH[1], H_)
        masks = X.unpack_sam3_masks_for_frames(
            sam3_npz, cam_idx, fly_indices=[male_slot, female_slot],
            frame_indices=[int(f) for f in vidx])

        # Crop centred on the pair's Scutellum midpoint, clamped to the frame
        # so a pair at the wall still yields a full-size window. A NaN centre
        # carries the nearest finite one rather than collapsing to (0, 0).
        center = X._pair_center_xyz(male_kp3d, female_kp3d,
                                    kp_names.index("Scutellum"), T)
        uvc = np.asarray(X._dlt_project(dlt, np.asarray(center, float) * KP_SCALE))
        good = np.isfinite(uvc).all(axis=1)
        if not good.any():
            raise RuntimeError("pair centre is NaN for the whole bout")
        idx = np.maximum.accumulate(np.where(good, np.arange(len(uvc)), -1))
        idx[idx < 0] = int(np.argmax(good))
        uvc = uvc[idx]

        cap = cv2.VideoCapture(str(session_dir / f"{EXEMPLAR_CAMERA}.mp4"))
        try:
            for i, fi in enumerate(int(f) for f in vidx):
                u, v = uvc[fi] if fi < len(uvc) else uvc[-1]
                roi = (max(0, min(int(round(u - cw / 2)), W_ - cw)),
                       max(0, min(int(round(v - ch / 2)), H_ - ch)), cw, ch)
                frame = X._read_frame(cap, fi + int(offset), roi)
                if frame is None:
                    raise RuntimeError(f"could not read video frame {fi}")
                x0, y0, w_r, h_r = roi
                m_crop = [np.asarray(ms[i], bool)[y0:y0 + h_r, x0:x0 + w_r]
                          for ms in masks]
                panels[f"video_{i}"] = {"type": "courtship.video_kp", "data": {
                    "kp_uv_male": X._project_kp_to_crop(
                        male_kp3d[fi], dlt, KP_SCALE, roi, frame.shape[:2],
                        project=X._dlt_project),
                    "kp_uv_female": X._project_kp_to_crop(
                        female_kp3d[fi], dlt, KP_SCALE, roi, frame.shape[:2],
                        project=X._dlt_project)},
                    "assets": {"img": X._blend_masks(
                        frame, m_crop, ["#e74c3c", "#3a7bff"], 0.35)}}
        finally:
            cap.release()
        print(f"panel A: {len(vidx)} native crops at {cw}x{ch}")
    except Exception as e:                        # noqa: BLE001
        skipped.append(f"panel A video strip ({type(e).__name__}: {e})")

    # --- panel H ---------------------------------------------------------
    try:
        # Red goes to whichever fly is passed first, and panel G labels male
        # red / female blue, so assert rather than assume the order.
        if rows_by_key[ex["key0"]]["sex"] != "male":
            raise RuntimeError(f"key0 {ex['key0']} is not the male; red would "
                               f"be applied to the female")
        q_pair = X._pair_qpos(np.asarray(data[ex["key0"]]["qpos"]),
                              np.asarray(data[ex["key1"]]["qpos"]), T)
        # A NaN qpos row renders as a black frame, so snap to renderable ones.
        ok_idx = np.flatnonzero(np.isfinite(q_pair[:T]).all(axis=1))
        if ok_idx.size == 0:
            raise RuntimeError("no finite qpos frames to render")
        ridx = X._snap_to_renderable(vidx, ok_idx)
        if not np.array_equal(ridx, vidx):
            print(f"panel H: snapped {list(vidx)} -> {list(ridx)}")
        rig_pos = None
        try:
            rig_pos = X._rig_pos_for_session(
                data, ctx["bout_keys"],
                [r["recording"] for r in ctx["rows"]], recording,
                offset=RIG_OFFSET)
        except Exception as e:                    # noqa: BLE001
            skipped.append(f"panel H arena placement ({type(e).__name__}: {e}) "
                           f"-- bare floor")
        imgs = X._render_frames(MODEL_XML, FLOOR_XML, q_pair, ridx,
                                size=RENDER_SIZE, rig_pos=rig_pos)
        for i, im in enumerate(imgs):
            panels[f"render_{i}"] = {"type": "image",
                                     "assets": {"img": np.asarray(im, np.uint8)}}
        print(f"panel H: {len(imgs)} renders at {RENDER_SIZE}px")
    except Exception as e:                        # noqa: BLE001
        skipped.append(f"panel H render strip ({type(e).__name__}: {e})")

    # --- panel I ---------------------------------------------------------
    try:
        camera_order = ([f"{c}_dlt.csv" for c in npz_cameras]
                        if npz_cameras else None)
        female_com = X.triangulate_sam3_female_com(
            str(sam3_root / sam3_bout / "sam3_masks.npz"), str(calib_dir),
            fly_idx=female_slot, camera_order=camera_order,
            min_cams=2, verbose=False) / KP_SCALE
        qm = np.asarray(data[ex["key0"]]["qpos"])
        n = min(T, qm.shape[0], female_com.shape[0], len(male_kp3d))
        male_pitch = X.body_pitch_deg_from_quat(qm[:n, 3:7])
        # The Scutellum comes from the tree's kp3d (world frame), not the h5's
        # re-centred kp_data: mixing a re-centred position against the
        # world-frame female COM puts target_pitch in no coherent frame.
        scut = male_kp3d[:n, kp_names.index("Scutellum"), :]
        vec = female_com[:n] - scut
        nrm = np.linalg.norm(vec, axis=-1)
        target_pitch = np.degrees(np.arcsin(np.divide(
            vec[..., 2], nrm, out=np.full_like(nrm, np.nan), where=nrm > 0)))
        panels["pitch"] = {"type": "courtship.male_pitch", "data": {
            "t_ms": (np.arange(n) / FS) * 1000.0,
            "male_pitch": male_pitch, "target_pitch": target_pitch}}
        print(f"panel I: {n} frames, male pitch "
              f"{np.nanmin(male_pitch):.1f}..{np.nanmax(male_pitch):.1f} deg")
    except Exception as e:                        # noqa: BLE001
        skipped.append(f"panel I pitch trace ({type(e).__name__}: {e})")

    # --- panel J ---------------------------------------------------------
    try:
        # Two independent records of which fly is the male: the tree's sex.json
        # and the h5's info/sex. on_conflict="raise" makes a disagreement fatal
        # rather than a silently shorter violin.
        h5_male_fly = {}
        for r in ctx["rows"]:
            if r["sex"] == "male" and r["bout_name"]:
                h5_male_fly[(r["recording"], r["bout_name"])] = int(r["fly"])
        bouts = A.find_mvq_bouts(
            str(PROCESSED_ROOT), str(VIDEO_ROOT), pose_dir=pose_dir,
            h5_male_fly=h5_male_fly or None, on_conflict="raise",
            skipped=skipped, accept_authorities=SEX_AUTHORITIES)
        if not bouts:
            raise RuntimeError(f"no bouts with sex.json + kp3d + masks under "
                               f"{PROCESSED_ROOT}/*/*/{pose_dir}/bouts")
        print(f"panel J: {len(bouts)} bouts pass the sex.json/info/sex "
              f"cross-check")
        al = A.compute_pitch_alignment_mvq(
            bouts, kp_scale=KP_SCALE, expected_kp_names=kp_names,
            scut_com_offset_tol=SCUT_COM_OFFSET_TOL)
        per_bout = np.asarray(al["median_abs_alignment_deg"], float)
        ex_idx = -1
        for i, (r, b) in enumerate(zip(al["recordings"], al["bout_names"])):
            if X._recording_base(r) == X._recording_base(recording) and b == sam3_bout:
                ex_idx = i
                break
        if ex_idx < 0:
            skipped.append(f"panel J exemplar marker: {recording}/{sam3_bout} "
                           f"not in the pooled {len(per_bout)} bouts")
        panels["align_violin"] = {"type": "courtship.pitch_violin", "data": {
            "per_bout": per_bout,
            "exemplar_idx": np.asarray([ex_idx], np.int64)}}
        print(f"panel J: {len(per_bout)} bouts over "
              f"{len(set(al['recordings']))} recordings; median "
              f"{np.nanmedian(per_bout):.2f} deg; exemplar at {ex_idx}")
    except Exception as e:                        # noqa: BLE001
        skipped.append(f"panel J align violin ({type(e).__name__}: {e})")

    return panels


# --------------------------------------------------------------------------
# Stage 3: layout
# --------------------------------------------------------------------------

def render_figure(panels, out_path, transparent=None):
    """Lay the panels out at fig4.json's rects and save SVG + PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        # Liberation Sans is metric-compatible with Arial and is what
        # fontconfig substitutes when Arial is absent, so the layout font
        # matches the font actually drawn on a machine without Arial.
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.0, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        # Text must export as real <text>, and PDF must embed Type-42.
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        # A fixed salt makes matplotlib's marker ids content-derived rather
        # than random, so the same inputs export identical SVG.
        "svg.hashsalt": "fig4",
    })
    matplotlib.rcParams.update(L.STYLE)

    mm = 1.0 / 25.4
    fig = plt.figure(figsize=(L.WIDTH_MM * mm, L.HEIGHT_MM * mm), dpi=L.DPI)

    drawn, missing = [], []
    for panel in L.PANELS:
        pid, ptype = panel["id"], panel["type"]
        pd = panels.get(pid)
        if pd is None or not (pd.get("data") or pd.get("assets")):
            missing.append(pid)
            continue
        ax = fig.add_axes(L.RECTS[pid], projection=P.PROJECTION.get(ptype))
        merged = {**(pd.get("data") or {}), **(pd.get("assets") or {})}
        spec = L.SPECS[pid]
        try:
            P.DRAW[ptype](ax, merged, spec)
            P.apply_cosmetics(ax, spec)
            drawn.append(pid)
        except Exception as e:                    # noqa: BLE001
            ax.set_axis_off()
            missing.append(f"{pid} (draw failed: {type(e).__name__}: {e})")

    for ann in L.ANNOTATIONS:
        parent = ann.get("parent")
        if parent not in L.RECTS:
            continue
        if ann.get("id", "").startswith("letter_") and parent not in drawn:
            continue
        x, y = L.annotation_xy(L.RECTS[parent], ann["pos_mm"])
        st = ann.get("style") or {}
        fig.text(x, y, ann["text"], fontsize=st.get("font_size_pt", 6),
                 fontweight="bold" if st.get("weight") == "bold" else "normal",
                 color=st.get("color", "#000000"),
                 ha="left", va="baseline")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tr = L.TRANSPARENT if transparent is None else transparent
    svg, png = out_path.with_suffix(".svg"), out_path.with_suffix(".png")
    fig.savefig(svg, transparent=tr)
    fig.savefig(png, dpi=L.DPI, transparent=tr)
    plt.close(fig)
    print(f"drew {len(drawn)}/{len(L.PANELS)} panels -> {png}")
    if missing:
        print(f"  not drawn: {', '.join(missing)}")
    return svg, png


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h5", default=str(H5), help="combined courtship h5")
    ap.add_argument("--pose-dir", default=POSE_DIR,
                    help="pose tree under each recording (kp3d, masks, sex.json)")
    ap.add_argument("-o", "--out", default=str(_HERE / "fig4"),
                    help="output path without extension (.svg and .png written)")
    ap.add_argument("--cache-dir", default=str(_HERE / "cache"))
    ap.add_argument("--refresh", action="store_true",
                    help="recompute both stages, ignoring the cache")
    ap.add_argument("--skip-assets", action="store_true",
                    help="stage 1 + 3 only: panels B-G, which need no video, "
                         "MuJoCo or SAM3 masks. Runs without a GPU.")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    h5_path = Path(args.h5)
    bundle = cache_dir / f"bundle_{h5_path.stem[:40]}.h5"
    skipped: list[str] = []

    if bundle.exists() and not args.refresh:
        panels = read_bundle(bundle)
        print(f"cache HIT  {bundle.name} ({len(panels)} panels) "
              f"-- pass --refresh to recompute")
    else:
        panels, ctx = stage_analysis(h5_path, args.pose_dir, skipped)
        if not args.skip_assets:
            panels.update(stage_assets(ctx, args.pose_dir, skipped))
        write_bundle(bundle, panels)
        print(f"cache MISS -> wrote {bundle.name} ({len(panels)} panels)")

    render_figure(panels, args.out)

    if skipped:
        print(f"\n{len(skipped)} thing(s) skipped -- reported, not hidden:")
        for s in skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
