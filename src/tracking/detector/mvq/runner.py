"""frames -> windows -> world keypoints -> pipeline format, for ONE mvq checkpoint."""

from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np
from flax import nnx

from tracking.detector.mvq.checkpoint import load_mvq_model
from tracking.detector.mvq.gates import gate_string
from tracking.detector.mvq.model import assemble
from tracking.detector.mvq.policy import EXIST_THRESH, N_SLOTS, SLOT_FEMALE, SLOT_MALE
from tracking.geometry.rig import CameraRig
from tracking.io.names import Order

# Window crop size (`v12_windows.py::CROP`) -- every camera is cropped to a
# CROP x CROP square centred on the projection of the window's 3D centre.
CROP = 448

# ImageNet normalisation stats the DINOv3 backbone was pretrained with
# (`data/transforms.py::IMAGENET_MEAN/STD`).
_IMAGENET_MEAN = jnp.asarray([0.485, 0.456, 0.406], jnp.float32)
_IMAGENET_STD = jnp.asarray([0.229, 0.224, 0.225], jnp.float32)

# Sex codes, fixed by the P3a spec: 0 = female, 1 = male,
# -1 = unknown (the single-fly rule).
SEX_FEMALE, SEX_MALE, SEX_UNKNOWN = 0, 1, -1

WINDOW_PREF_MODES = ("own", "any")
DEFAULT_WINDOW_PREF = "own"

COLLAPSE_DIST_UNITS = 3.0

HALLUCINATION_VIS_THRESH = 0.5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def resolved_window_pref(window_pref):
    """`window_pref` as one of `WINDOW_PREF_MODES` (None -> `DEFAULT_WINDOW_PREF`)."""
    if window_pref is None:
        return DEFAULT_WINDOW_PREF
    window_pref = str(window_pref).strip().lower()
    if window_pref not in WINDOW_PREF_MODES:
        raise ValueError(
            f"window_pref must be one of {list(WINDOW_PREF_MODES)}, got {window_pref!r}"
        )
    return window_pref


def _affine_np(matrices_f64):
    """`(C,3,4)` untransposed DLT projection matrices -> `M (C,2,3)`, `t (C,2)`
    (float64) such that a WORLD point projects to CROP pixels as
    `uv = M @ X + t` (before subtracting the crop origin).
    """
    P = np.asarray(matrices_f64, np.float64)  # (C,3,4)
    if not np.allclose(P[:, 2, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera is not affine: projection row 3 != [0,0,0,1]")
    return P[:, :2, :3], P[:, :2, 3]


def crop_origin(bbox, img_w, img_h, crop=CROP):
    """Top-left `(x0, y0)` of a `crop`x`crop` window centred on `bbox`'s
    centre, clamped so the window stays inside the image. `bbox` is
    `[x, y, w, h]`."""
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    x0 = int(round(cx - crop / 2.0))
    y0 = int(round(cy - crop / 2.0))
    x0 = max(0, min(x0, img_w - crop))
    y0 = max(0, min(y0, img_h - crop))
    return x0, y0


def normalize_crops(u8):
    return (u8.astype(jnp.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD


@nnx.jit
def _fwd(model, crops, cam_valid, M, t_local, prompt_mask, prompt_on):
    return model(crops, cam_valid, M, t_local, prompt_mask, prompt_on=prompt_on)


def slot_read(out, bi, slot):
    """One instance slot of one window of an `infer` output, as a plain dict."""
    slot = int(slot)
    return {
        "slot": slot,
        "kp3d": out["kp3d"][bi, slot],
        "kp2d": out["kp2d"][bi, slot],
        "vis": out["vis"][bi, slot],
        "conf_raw": out["conf_raw"][bi, slot],
        "exist": float(out["exist"][bi, slot]),
        "sex_prob": float(out["sex_prob"][bi, slot]),
    }


def _passes_vis_guard(read, *, min_vis):
    """False iff `read['vis']` (C,K) per-view visibility sigmoid is below
    `min_vis` in EVERY camera -- catches a window that has drifted off both
    flies (`exist` can still clear threshold while `vis` is ~0 everywhere).
    """
    if min_vis is None:
        return True
    vis = read.get("vis") if isinstance(read, dict) else None
    if vis is None:
        return True
    vis = np.asarray(vis, np.float64)
    if vis.size == 0:
        return True
    with np.errstate(invalid="ignore"):
        per_cam = np.nanmean(vis, axis=-1)  # (C,) mean over KEYPOINTS
    if not np.isfinite(per_cam).any():
        return True
    return bool(np.nanmax(per_cam) >= float(min_vis))


def prefer_own_window(cands, own_row, pref, b_at):
    """The best of `cands` (sorted-comparable tuples, best == `min`), taken
    from window `own_row` if that window offers one.
    """
    if pref == "own" and int(own_row) >= 0:
        own_only = [c for c in cands if b_at(c) == int(own_row)]
        if own_only:
            return min(own_only)
    return min(cands)


def _own_rows(own_windows, off, nb):
    """Per-fly LOCAL window indices -> ABSOLUTE rows of an `infer` output."""
    if own_windows is None:
        return [-1, -1]
    return [(int(off) + int(w)) if 0 <= int(w) < int(nb) else -1 for w in own_windows]


def pick_typed_pair(
    out,
    off,
    nb,
    runner,
    collapse_dist_units=COLLAPSE_DIST_UNITS,
    *,
    own_windows=None,
    window_pref=None,
    min_vis=None,
):
    """Read the FEMALE (fi=0) and MALE (fi=1) typed slots for one pending
    frame's windows `[off, off + nb)` of an `infer` output, apply the
    collapse guard, and return the survivors.

    Returns:
        picks: {fi: (read_typed_result, absolute_window_index)} for fi in
            {0, 1} that a typed slot was found for -- 0, 1 or 2 entries.
        collapsed: bool, whether the collapse guard fired this frame.
        dropped_fi: the fi (0 or 1) the guard NaN'd, or None if it did not
            fire (or fewer than 2 typed slots were read at all).
        own: {fi: bool} for each fly PICKED -- whether its read came out of
            its own window.

    `min_vis`: the existence hallucination guard (`HALLUCINATION_VIS_THRESH`,
    `_passes_vis_guard`) -- a candidate whose per-view visibility is below
    `min_vis` in EVERY camera is dropped BEFORE the argmax/own-window choice
    above, even if its `exist` clears `runner.exist_thresh`. `None` (the
    default) is a no-op.
    """
    pref = resolved_window_pref(window_pref)
    own_row = _own_rows(own_windows, off, nb)
    picks = {}
    for fi, want_sex in enumerate((SEX_FEMALE, SEX_MALE)):
        # (-exist, b): `min` == highest existence, ties to the LOWEST window row.
        cands = []
        for b in range(off, off + nb):
            r = runner.read_typed(out, b, want_sex=want_sex)
            if r is not None and _passes_vis_guard(r, min_vis=min_vis):
                cands.append((-float(r["exist"]), int(b), r))
        if cands:
            _ne, best_b, best = prefer_own_window(cands, own_row[fi], pref, lambda c: c[1])
            picks[fi] = (best, best_b)

    collapsed, dropped_fi = False, None
    if len(picks) == 2:
        a3 = np.asarray(picks[0][0]["kp3d"], np.float64)
        b3 = np.asarray(picks[1][0]["kp3d"], np.float64)
        d = np.linalg.norm(a3 - b3, axis=-1)
        d = d[np.isfinite(d)]
        if d.size and float(np.median(d)) < float(collapse_dist_units):
            collapsed = True
            dropped_fi = 1 if picks[1][0]["exist"] <= picks[0][0]["exist"] else 0
            picks.pop(dropped_fi)
    own = {fi: (own_row[fi] >= 0 and b == own_row[fi]) for fi, (_r, b) in picks.items()}
    return picks, collapsed, dropped_fi, own


class MVQRunner:
    """frames -> windows -> world keypoints -> pipeline format, for ONE checkpoint."""

    def __init__(
        self,
        run_dir_or_final,
        *,
        step=None,
        attn_impl=None,
        rig: CameraRig,
        kp_order: Order,
        batch=32,
        exist_thresh=EXIST_THRESH,
        window_pref="own",
    ):
        names = list(rig.cameras.names)
        if names != sorted(names):
            raise ValueError(
                f"rig camera order {names} is not the canonical glob order "
                f"{sorted(names)} -- every camera axis this runner produces assumes "
                f"they are the same, and a mismatch plots one camera's keypoints on "
                f"another camera's image and still looks almost plausible"
            )

        self.checkpoint = os.path.abspath(str(run_dir_or_final))
        self.window_pref = resolved_window_pref(window_pref)
        if step == "latest":
            import orbax.checkpoint as ocp

            mgr = ocp.CheckpointManager(
                os.path.abspath(os.path.join(self.checkpoint, "ckpt")),
                options=ocp.CheckpointManagerOptions(read_only=True),
            )
            step = int(mgr.latest_step())
        self.step = None if step is None else int(step)
        self.step_label = "final" if self.step is None else self.step
        self.model, self.meta = load_mvq_model(self.checkpoint, step=self.step, attn_impl=attn_impl)

        self.model_order = Order(self.meta["keypoint_names"])
        self.kp_order = kp_order
        self.kp_names = list(kp_order.names)
        self.K = len(self.kp_names)
        self.I = int(self.meta["model"]["n_instances"])

        self.rig = rig
        self.cameras = names
        self.C = len(self.cameras)
        self.cam_mats = rig.matrices_f32  # (C,4,3) f32, the model's own layout
        self.M, self.t = _affine_np(rig.matrices_f64)  # (C,2,3), (C,2) float64

        self.batch = int(batch)
        self.exist_thresh = float(exist_thresh)
        calib = self.meta.get("calibration") or {}
        self.exist_temperature = float(calib.get("exist_temperature", 1.0))
        self.vis_temperature = float(calib.get("vis_temperature", 1.0))
        self._gates = None

    # ------------------------------------------------------------------ windows
    def windows(self, frames, present, centres):
        """`V12WindowDataset._build`'s inference geometry, vectorised over
        centres. T=1 only (`temporal="none"`, the only mode this runner
        supports).
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[0] != self.C:
            raise ValueError(
                f"frames must be (C={self.C},H,W,3) in camera order {self.cameras}, "
                f"got {frames.shape}"
            )
        H, W = int(frames.shape[1]), int(frames.shape[2])
        centres = np.atleast_2d(np.asarray(centres, np.float64))
        if centres.shape[-1] != 3:
            raise ValueError(f"centres must be (B,3) world coordinates, got {centres.shape}")
        B = centres.shape[0]
        present = np.asarray(present, bool)
        if present.shape != (self.C,):
            raise ValueError(f"present must be (C={self.C},), got {present.shape}")

        uv = np.einsum("cij,bj->bci", self.M, centres) + self.t[None]  # (B,C,2)
        origin = np.zeros((B, self.C, 2), np.int32)
        crops = np.zeros((B, 1, self.C, CROP, CROP, 3), np.uint8)
        for b in range(B):
            for c in range(self.C):
                x0, y0 = crop_origin([uv[b, c, 0], uv[b, c, 1], 0, 0], W, H, CROP)
                origin[b, c] = (x0, y0)
                crops[b, 0, c] = frames[c][y0 : y0 + CROP, x0 : x0 + CROP]
        return {
            "crops": crops,
            "cam_valid": np.broadcast_to(present[None, None], (B, 1, self.C)).copy(),
            "M": np.broadcast_to(self.M.astype(np.float32)[None], (B, self.C, 2, 3)).copy(),
            "t_local": np.broadcast_to(
                (uv - origin).astype(np.float32)[:, None], (B, 1, self.C, 2)
            ).copy(),
            "origin": origin,
            "centres": centres.astype(np.float32),
        }

    # ------------------------------------------------------------------ forward
    def _infer_frames(self, w):
        """One forward on a windows dict, padded to `self.batch`, T axis kept
        (always 1 in this runner). No prompt path: this repo carries no
        masks, so every window is unprompted.
        """
        B0 = int(w["crops"].shape[0])
        if B0 > self.batch:
            raise ValueError(
                f"{B0} windows > runner batch {self.batch}; build at most `batch` "
                f"windows per infer call"
            )
        pad = self.batch - B0

        def _pad(a):
            return a if pad == 0 else np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0)

        crops, cam_valid = _pad(w["crops"]), _pad(w["cam_valid"])
        origin = _pad(w["origin"]).astype(np.float32)
        centres = _pad(w["centres"]).astype(np.float32)
        T = int(w["crops"].shape[1])
        prompt = np.zeros((self.batch, T, self.C, CROP, CROP), bool)
        prompt_on = np.zeros(self.batch, bool)
        out = _fwd(
            self.model,
            normalize_crops(jnp.asarray(crops)),
            jnp.asarray(cam_valid),
            jnp.asarray(_pad(w["M"])),
            jnp.asarray(_pad(w["t_local"])),
            jnp.asarray(prompt),
            jnp.asarray(prompt_on),
        )
        exist_logit = np.asarray(out["exist_logit"]) / self.exist_temperature
        vis_logit = np.asarray(out["vis_logit"]) / self.vis_temperature
        out_cal = dict(out)
        out_cal["exist_logit"] = exist_logit
        kp3d, conf3d, kp2d, sex_prob = assemble(
            out_cal,
            centres,
            origin,
            exist_thresh=self.exist_thresh,
            cam_valid=np.asarray(cam_valid),
        )
        return {
            "kp3d": kp3d[:B0],
            "kp2d": kp2d[:B0],
            "vis": _sigmoid(vis_logit)[:B0].astype(np.float32),
            "exist": _sigmoid(exist_logit)[:B0].astype(np.float32),
            "sex_prob": sex_prob[:B0],
            "conf_raw": conf3d[:B0],
            "xyz": np.asarray(out["xyz"])[:B0],
        }

    def infer(self, w, *, t_out=0):
        """One forward on a windows dict (see `_infer_frames`), for ONE
        frame of the window. `t_out` (default 0, the only frame a T=1 window
        has) is kept as a parameter for interface parity with the ported
        source, which also supported T>1 "pair" windows -- this runner is
        `temporal="none"` only, so every window built by `windows()` has
        exactly one frame.
        """
        out = self._infer_frames(w)
        t_out = int(t_out)
        T = int(out["kp3d"].shape[2])
        if not -T <= t_out < T:
            raise ValueError(f"t_out={t_out} is not a frame of a T={T} window")
        # `exist`/`sex_prob` have NO T axis -- a window's existence and sex
        # cover all of its frames.
        per_frame = ("kp3d", "kp2d", "vis", "conf_raw", "xyz")
        return {k: (v[:, :, t_out] if k in per_frame else v) for k, v in out.items()}

    # ------------------------------------------------------------------ reading slots
    def read_typed(self, out, bi, want_sex):
        """The typed slot for one sex of window `bi`, or None below threshold."""
        if self.I != N_SLOTS:
            raise ValueError(
                f"this checkpoint has {self.I} instance slots, not the typed {N_SLOTS} "
                f"(0 prompted, 1 female, 2 male, 3 other) -- a legacy untyped run's slots "
                f"have no fixed meaning, so read_typed cannot name one"
            )
        want = int(want_sex)
        if want not in (SEX_FEMALE, SEX_MALE, SEX_UNKNOWN):
            raise ValueError(
                f"want_sex must be {SEX_FEMALE} (female), {SEX_MALE} (male) or "
                f"{SEX_UNKNOWN} (unknown -- the single-fly rule), got {want_sex!r}"
            )
        if want == SEX_UNKNOWN:
            live = [
                s
                for s in (SLOT_FEMALE, SLOT_MALE)
                if float(out["exist"][bi, s]) >= self.exist_thresh
            ]
            if not live:
                return None
            slot = max(live, key=lambda s: float(out["exist"][bi, s]))
        else:
            slot = SLOT_FEMALE if want == SEX_FEMALE else SLOT_MALE
            if float(out["exist"][bi, slot]) < self.exist_thresh:
                return None
        return slot_read(out, bi, slot)

    # ------------------------------------------------------------------ pipeline format
    def to_pipeline(self, kp3d, kp2d, vis, conf_raw):
        """Permute the keypoint axis of each (non-`None`) array from the
        model's own order into the pipeline's canonical order.
        """
        perm = self.model_order.permutation_to(self.kp_order)
        out = []
        for a, kp_axis in ((kp3d, -2), (kp2d, -2), (vis, -1), (conf_raw, -1)):
            if a is None:
                out.append(None)
                continue
            a = np.asarray(a)
            idx = [slice(None)] * a.ndim
            idx[kp_axis] = perm
            out.append(a[tuple(idx)])
        return tuple(out)

    def gates_string(self):
        """This checkpoint's Stage-B `gates` payload as a stable string (see
        `tracking.detector.mvq.gates.gate_string`). Computed once and cached.
        """
        if self._gates is None:
            self._gates = gate_string(
                self.checkpoint,
                step=self.step,
                exist_thresh=self.exist_thresh,
                window_pref=self.window_pref,
            )
        return self._gates
