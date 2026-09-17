"""Affine-exact multi-view copy-paste. A donor fly (the host of another
window of the same calibration group) is translated by a 3D offset `D` in
the target's ROI-local frame; under affine cameras that is the per-camera 2D
translation `M_c D + t_local_tgt[c] - t_local_src[c]` in crop px, so pixels,
2D labels and 3D labels stay mutually consistent in every view.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from tracking.detector.mvq.slots import SEX_UNKNOWN


@dataclasses.dataclass(frozen=True)
class CopyPasteParams:
    p: float = 0.0
    opposite_sex_p: float = 0.7
    contact_p: float = 0.3
    contact_sep: tuple = (8.0, 30.0)
    far_sep: tuple = (15.0, 60.0)
    max_tries: int = 8
    gain_clip: tuple = (0.7, 1.4)


def body_plane_axes(kp3d_local, has3d):
    """(2,3) orthonormal axes spanning the two largest principal directions
    of the labelled points (the floor for a walking fly, the wall for a
    climber)."""
    pts = np.asarray(kp3d_local, np.float64)[np.asarray(has3d, bool)]
    if pts.shape[0] < 3:
        return np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    pts = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    return vt[:2].astype(np.float32)


def sample_offset(rng, axes, params: CopyPasteParams):
    """A 3D offset `D` on the body plane spanned by `axes`, at a separation
    drawn from `contact_sep` (probability `contact_p`) or `far_sep`."""
    lo, hi = params.contact_sep if rng.uniform() < params.contact_p else params.far_sep
    sep = rng.uniform(lo, hi)
    ang = rng.uniform(0, 2 * np.pi)
    return (sep * (np.cos(ang) * axes[0] + np.sin(ang) * axes[1])).astype(np.float32)


def view_shifts(M, D, t_local_src, t_local_tgt):
    """(C,2) crop-px translation moving the donor's crop content to where it
    would appear in the target crop after the 3D offset `D`.

    `kp2d_local = M @ X_local + t_local` in both frames, so the donor's own
    pixel is `uv_src = M @ X + t_local_src` and the pasted pixel is
    `uv_tgt = M @ (X + D) + t_local_tgt`; the shift is `M @ D + t_local_tgt -
    t_local_src`.
    """
    return (
        np.einsum("cij,j->ci", np.asarray(M, np.float64), np.asarray(D, np.float64))
        + np.asarray(t_local_tgt, np.float64)
        - np.asarray(t_local_src, np.float64)
    ).astype(np.float32)


def _translate(img, shift, nearest=False):
    h, w = img.shape[:2]
    A = np.array([[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]], np.float64)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img, A, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def composite(tgt, src, D, params: CopyPasteParams):
    """Paste `src`'s host fly into `tgt` as fly 1. Both are T>=1 window
    samples (numpy); `tgt`/`src` need not share window spacing, only window
    length `T` (a donor of a different `T` is rejected). `tgt` must have
    exactly one labelled fly (slot 0) and no unlabelled animal.

    The donor is pasted into every frame with the SAME 3D offset `D` (so
    `view_shifts`, frame-independent, is computed once) but its OWN
    per-frame pixels/mask/labels -- the pasted fly keeps the donor's own
    motion, not a pose frozen at frame 0.

    Returns a new sample dict, or `None` when: `src` has a different `T`
    than `tgt`; a camera valid in every target frame is one the donor lacks
    in any of its own frames; the shift pushes the donor's mask entirely off
    a target-valid camera that had donor mask content, in any frame; or the
    donor has no mask content in any target-valid camera in some frame
    (nothing to composite there, so its labels would claim a fly with zero
    pixel evidence).

    A pasted keypoint that scatters past the crop edge, or falls in a camera
    the donor mask never painted, is marked invisible per-view rather than
    rejecting the whole paste; one invisible in every view also loses its 3D
    label. These checks are per frame: a keypoint dead in frame 0 can still
    be alive in frame 1.
    """
    if (
        tgt["fly_valid"].shape[0] < 2
        or bool(tgt["fly_valid"][1])
        or int(tgt["unlabelled_sex"]) != SEX_UNKNOWN
    ):
        raise ValueError(
            "composite: target must have exactly one labelled fly and no unlabelled animal"
        )
    crops = tgt["crops"]
    T, C, H, W, _ = crops.shape
    if src["crops"].shape[0] != T:
        return None
    tv, sv = tgt["cam_valid"].all(0), src["cam_valid"].all(0)
    if np.any(tv & ~sv):
        return None
    shifts = view_shifts(tgt["M"], D, src["t_local"][0], tgt["t_local"][0])

    shifted_masks = [[None] * C for _ in range(T)]
    painted = np.zeros((T, C), bool)
    for ti in range(T):
        for c in range(C):
            if not tv[c]:
                continue
            src_mask_c = src["donor_mask"][ti, c]
            if not src_mask_c.any():
                continue
            m = _translate(src_mask_c.astype(np.uint8), shifts[c], nearest=True).astype(bool)
            if not m.any():
                return None
            shifted_masks[ti][c] = m
            painted[ti, c] = True
        if not painted[ti].any():
            return None

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tgt.items()}
    lo, hi = params.gain_clip
    for ti in range(T):
        kp2d_d = src["kp2d"][0, ti] + shifts[:, None, :]
        vis_d = src["vis2d"][0, ti].copy()
        inside = (kp2d_d >= 0).all(-1) & (kp2d_d[..., 0] <= W - 1) & (kp2d_d[..., 1] <= H - 1)
        for c in range(C):
            if not tv[c]:
                continue
            m = shifted_masks[ti][c]
            if m is None:
                continue
            donor = _translate(src["crops"][ti, c], shifts[c])
            gain = float(
                np.clip(
                    (np.median(crops[ti, c, ::4, ::4]) + 1.0)
                    / (np.median(src["crops"][ti, c, ::4, ::4]) + 1.0),
                    lo,
                    hi,
                )
            )
            donor = np.clip(donor.astype(np.float32) * gain, 0, 255)
            alpha = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]
            out["crops"][ti, c] = (crops[ti, c] * (1 - alpha) + donor * alpha).astype(np.uint8)
            hk = np.round(tgt["kp2d"][0, ti, c]).astype(int)
            ok = (hk[:, 0] >= 0) & (hk[:, 0] < W) & (hk[:, 1] >= 0) & (hk[:, 1] < H)
            covered = np.zeros(hk.shape[0], bool)
            covered[ok] = m[hk[ok, 1], hk[ok, 0]]
            out["vis2d"][0, ti, c] &= ~covered
            out["donor_mask"][ti, c] &= ~m
        final_vis = vis_d & inside & tv[:, None] & painted[ti][:, None]
        dead = vis_d.any(0) & ~final_vis.any(0)
        out["kp2d"][1, ti] = kp2d_d.astype(np.float32)
        out["vis2d"][1, ti] = final_vis
        out["has3d"][1, ti] = src["has3d"][0, ti] & ~dead
        out["kp3d_local"][1, ti] = (src["kp3d_local"][0, ti] + D) * out["has3d"][1, ti][:, None]
    out["fly_valid"][1] = True
    out["fly_sex"][1] = src["fly_sex"][0]
    out["unlabelled_sex"] = np.int8(SEX_UNKNOWN)
    return out
