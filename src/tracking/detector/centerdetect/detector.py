"""CenterDetect inference (`CenterDetector`) and the greedy multi-view
lift from per-camera 2D peaks to 3D fly centres (`lift_peaks_to_centres`,
`cluster_centres`).

`lift_peaks_to_centres` takes a `geometry.rig.CameraRig` (addressed by camera
name) and uses `rig.matrices_f32` internally.

Pipeline, per frame:

  1. `CenterDetector.peaks(frames)` -- one CenterDetect forward pass per
     camera, top-2 peaks each, rescaled to full-image pixels.
     CenterDetect has NO cross-camera identity: camera A's "peak 0" and
     camera B's "peak 0" are not guaranteed to be the same animal, and a
     camera can be missing a peak (dim second blob below `min_score`) or
     have one moved by mask/BG noise. `lift_peaks_to_centres` must not
     assume peak *index* is animal identity.

  2. `lift_peaks_to_centres` resolves that with a greedy RANSAC-flavoured
     search: seed a 3D candidate from every (camera pair, peak pair)
     combination that is geometrically possible (DLT triangulation),
     reproject each candidate to every camera, and count INLIERS -- cameras
     whose reprojection lands within `max_resid_px` of *either* of that
     camera's two peaks. The candidate with the most inliers wins (ties
     broken by summed peak score), gets refined by a full DLT over its own
     inlier set, and those specific peaks are consumed (marked NaN) so the
     next iteration finds a DIFFERENT animal. Stops when the best remaining
     candidate has fewer than `min_views` inliers, or `max_animals` centres
     have been found.

  3. `cluster_centres` merges centres that are the same physical animal
     seen twice (e.g. a spurious extra candidate). Padding is always NaN
     (so a fixed-shape `(N, 3)` array survives however many animals were
     actually found), never a variable-length list -- this is what keeps
     row `i` naming the same animal slot across frames when one fly goes
     undetected (see `cluster_centres`'s own docstring).

Every distance/centre in this module is in the rig's world units and every
pixel is FULL-IMAGE px (not heatmap px, not crop px) unless named otherwise
-- see CLAUDE.md on labelling with real names and units.

CAMERA-ORDER CONTRACT: every per-camera axis in this module -- `peaks[c]`,
`scores[c]` and `rig.matrices_f32[c]` -- MUST be the same camera `c`, in
the rig's own (canonical) order. Nothing here re-sorts or matches by name;
a caller holding an array in a DIFFERENT camera order must permute it into
the rig's order FIRST. `lift_peaks_to_centres` checks the one thing it can
verify cheaply -- that `peaks` and the rig agree on camera COUNT -- and
raises `ValueError` on a mismatch; it cannot detect a same-length but
wrongly-ordered camera axis.
"""

from __future__ import annotations

import numpy as np

from tracking.detector.centerdetect.decode import peaks_from_heatmap
from tracking.detector.centerdetect.decode import preprocess as _centerdetect_preprocess
from tracking.detector.centerdetect.model import jit_centerdetect_forward, restore_centerdetect
from tracking.geometry.rig import CameraRig


class CenterDetector:
    """Thin CenterDetect inference wrapper: full frames -> top-2 peaks per
    camera in full-image px. GPU/checkpoint-heavy -- exercised in unit
    tests only via `preprocess`/the pure decode helpers on synthetic
    inputs, never via a real forward pass without a checkpoint.
    """

    def __init__(self, ckpt_dir, *, min_score: float = 0.2):
        self.min_score = float(min_score)
        self._model = restore_centerdetect(ckpt_dir)
        # Built ONCE here, never inside `peaks` (see `jit_centerdetect_forward`).
        self._forward = jit_centerdetect_forward(self._model)

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        """One full RGB frame -> the 320x320 CenterDetect model input.

        A static method (not a bound instance method) so it can be passed
        as `SlotReader(preprocess=CenterDetector.preprocess)` and run
        inside the per-camera decode threads instead of serially inside
        `peaks` -- measured at 1.87x on the mask-free pass.
        """
        return _centerdetect_preprocess(frame)

    def peaks(self, frames, pre=None):
        """frames: (C, H, W, 3) uint8 RGB full frames -- camera `c` here
        must be the SAME camera as the rig later passed to
        `lift_peaks_to_centres` (see module docstring's camera-order
        contract); this method itself has no camera identity of its own to
        check that against, so the contract is the caller's to keep.

        `pre`, when given, is the ALREADY-preprocessed `(C, 320, 320, 3)`
        model input for these same frames -- what a `SlotReader(preprocess=
        CenterDetector.preprocess)` computes inside the per-camera decode
        threads instead of serially here. `frames` is still required (and
        still supplies the ORIGINAL `img_w`/`img_h` that `peaks_from_heatmap`
        needs to undo the anisotropic squash), so a `pre` built from a
        different frame set cannot silently change the px scale.

        Returns:
            peaks: (C, 2, 2) float32 full-image px, NaN where score < min_score.
            scores: (C, 2) float32 raw confidence (unthresholded; see
                `peaks_from_heatmap`).
        """
        import jax.numpy as jnp

        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (C,H,W,3), got {frames.shape}")
        c, img_h, img_w = frames.shape[0], frames.shape[1], frames.shape[2]

        if pre is None:
            pre = np.stack([self.preprocess(frames[i]) for i in range(c)])
        else:
            pre = np.asarray(pre)
            # A per-camera COUNT mismatch is the one way a precomputed stack
            # can be wrong that is detectable here: it would pair camera i's
            # heatmap with camera j's image size downstream.
            if pre.shape[0] != c:
                raise ValueError(
                    f"pre has {pre.shape[0]} cameras but frames has {c} -- the "
                    f"precomputed CenterDetect input must come from the same "
                    f"reader call as `frames`"
                )
        x = jnp.asarray(pre)
        hm = np.asarray(self._forward(x))  # (C,160,160,1)
        return peaks_from_heatmap(hm, img_w, img_h, self.min_score)


def _triangulate_dlt_batched(points2d, cam_mats, valid):
    """Batched DLT triangulation. Invalid cameras contribute zero rows.

    Args:
        points2d: (M, C, 2) full-image px [u, v].
        cam_mats: (M, C, 4, 3) = P.T (P is the 3x4 DLT matrix).
        valid:    (M, C) bool.
    Returns:
        (M, 3) triangulated points (perspective-divided).
    """
    points2d = np.asarray(points2d, np.float64)
    cam_mats = np.asarray(cam_mats, np.float64)
    P = np.swapaxes(cam_mats, -1, -2)  # (M,C,3,4)
    u = points2d[..., 0:1]
    v = points2d[..., 1:2]  # (M,C,1)
    row_u = u * P[:, :, 2, :] - P[:, :, 0, :]  # (M,C,4)
    row_v = v * P[:, :, 2, :] - P[:, :, 1, :]  # (M,C,4)
    w = valid[..., None].astype(np.float64)  # (M,C,1)
    A = np.concatenate([row_u * w, row_v * w], axis=1)  # (M,2C,4)
    _, _, Vh = np.linalg.svd(A, full_matrices=False)
    X = Vh[:, -1, :]  # (M,4) null vec
    return (X[:, :3] / X[:, 3:4]).astype(np.float32)


def _seed_candidates(peaks, cam_mats):
    """All (camera pair x peak pair) triangulation candidates with both
    peaks non-NaN. Returns (M, 3) candidate 3D points, or an (0, 3) array if
    fewer than 2 cameras have any usable peak.

    Generic over the number of peak slots per camera (`peaks.shape[1]`) --
    CenterDetect always emits 2 (top-2 peaks), but this is not hardcoded so
    a caller with a single peak slot per camera (e.g. a known-count scene)
    still lifts correctly."""
    n_cams, n_slots = peaks.shape[0], peaks.shape[1]
    valid_peak = ~np.isnan(peaks).any(axis=-1)  # (C,n_slots)

    idx_i, idx_j, slot_i, slot_j = [], [], [], []
    for i in range(n_cams):
        for j in range(i + 1, n_cams):
            for pi in range(n_slots):
                if not valid_peak[i, pi]:
                    continue
                for pj in range(n_slots):
                    if not valid_peak[j, pj]:
                        continue
                    idx_i.append(i)
                    idx_j.append(j)
                    slot_i.append(pi)
                    slot_j.append(pj)

    m = len(idx_i)
    if m == 0:
        return np.zeros((0, 3), np.float32)

    points2d = np.zeros((m, n_cams, 2), np.float32)
    valid = np.zeros((m, n_cams), bool)
    for k in range(m):
        points2d[k, idx_i[k]] = peaks[idx_i[k], slot_i[k]]
        points2d[k, idx_j[k]] = peaks[idx_j[k], slot_j[k]]
        valid[k, idx_i[k]] = True
        valid[k, idx_j[k]] = True
    cam_mats_b = np.broadcast_to(cam_mats, (m,) + cam_mats.shape)
    return _triangulate_dlt_batched(points2d, cam_mats_b, valid)


def _project_batch(centres3d, cam_mats):
    """(N,3) world points, (C,4,3) cam_mats -> (N,C,2) full-image px."""
    M = np.asarray(cam_mats, np.float64)
    ph = np.concatenate(
        [np.asarray(centres3d, np.float64), np.ones((centres3d.shape[0], 1))], axis=1
    )
    proj = np.einsum("nd,cdk->nck", ph, M)
    return proj[..., :2] / proj[..., 2:3]


def _score_candidates(candidates, peaks, scores, cam_mats, max_resid_px):
    """For every candidate, count inlier cameras (nearest of that camera's
    two peaks within `max_resid_px`) and the summed score of the matched
    peaks. Returns (n_inliers (M,), summed_score (M,), best_slot (M,C),
    inlier (M,C) bool)."""
    proj = _project_batch(candidates, cam_mats)  # (M,C,2)
    diff = peaks[None, :, :, :] - proj[:, :, None, :]  # (M,C,2,2)
    dist = np.linalg.norm(diff, axis=-1)  # (M,C,2)
    dist_filled = np.where(np.isnan(dist), np.inf, dist)
    best_slot = np.argmin(dist_filled, axis=-1)  # (M,C)
    best_dist = np.take_along_axis(dist_filled, best_slot[..., None], axis=-1)[..., 0]
    inlier = best_dist <= max_resid_px  # (M,C)
    n_inliers = inlier.sum(axis=-1)  # (M,)

    m, n_cams = best_slot.shape
    n_slots = scores.shape[1]
    scores_b = np.broadcast_to(scores, (m, n_cams, n_slots))
    matched_score = np.take_along_axis(scores_b, best_slot[..., None], axis=-1)[..., 0]
    matched_score = np.where(np.isnan(matched_score), 0.0, matched_score)
    summed_score = np.where(inlier, matched_score, 0.0).sum(axis=-1)  # (M,)
    return n_inliers, summed_score, best_slot, inlier


def lift_peaks_to_centres(
    peaks,
    scores,
    rig: CameraRig,
    *,
    min_views: int = 3,
    max_resid_px: float = 25.0,
    max_animals: int = 2,
) -> np.ndarray:
    """CenterDetect peaks (no cross-camera identity) -> up to `max_animals`
    triangulated 3D centres, greedily, most-consistent animal first.

    Args:
        peaks: (C, 2, 2) float32 full-image px, NaN for a missing peak.
            `peaks[c]` MUST be the same camera as `rig.cameras[c]` -- see
            the module docstring's camera-order contract.
        scores: (C, 2) float32 RAW per-peak confidence (as returned by
            `peaks_from_heatmap`/`CenterDetector.peaks`, unthresholded). A
            peak being unusable is driven entirely by ITS COORDINATES being
            NaN in `peaks`; `scores` need not be NaN'd in lockstep (a NaN
            score is tolerated, treated as 0 when summing).
        rig: `CameraRig` -- `rig.matrices_f32` supplies the (C,4,3) DLT
            projection matrices, in `rig.cameras` order.
        min_views: stop once the best remaining candidate has fewer inlier
            cameras than this (default 3 -- triangulation needs >=2, this
            requires one spare to reject a single bad view).
        max_resid_px: reprojection distance below which a camera counts as
            an inlier for a candidate centre.
        max_animals: stop after finding this many centres.

    Returns:
        centres: (max_animals, 3) float32, NaN-padded.

    Raises:
        ValueError: if `peaks` and the rig disagree on camera count -- the
            one camera-order mismatch this function can detect (it cannot
            tell a same-length but wrongly-ORDERED camera axis; that is on
            the caller, per the module docstring).
    """
    peaks = np.array(peaks, dtype=np.float32, copy=True)
    scores = np.array(scores, dtype=np.float32, copy=True)
    cam_mats = np.asarray(rig.matrices_f32, np.float32)
    if peaks.shape[0] != cam_mats.shape[0]:
        raise ValueError(
            f"peaks camera axis ({peaks.shape[0]}) must match the rig's camera "
            f"axis ({cam_mats.shape[0]}) -- peaks[c]/scores[c] and rig.cameras[c] "
            f"must be the SAME camera, in the rig's own order (see module "
            f"docstring's camera-order contract)."
        )
    n_cams = peaks.shape[0]

    out_centres = np.full((max_animals, 3), np.nan, np.float32)

    for a in range(max_animals):
        candidates = _seed_candidates(peaks, cam_mats)
        if candidates.shape[0] == 0:
            break
        n_inliers, summed_score, best_slot, inlier = _score_candidates(
            candidates, peaks, scores, cam_mats, max_resid_px
        )
        best = np.lexsort((-summed_score, -n_inliers))[0]
        if n_inliers[best] < min_views:
            break

        inlier_cams = np.flatnonzero(inlier[best])
        refine_points = np.zeros((n_cams, 2), np.float32)
        refine_valid = np.zeros((n_cams,), bool)
        for c_idx in inlier_cams:
            slot = best_slot[best, c_idx]
            refine_points[c_idx] = peaks[c_idx, slot]
            refine_valid[c_idx] = True
        refined = _triangulate_dlt_batched(refine_points[None], cam_mats[None], refine_valid[None])[
            0
        ]

        out_centres[a] = refined

        for c_idx in inlier_cams:
            peaks[c_idx, best_slot[best, c_idx]] = np.nan
            scores[c_idx, best_slot[best, c_idx]] = np.nan

    return out_centres


def cluster_centres(centres, *, min_sep_units: float = 15.0) -> np.ndarray:
    """Merge centres closer than `min_sep_units` by repeatedly averaging the
    closest pair below threshold. NaN-padded back to the input length so the
    shape never depends on how many merges happened.

    NaN-padding the output to a fixed `(N, 3)` shape -- rather than
    returning only the surviving/merged rows -- matters beyond "shape
    stays constant": this is what keeps row `i` meaning the same animal
    slot across frames in the common partial-detection case (one of two
    flies not found in a frame, so its row is NaN). A shrinking output
    would silently shift every later slot down by one, which downstream
    is indistinguishable from a fly-identity swap.

    Args:
        centres: (N, 3) float32, may already contain NaN rows (ignored,
            i.e. not merged with anything -- but still occupying a slot in
            the output).
    Returns:
        (N, 3) float32, NaN-padded.
    """
    centres = np.asarray(centres, np.float32)
    n = centres.shape[0]
    pts = [centres[i].copy() for i in range(n) if not np.isnan(centres[i]).any()]

    merged = True
    while merged and len(pts) > 1:
        merged = False
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < min_sep_units and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is not None:
            _, i, j = best
            new_pt = (pts[i] + pts[j]) / 2.0
            pts = [p for k, p in enumerate(pts) if k not in (i, j)] + [new_pt]
            merged = True

    out = np.full((n, 3), np.nan, np.float32)
    for k, p in enumerate(pts):
        out[k] = p
    return out
