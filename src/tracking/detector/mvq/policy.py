"""How a real mvq inference call picks ONE instance slot, and how that choice
is scored against the host mask (P3a spec §7).

Both live here because three call sites need them to be IDENTICAL and used to
carry their own inline copies that drifted: `train/train_mvq.py::evaluate`
(the numbers the acceptance gates read), `scripts/viz/mvq_overlay.py` (the
figure those numbers are checked against) and
`scripts/benchmark/mvq_val_baselines.py` (the like-for-like baseline table).
A policy difference between the figure and the metric is exactly the kind of
disagreement that makes a plot "confirm" a number it never measured.

Numpy in, plain Python out -- no jit, one sample at a time.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tracking.detector.mvq.geometry import project_local

# Slot table, fixed by the P3a spec §3:
# 0 = prompted, 1 = female, 2 = male, 3 = other.
SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER = 0, 1, 2, 3
N_SLOTS = 4

EXIST_THRESH = 0.5


def typed_candidates(n_instances: int) -> list[int]:
    """The slots the UNPROMPTED policy is allowed to pick from.

    With the P3a slot table (`n_instances == N_SLOTS == 4`: 0 prompted, 1
    female, 2 male, 3 other) slot 0 is never a candidate unprompted -- its
    existence target is exactly `prompt_on`, so an unprompted window's slot 0
    is trained to NOT exist and picking it would be picking a slot the loss
    told to be empty. A model with any other slot count is a LEGACY,
    untyped one (e.g. the 3-slot 30k run) whose slots have no fixed meaning,
    so all of them are candidates."""
    return list(range(1, n_instances)) if n_instances == N_SLOTS else list(range(n_instances))


def policy_instance(
    exist_probs, xyz, *, prompted: bool, has_mask: bool, exist_thresh: float = EXIST_THRESH
):
    """The instance a real inference call would report -- no ground truth.

    `exist_probs` (I,): sigmoid of the existence head. `xyz` (I,T,K,3):
    ROI-LOCAL predicted keypoints, so the ROI origin is (0,0,0).

    Prompted AND the window has a usable mask -> slot 0, which the prompt
    token targets. Otherwise (unprompted, or prompted with no usable mask):
    among `typed_candidates` that clear `exist_thresh`, the one whose mean
    predicted centroid over (T,K) sits nearest the ROI origin -- the host fly
    is cropped to be near that centre by construction
    (`data/v12_windows.py`), so this is the cheapest correct proxy for "which
    instance is the host" without seeing labels. Returns None (a MISS) when
    no candidate clears the threshold: there is nothing to score, so the
    sample is excluded from the policy MPJPE and counted in
    `policy_miss_frac` instead.

    `exist_thresh` defaults to the module constant, which is what every
    metric-producing call site passes (the acceptance numbers, the overlay
    figure and the val baselines must all threshold identically). A caller
    that runs the model at a DIFFERENT threshold -- `tracking/lift_mvq.py`'s
    `MVQRunner(exist_thresh=...)`, whose `read_typed` gates on its own value
    -- passes it here too, or the policy would hand back the very slot the
    typed read just refused.
    """
    if prompted and has_mask:
        return SLOT_PROMPTED
    exist_probs = np.asarray(exist_probs)
    xyz = np.asarray(xyz)
    cand = np.array(
        [s for s in typed_candidates(xyz.shape[0]) if exist_probs[s] >= exist_thresh], dtype=int
    )
    if cand.size == 0:
        return None
    return int(cand[np.argmin(np.linalg.norm(xyz[cand].mean(axis=(1, 2)), axis=-1))])


def mask_containment(xyz_kp, M, t_local, prompt_mask, cam_valid, vis):
    """Fraction of an instance's reprojected VISIBLE keypoints that land
    inside the host's mask -- the direct measure of cross-fly mixing (spec
    §7): a slot whose points wander onto the other fly's body scores low
    however small its MPJPE against whichever body it landed on.

    `xyz_kp` (K,3) ROI-local; `M` (C,2,3), `t_local` (C,2) this frame's
    camera geometry; `prompt_mask` (C,H,W) bool; `cam_valid` (C,) bool;
    `vis` (C,K) bool = the HOST's labelled visibility. Cameras that are
    invalid or whose mask is empty contribute nothing (an empty mask would
    otherwise score 0 for every keypoint and drag the mean down for a reason
    that has nothing to do with mixing). `nan` when no camera qualifies.
    """
    uv = np.asarray(
        project_local(jnp.asarray(xyz_kp), jnp.asarray(M), jnp.asarray(t_local))
    )  # (K,C,2)
    pm = np.asarray(prompt_mask)
    hits = []
    for c in range(pm.shape[0]):
        if not bool(np.asarray(cam_valid)[c]) or not pm[c].any():
            continue
        v = np.asarray(vis)[c]
        p = np.round(uv[v, c]).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < pm.shape[2]) & (p[:, 1] >= 0) & (p[:, 1] < pm.shape[1])
        inside = np.zeros(len(p), bool)
        inside[ok] = pm[c][p[ok, 1], p[ok, 0]]
        hits.extend(inside.tolist())
    return float(np.mean(hits)) if hits else float("nan")
