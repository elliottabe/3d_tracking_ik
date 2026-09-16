"""How a real mvq inference call picks ONE instance slot, and how that choice
is scored against the host mask (P3a spec §7).
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
    """The slots the UNPROMPTED policy is allowed to pick from."""
    return list(range(1, n_instances)) if n_instances == N_SLOTS else list(range(n_instances))


def policy_instance(
    exist_probs, xyz, *, prompted: bool, has_mask: bool, exist_thresh: float = EXIST_THRESH
):
    """The instance a real inference call would report -- no ground truth."""
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
