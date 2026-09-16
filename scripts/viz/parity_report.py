#!/usr/bin/env python3
"""Bout-28 parity figure: is the port's residual a real divergence, or run-to-run noise?

Usage:
    python scripts/viz/parity_report.py \
        --arrays figures/2026-09-10-parity/bout28_parity_arrays.npz \
        --out    figures/2026-09-10-parity/bout28_parity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

# Headless: the backend must be chosen before pyplot is imported, so the
# pyplot import deliberately sits below this call in its own import block.
matplotlib.use("Agg")

from matplotlib import pyplot as plt

LOO_FLOOR_PX = 0.81
PX_PER_UNIT = 0.018 / 0.0029575
LOO_FLOOR_UNITS = LOO_FLOOR_PX / PX_PER_UNIT  # ~0.133 world units
CONTACT_ABS_FRAME = 447800
# viz.core.colors' shared language: cyan = observed/detector arm, orange = fly1.
C_NEW_OLD = "#1f9ecf"  # cyan   -- port vs reference (what the gate measures)
C_NEW_NEW = "#e8833a"  # orange -- port vs port     (what the pipeline owes itself)


def _pair_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T,K,3) vs (T,K,3) -> (T,K) euclidean distance, NaN where either is NaN."""
    d = np.linalg.norm(a - b, axis=-1)
    both = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    return np.where(both, d, np.nan)


def _nanmed(x, axis=None):
    with np.errstate(all="ignore"):
        return np.nanmedian(x, axis=axis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fly", type=int, default=0, help="0 = female, 1 = male")
    args = ap.parse_args()

    z = np.load(args.arrays, allow_pickle=True)
    f = args.fly
    kp_names = [str(s) for s in z["kp_names"]]
    start = int(z["bout_start_frame"])

    ref, run1 = z[f"ref_fly{f}"], z[f"run1_fly{f}"]
    has_run2 = f"run2_fly{f}" in z.files
    d_new_old = _pair_diff(run1, ref)
    d_new_new = _pair_diff(run1, z[f"run2_fly{f}"]) if has_run2 else None

    T = d_new_old.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    sex = "female" if f == 0 else "male"
    fig.suptitle(
        f"Bout 28 parity — fly{f} ({sex}), frames {start}..{start + T - 1}  |  "
        f"cyan = port vs reference run   orange = port vs port (same code, fresh process)",
        fontsize=12,
    )

    # ---- panel 1: per-keypoint median, BY NAME -----------------------------
    ax = axes[0]
    med_old = _nanmed(d_new_old, axis=0)
    order = np.argsort(np.nan_to_num(med_old, nan=-1))[::-1]
    y = np.arange(len(order))
    ax.barh(y, med_old[order], color=C_NEW_OLD, height=0.8, label="port vs reference")
    if d_new_new is not None:
        ax.barh(
            y, _nanmed(d_new_new, axis=0)[order], color=C_NEW_NEW, height=0.45, label="port vs port"
        )
    ax.set_yticks(y)
    ax.set_yticklabels([kp_names[i] for i in order], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("median |difference| per keypoint (world units)")
    ax.set_title("Where the error sits — by keypoint NAME")
    ax.axvline(LOO_FLOOR_UNITS, color="0.35", ls="--", lw=1)
    ax.legend(loc="lower right", fontsize=8)

    # ---- panel 2: per-frame median across the bout -------------------------
    ax = axes[1]
    frames = np.arange(T) + start
    ax.plot(frames, _nanmed(d_new_old, axis=1), color=C_NEW_OLD, lw=1.0, label="port vs reference")
    if d_new_new is not None:
        ax.plot(frames, _nanmed(d_new_new, axis=1), color=C_NEW_NEW, lw=1.0, label="port vs port")
    if start <= CONTACT_ABS_FRAME < start + T:
        ax.axvline(CONTACT_ABS_FRAME, color="0.35", ls=":", lw=1.2)
        ax.text(
            CONTACT_ABS_FRAME,
            ax.get_ylim()[1],
            "  contact stretch",
            fontsize=8,
            va="top",
            color="0.35",
        )
    ax.axhline(LOO_FLOOR_UNITS, color="0.35", ls="--", lw=1)
    ax.set_xlabel("absolute frame")
    ax.set_ylabel("median |difference| over keypoints (world units)")
    ax.set_title("FLAT = non-accumulating   GROWING = accumulating drift")
    ax.legend(fontsize=8)

    # ---- panel 3: CDF, against the LOO floor -------------------------------
    ax = axes[2]
    for d, c, lab in (
        (d_new_old, C_NEW_OLD, "port vs reference"),
        (d_new_new, C_NEW_NEW, "port vs port"),
    ):
        if d is None:
            continue
        v = np.sort(d[np.isfinite(d)].ravel())
        if v.size and not np.any(v > 0):
            # An all-zero curve has no place on a log axis -- it would silently
            # vanish and read as "not plotted". Say so in the legend instead.
            ax.plot([], [], color=c, lw=1.6, label=f"{lab} (identically 0)")
            continue
        ax.plot(v, np.linspace(0, 1, v.size), color=c, lw=1.6, label=lab)
    ax.axvline(LOO_FLOOR_UNITS, color="0.35", ls="--", lw=1)
    ax.text(
        LOO_FLOOR_UNITS,
        0.02,
        f" LOO floor {LOO_FLOOR_PX} px",
        fontsize=8,
        color="0.35",
        rotation=90,
    )
    ax.set_xscale("log")
    ax.set_xlabel("|difference| (world units, log scale)")
    ax.set_ylabel("fraction of (frame, keypoint) pairs below")
    ax.set_title("Residual vs the pipeline's own LOO noise floor")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)

    summary = {
        "fly": f,
        "sex": sex,
        "n_frames": int(T),
        "bout_start_frame": start,
        "new_vs_old": {
            "median": float(_nanmed(d_new_old)),
            "p90": float(np.nanpercentile(d_new_old, 90)),
            "max": float(np.nanmax(d_new_old)),
            "frac_above_loo_floor": float(np.nanmean(d_new_old > LOO_FLOOR_UNITS)),
        },
        "worst_keypoints_new_vs_old": [
            {"name": kp_names[i], "median": float(med_old[i])} for i in order[:8]
        ],
    }
    if d_new_new is not None:
        summary["new_vs_new"] = {
            "median": float(_nanmed(d_new_new)),
            "p90": float(np.nanpercentile(d_new_new, 90)),
            "max": float(np.nanmax(d_new_new)),
            "frac_above_loo_floor": float(np.nanmean(d_new_new > LOO_FLOOR_UNITS)),
        }
        floor = summary["new_vs_new"]["median"]
        resid = summary["new_vs_old"]["median"]
        if floor > 0:
            summary["median_ratio_old_over_new"] = float(resid / floor)
        if floor == 0.0 and summary["new_vs_new"]["max"] == 0.0:
            # A zero floor is not a divergence signal -- it means the pass is
            # deterministic, so the residual must be judged on MAGNITUDE.
            summary["reading"] = (
                "pass is deterministic (new-vs-new bit-identical); residual is "
                "systematic. "
                + (
                    f"Magnitude {resid:.4g} units is below the pipeline's own LOO "
                    f"floor ({LOO_FLOOR_UNITS:.3f} units = {LOO_FLOOR_PX} px) -- "
                    "physically negligible; the px gate in test_mvq_parity.py decides."
                    if resid < LOO_FLOOR_UNITS
                    else f"Magnitude {resid:.4g} units EXCEEDS the LOO floor "
                    f"({LOO_FLOOR_UNITS:.3f} units) -- investigate further."
                )
            )
        elif resid / max(floor, 1e-30) < 2.0:
            summary["reading"] = "residual is within run-to-run nondeterminism"
        else:
            summary["reading"] = (
                "residual exceeds the pipeline's self-consistency; investigate further"
            )
    else:
        summary["reading"] = "run2 absent -- new-vs-new not measured, verdict withheld"

    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}\nwrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
