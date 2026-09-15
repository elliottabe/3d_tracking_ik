#!/usr/bin/env python3
"""Does the keypoint filter help or hurt the rigid-bone invariant?

WHAT THIS ANSWERS. A rigid skeletal segment's length is CONSTANT -- that is the
invariant that caught this pipeline's keypoint-order bug when jitter, spike
rate, confidence and multi-view residual were all excellent. The temporal
filter (confidence mask -> bone-length reject -> spike removal -> PCHIP fill ->
Savitzky-Golay) exists to clean distal leg keypoints. This asks whether the
cleaned track is actually MORE rigid than the raw triangulation it replaced.

STATE THE EXPECTATION BEFORE READING THE FIGURE.
  * If the filter is doing its job, every bone's CV goes DOWN or stays flat
    between raw and filtered, on both flies, and the bars in panel 1 lean left.
  * If the filter's replacement values are worse than what they replaced, the
    female's bars lean RIGHT while the male's stay put -- she is the fly with
    something to replace (8601 confidence-masked keypoint-frames on bout 28 to
    his 0, 33066 interpolated coordinate-values to his 162).
  * Panel 2 is the discriminator. Split the filtered track by whether the
    filter ALTERED each pair: if the damage is in the replacements, untouched
    pairs sit at the raw CV and altered pairs sit far above it. If instead the
    whole track degraded, both halves move together.
  * A CV above WITHIN_BONE_CV_WARN_THRESH = 0.15 is the source's own "this bone
    is not behaving rigidly" line, drawn in panels 1 and 2.

WHY IT MATTERS. `bout_kp3d_paths` prefers kp3d_filt.npz, so body scale -- the
quantity behind this pipeline's 38x defect -- is estimated from the filtered
track. If filtering degrades rigidity, the scale is fit to the worse of the two
available measurements.

Usage:
    python scripts/viz/filter_rigidity_check.py \
        --bout <run_root>/bouts/bout_00028 \
        --out  figures/2026-09-11-filter-rigidity/bout28_rigidity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from tracking.io.names import Order
from tracking.preprocess.scale import WITHIN_BONE_CV_WARN_THRESH, rigid_segment_pairs

C_RAW, C_FILT = "#1f9ecf", "#e8833a"  # cyan = observed/detector, orange = derived


def _cv(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x) / np.mean(x)) if x.size > 10 else np.nan


def _len(kp: np.ndarray, ia: int, ib: int) -> np.ndarray:
    return np.linalg.norm(kp[:, ia] - kp[:, ib], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", type=Path, required=True)
    ap.add_argument("--anatomy", type=Path, default=Path("configs/anatomy/v1.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    order = Order(yaml.safe_load(args.anatomy.read_text())["model"]["KP_NAMES"])
    pairs = rigid_segment_pairs(order)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    summary: dict = {
        "bout": str(args.bout),
        "warn_thresh": WITHIN_BONE_CV_WARN_THRESH,
        "n_pairs": len(pairs),
        "flies": {},
    }

    # ---- panel 1: per-bone CV, raw vs filtered, by NAME, both flies ----------
    ax = axes[0]
    rows = []
    for fly, sex in ((0, "female"), (1, "male")):
        raw = np.load(args.bout / f"fly{fly}" / "kp3d.npz")["kp3d"]
        filt = np.load(args.bout / f"fly{fly}" / "kp3d_filt.npz")["kp3d"]
        cr, cf = [], []
        for a, b in pairs:
            ia, ib = order.index(a), order.index(b)
            cr.append(_cv(_len(raw, ia, ib)))
            cf.append(_cv(_len(filt, ia, ib)))
        cr, cf = np.array(cr), np.array(cf)
        summary["flies"][f"fly{fly}"] = {
            "sex": sex,
            "median_cv_raw": float(np.nanmedian(cr)),
            "median_cv_filtered": float(np.nanmedian(cf)),
            "n_bones_over_thresh_raw": int(np.nansum(cr > WITHIN_BONE_CV_WARN_THRESH)),
            "n_bones_over_thresh_filtered": int(np.nansum(cf > WITHIN_BONE_CV_WARN_THRESH)),
            "worst_bones_filtered": [
                {
                    "bone": f"{pairs[i][0]}->{pairs[i][1]}",
                    "cv_raw": float(cr[i]),
                    "cv_filtered": float(cf[i]),
                }
                for i in np.argsort(-np.nan_to_num(cf))[:5]
            ],
        }
        rows.append((fly, sex, cr, cf))

    y = np.arange(len(pairs))
    fly0 = rows[0]
    o = np.argsort(-np.nan_to_num(fly0[3]))
    ax.barh(y - 0.2, fly0[2][o], 0.4, color=C_RAW, label="raw (kp3d)")
    ax.barh(y + 0.2, fly0[3][o], 0.4, color=C_FILT, label="filtered (kp3d_filt)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{pairs[i][0]}→{pairs[i][1]}" for i in o], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(WITHIN_BONE_CV_WARN_THRESH, color="0.35", ls="--", lw=1)
    ax.text(
        WITHIN_BONE_CV_WARN_THRESH,
        len(pairs) - 0.5,
        " warn 0.15",
        fontsize=7,
        color="0.35",
        rotation=90,
        va="bottom",
    )
    ax.set_xlabel("within-bone length CV (dimensionless)")
    ax.set_title("fly0 (female) — rigid bones, by NAME")
    ax.legend(fontsize=8)

    # ---- panel 2: the discriminator -- untouched vs altered pairs ------------
    ax = axes[1]
    raw = np.load(args.bout / "fly0" / "kp3d.npz")["kp3d"]
    filt = np.load(args.bout / "fly0" / "kp3d_filt.npz")["kp3d"]
    unt, alt = [], []
    for a, b in pairs:
        ia, ib = order.index(a), order.index(b)
        same = (filt[:, ia] == raw[:, ia]).all(-1) & (filt[:, ib] == raw[:, ib]).all(-1)
        d = _len(filt, ia, ib)
        unt.append(_cv(d[same]))
        alt.append(_cv(d[~same]))
    unt, alt = np.array(unt), np.array(alt)
    summary["flies"]["fly0"]["median_cv_untouched_frames"] = float(np.nanmedian(unt))
    summary["flies"]["fly0"]["median_cv_altered_frames"] = float(np.nanmedian(alt))
    ax.scatter(unt, alt, c=C_FILT, s=28, zorder=3)
    lim = float(np.nanmax([np.nanmax(unt), np.nanmax(alt)])) * 1.1
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls=":", label="no change")
    ax.axhline(WITHIN_BONE_CV_WARN_THRESH, color="0.35", ls="--", lw=1)
    ax.axvline(WITHIN_BONE_CV_WARN_THRESH, color="0.35", ls="--", lw=1)
    ax.set_xlabel("CV on frames the filter LEFT ALONE")
    ax.set_ylabel("CV on frames the filter ALTERED")
    ax.set_title("fly0 — is the damage in the replacements?")
    ax.legend(fontsize=8, loc="lower right")

    # ---- panel 3: the worst bone, raw vs filtered, over time -----------------
    ax = axes[2]
    worst = int(np.argsort(-np.nan_to_num(fly0[3]))[0])
    a, b = pairs[worst]
    ia, ib = order.index(a), order.index(b)
    dr, df = _len(raw, ia, ib), _len(filt, ia, ib)
    # The WHOLE bout, not a window: the CV in the legend is a full-track number,
    # and the excursions that produce it are not all in the first few hundred
    # frames. A windowed plot under a full-track label is a figure that lies.
    ax.plot(dr, color=C_RAW, lw=0.7, label=f"raw (CV {_cv(dr):.3f})")
    ax.plot(df, color=C_FILT, lw=0.7, label=f"filtered (CV {_cv(df):.3f})")
    ax.set_xlabel(f"frame (all {len(dr)} of the bout)")
    ax.set_ylabel("segment length (0.1 mm world units)")
    ax.set_title(f"worst bone: {a}→{b}\na rigid segment's length must be CONSTANT")
    ax.legend(fontsize=8)
    summary["worst_bone"] = f"{a}->{b}"

    fig.suptitle(
        "Does the keypoint filter help the rigid-bone invariant?  "
        "cyan = raw triangulation, orange = filtered",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}\nwrote {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
