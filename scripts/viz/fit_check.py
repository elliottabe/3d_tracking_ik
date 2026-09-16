"""Does the STAC fit actually land on the fly? Four panels, on real frames.

Usage:
    python scripts/viz/fit_check.py --fly 0 --out figures/<date>-fit-check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from tracking.viz.colors import leg_chains  # noqa: E402

REF = (
    "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/"
    "2025_10_20_13_20_04/pose_maskfree_v2"
)
UNITS_TO_MM = 0.1


def rigid_pairs(kp_names):
    """Adjacent pairs within each leg chain -- segments whose length is fixed
    by the skeleton and must therefore be constant over time."""
    pairs = []
    for chain in leg_chains(list(kp_names)):
        pairs += list(zip(chain[:-1], chain[1:], strict=False))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3, help="skeleton panels to draw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{REF}/offsets_fly{args.fly}.h5", "r") as f:
        sites = np.asarray(f["marker_sites"][:], np.float64)  # (T, K, 3) model units
        kp_flat = np.asarray(f["kp_data"][:], np.float64)
        kp_names = [n.decode() for n in f["kp_names"][:]]
    kp = kp_flat.reshape(len(kp_flat), -1, 3)
    scale = json.loads(Path(f"{REF}/scale.json").read_text())["scale_by_fly"][str(args.fly)]
    # model units -> world units -> mm
    to_mm = UNITS_TO_MM / scale

    finite = np.isfinite(kp).all(-1)
    resid = np.full(kp.shape[:2], np.nan)
    resid[finite] = np.linalg.norm((sites - kp)[finite], axis=-1) * to_mm

    fig = plt.figure(figsize=(16, 9))
    sex = "female" if args.fly == 0 else "MALE"

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(np.nanmedian(resid, axis=1), lw=1, color="#2b8a3e", label="median")
    ax.plot(np.nanpercentile(resid, 95, axis=1), lw=0.8, color="#e8590c", label="p95")
    ax.set_xlabel("frame of the fit sample")
    ax.set_ylabel("fitted-vs-observed (mm)")
    ax.set_title(f"fly{args.fly} ({sex}) -- per-frame marker residual")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    per_kp = np.nanmedian(resid, axis=0)
    order = np.argsort(-per_kp)
    ax.barh([kp_names[i] for i in order[:18]], per_kp[order[:18]], color="#e8590c")
    ax.invert_yaxis()
    ax.set_xlabel("median residual (mm)")
    ax.set_title("worst 18 keypoints, by name")
    ax.tick_params(axis="y", labelsize=6)

    ax = fig.add_subplot(2, 2, 3)
    chains = leg_chains(kp_names)
    idx = {n: i for i, n in enumerate(kp_names)}
    t = len(kp) // 2
    for chain in chains:
        ii = [idx[n] for n in chain if n in idx]
        ax.plot(kp[t, ii, 0], kp[t, ii, 1], "-o", ms=2.5, lw=1.0, color="#4dabf7", alpha=0.9)
        ax.plot(sites[t, ii, 0], sites[t, ii, 1], "-o", ms=2.5, lw=1.0, color="#2b8a3e", alpha=0.9)
    ax.plot([], [], color="#4dabf7", label="observed keypoints")
    ax.plot([], [], color="#2b8a3e", label="fitted marker sites")
    ax.set_aspect("equal")
    ax.set_title(f"skeleton, frame {t} (x-y, model units)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 4)
    pairs = rigid_pairs(kp_names)
    ratios = []
    for a, b in pairs:
        if a not in idx or b not in idx:
            continue
        fit_len = np.linalg.norm(sites[:, idx[a]] - sites[:, idx[b]], axis=-1) * to_mm
        ok = finite[:, idx[a]] & finite[:, idx[b]]
        if ok.sum() < 10:
            continue
        obs_len = np.linalg.norm(kp[ok][:, idx[a]] - kp[ok][:, idx[b]], axis=-1) * to_mm
        mo = float(np.median(obs_len))
        if mo <= 0:
            continue
        ratios.append((f"{a}->{b}", float(np.median(fit_len)) / mo, mo))
    ratios.sort(key=lambda r: -abs(r[1] - 1.0))
    names = [n for n, _, _ in ratios[:18]]
    vals = [v for _, v, _ in ratios[:18]]
    ax.barh(names, vals, color=["#e8590c" if abs(v - 1) > 0.10 else "#7048e8" for v in vals])
    ax.invert_yaxis()
    ax.axvline(1.0, color="0.3", lw=1)
    ax.axvspan(0.9, 1.1, color="0.85", zorder=0, label="within 10%")
    lo = min(0.88, min(vals) - 0.02) if vals else 0.88
    hi = max(1.12, max(vals) + 0.02) if vals else 1.12
    ax.set_xlim(lo, hi)  # every ratio is near 1; a bar from 0 hides the spread
    ax.set_xlabel("fitted segment length / observed segment length")
    ax.set_title("does the fitted skeleton match the animal?")
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / f"fit_check_fly{args.fly}.png", dpi=140)

    summary = dict(
        fly=args.fly,
        residual_median_mm=float(np.nanmedian(resid)),
        residual_p95_mm=float(np.nanpercentile(resid, 95)),
        residual_max_mm=float(np.nanmax(resid)),
        worst_keypoints=[[kp_names[i], float(per_kp[i])] for i in order[:5]],
        worst_length_ratio=[[n, v, mo] for n, v, mo in ratios[:5]],
        segments_outside_10pct=int(sum(1 for _, v, _ in ratios if abs(v - 1) > 0.10)),
        n_segments=len(ratios),
    )
    (out / f"fit_check_fly{args.fly}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
