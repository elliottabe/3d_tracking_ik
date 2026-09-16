"""Are the fitted WINGS right? Marker residual cannot answer this.

Usage:
    python scripts/viz/wing_check.py --fly 1 --out figures/<date>-wing-check
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

REF = (
    "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/"
    "2025_10_20_13_20_04/pose_maskfree_v2"
)
UNITS_TO_MM = 0.1
VEINS = [("base", "V12"), ("V12", "V13"), ("base", "V13")]
DOFS = ["wing_yaw", "wing_roll", "wing_pitch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(f"{REF}/offsets_fly{args.fly}.h5", "r") as f:
        sites = np.asarray(f["marker_sites"][:], np.float64)
        kp = np.asarray(f["kp_data"][:], np.float64).reshape(len(f["kp_data"]), -1, 3)
        kp_names = [n.decode() for n in f["kp_names"][:]]
        qpos = np.asarray(f["qpos"][:], np.float64)
        names_qpos = [n.decode() for n in f["names_qpos"][:]]
    scale = json.loads(Path(f"{REF}/scale.json").read_text())["scale_by_fly"][str(args.fly)]
    to_mm = UNITS_TO_MM / scale
    idx = {n: i for i, n in enumerate(kp_names)}
    finite = np.isfinite(kp).all(-1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    sex = "female" if args.fly == 0 else "MALE"
    summary: dict = {"fly": args.fly}

    # 1 -- residual per wing keypoint, L vs R
    ax = axes[0, 0]
    wing_kps = [n for n in kp_names if n.startswith("Wing")]
    res = {}
    for n in wing_kps:
        i = idx[n]
        d = np.full(len(kp), np.nan)
        d[finite[:, i]] = np.linalg.norm((sites - kp)[finite[:, i], i], axis=-1) * to_mm
        res[n] = d
        ax.plot(d, lw=0.8, label=f"{n} ({np.nanmedian(d):.3f} mm)")
    body = np.nanmedian(
        [
            np.linalg.norm((sites - kp)[finite[:, i], i], axis=-1) * to_mm
            for i in range(len(kp_names))
            if not kp_names[i].startswith("Wing")
        ][0]
    )
    ax.axhline(body, color="0.4", ls=":", label=f"a body keypoint ({body:.3f} mm)")
    ax.set_xlabel("frame")
    ax.set_ylabel("fitted-vs-observed (mm)")
    ax.set_title(f"fly{args.fly} ({sex}) -- wing marker residual")
    ax.legend(fontsize=7)
    summary["wing_residual_mm"] = {n: float(np.nanmedian(d)) for n, d in res.items()}

    # 2 -- vein lengths, fitted vs observed: the collapsed-wing check
    ax = axes[0, 1]
    labels, fitted, observed = [], [], []
    for side in ("L", "R"):
        for a, b in VEINS:
            na, nb = f"Wing{side}_{a}", f"Wing{side}_{b}"
            if na not in idx or nb not in idx:
                continue
            ok = finite[:, idx[na]] & finite[:, idx[nb]]
            if ok.sum() < 10:
                continue
            labels.append(f"{side}: {a}->{b}")
            fitted.append(
                float(np.median(np.linalg.norm(sites[:, idx[na]] - sites[:, idx[nb]], axis=-1)))
                * to_mm
            )
            observed.append(
                float(np.median(np.linalg.norm(kp[ok][:, idx[na]] - kp[ok][:, idx[nb]], axis=-1)))
                * to_mm
            )
    y = np.arange(len(labels))
    ax.barh(y - 0.2, observed, height=0.4, color="#4dabf7", label="observed")
    ax.barh(y + 0.2, fitted, height=0.4, color="#2b8a3e", label="fitted")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("vein length (mm)")
    ax.set_title("wing veins: a collapsed wing shows up HERE, not in the residual")
    ax.legend(fontsize=8)
    summary["vein_fitted_over_observed"] = {
        lab: (f / o if o > 0 else None) for lab, f, o in zip(labels, fitted, observed, strict=False)
    }

    # 3 -- the three wing DOFs over time
    ax = axes[1, 0]
    for side, style in (("left", "-"), ("right", "--")):
        for dof, colour in zip(DOFS, ("#1c7ed6", "#2b8a3e", "#e8590c"), strict=False):
            name = f"{dof}_{side}"
            if name not in names_qpos:
                continue
            ax.plot(
                np.degrees(qpos[:, names_qpos.index(name)]),
                style,
                lw=0.9,
                color=colour,
                alpha=0.85,
                label=name,
            )
    ax.axhline(-57.3, color="0.5", ls=":", lw=1)
    ax.text(
        0.01,
        -57.3,
        " pitch spring rest",
        fontsize=7,
        color="0.4",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlabel("frame")
    ax.set_ylabel("angle (deg)")
    ax.set_title("wing DOFs -- pitch pinned at rest means the solver never found it")
    ax.legend(fontsize=7, ncol=2)

    inset = ax.inset_axes([0.52, 0.06, 0.46, 0.38])
    lo, hi = len(qpos) // 2, len(qpos) // 2 + 60
    for side, style in (("left", "-"), ("right", "--")):
        for dof, colour in zip(DOFS, ("#1c7ed6", "#2b8a3e", "#e8590c"), strict=False):
            name = f"{dof}_{side}"
            if name in names_qpos:
                inset.plot(
                    range(lo, hi),
                    np.degrees(qpos[lo:hi, names_qpos.index(name)]),
                    style,
                    lw=0.9,
                    color=colour,
                    alpha=0.9,
                )
    inset.set_title(f"frames {lo}-{hi}", fontsize=7)
    inset.tick_params(labelsize=6)

    # 4 -- how often each wing DOF sits at a limit
    ax = axes[1, 1]
    names, fracs = [], []
    for side in ("left", "right"):
        for dof in DOFS:
            name = f"{dof}_{side}"
            if name not in names_qpos:
                continue
            v = qpos[:, names_qpos.index(name)]
            at_rest = (
                float(np.mean(np.abs(np.degrees(v) - (-57.3)) < 0.5)) if "pitch" in dof else np.nan
            )
            names.append(name)
            fracs.append(float(np.mean(np.abs(np.degrees(v)) > 85.0)))
            if "pitch" in dof:
                summary.setdefault("pitch_fraction_at_spring_rest", {})[name] = at_rest
    ax.barh(names, fracs, color="#7048e8")
    ax.invert_yaxis()
    ax.set_xlabel("fraction of frames with |angle| > 85 deg (near a stop)")
    ax.set_title("yaw's stop IS its rest pose -- a folded wing, not a defect")
    ax.set_xlim(0, 1)
    summary["fraction_near_stop"] = dict(zip(names, fracs, strict=False))

    fig.tight_layout()
    fig.savefig(out / f"wing_check_fly{args.fly}.png", dpi=140)
    (out / f"wing_check_fly{args.fly}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
