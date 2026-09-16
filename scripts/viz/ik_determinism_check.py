"""Compare two `stac_ik.h5` solves of the same bout-fly: what is reproducible?

Usage:
    python scripts/viz/ik_determinism_check.py --a REF.h5 --b RERUN.h5 \
        --out figures/<date>-ik-determinism
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


def _read(path):
    with h5py.File(path, "r") as f:
        return dict(
            qpos=np.asarray(f["qpos"][:], np.float64),
            start=np.asarray(f["ik_start_idx"][:]),
            costs=np.asarray(f["candidate_costs"][:], np.float64),
            names_qpos=[n.decode() for n in f["names_qpos"][:]],
        )


def compare(a, b):
    solved = np.isfinite(a["qpos"]).all(1) & np.isfinite(b["qpos"]).all(1)
    d = np.abs(b["qpos"] - a["qpos"])
    per_frame = np.where(solved, d.max(1), np.nan)
    agree = solved & (a["start"] == b["start"])
    flip = solved & (a["start"] != b["start"])
    t = np.arange(len(solved))
    cost_a = a["costs"][a["start"].clip(0), t]
    cost_b = b["costs"][b["start"].clip(0), t]
    return dict(
        solved=solved,
        per_frame=per_frame,
        agree=agree,
        flip=flip,
        d=d,
        cost_a=cost_a,
        cost_b=cost_b,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference stac_ik.h5")
    ap.add_argument("--b", required=True, help="re-run stac_ik.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="reference")
    ap.add_argument("--label-b", default="re-run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a, b = _read(args.a), _read(args.b)
    c = compare(a, b)
    solved, flip, agree = c["solved"], c["flip"], c["agree"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    pf = c["per_frame"][solved]
    zero = int((pf == 0.0).sum())
    nz = pf[pf > 0]
    ax[0].bar(["exactly 0.0", "> 0"], [zero, int(nz.size)], color=["#2b8a3e", "#e8590c"])
    for i, v in enumerate([zero, int(nz.size)]):
        ax[0].text(i, v, f"{v}\n({v / solved.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax[0].set_ylim(0, max(zero, int(nz.size), 1) * 1.18)
    ax[0].set_ylabel("frames")
    span = f"{nz.min():.2g} – {nz.max():.2g} rad" if nz.size else "none"
    ax[0].set_xlabel(f"per-frame max |Δqpos|          non-zero lobe spans {span}")
    bimodal = zero > 0.5 * solved.sum()
    decades = np.log10(nz.max() / nz.min()) if nz.size else 0.0
    ax[0].set_title(
        f"Reproducibility is bimodal\n{zero}/{solved.sum()} frames bit-identical"
        if bimodal
        else f"NOT reproducible to the bit\n{zero}/{solved.sum()} bit-identical; "
        f"the rest spread over {decades:.0f} decades"
    )

    # 2. the discriminator: does agreement on the candidate explain it?
    ax[1].scatter(
        np.flatnonzero(agree),
        c["per_frame"][agree],
        s=6,
        color="#2b8a3e",
        label=f"same candidate ({agree.sum()})",
    )
    ax[1].scatter(
        np.flatnonzero(flip),
        c["per_frame"][flip],
        s=14,
        color="#e8590c",
        label=f"different candidate ({flip.sum()})",
    )
    ax[1].set_yscale("symlog", linthresh=1e-6)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("max |Δqpos|")
    agree_max = float(np.nanmax(c["per_frame"][agree])) if agree.sum() else 0.0
    ax[1].set_title(
        "Every difference is a candidate flip"
        if agree_max == 0.0
        else f"Differences are NOT only candidate flips\n"
        f"same-candidate frames reach {agree_max:.2g} rad"
    )
    ax[1].legend(fontsize=8)

    # 3. are the flipped frames actually near-degenerate?
    if flip.sum():
        lo = min(c["cost_a"][flip].min(), c["cost_b"][flip].min())
        hi = max(c["cost_a"][flip].max(), c["cost_b"][flip].max())
        ax[2].plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=0)
        ax[2].scatter(c["cost_a"][flip], c["cost_b"][flip], s=16, color="#e8590c")
        ax[2].set_xscale("log")
        ax[2].set_yscale("log")
        better = int((c["cost_b"][flip] < c["cost_a"][flip]).sum())
        ax[2].set_title(
            f"Flipped frames cost the same either way\n"
            f"{args.label_b} better on {better}/{int(flip.sum())}"
        )
    ax[2].set_xlabel(f"chosen marker cost — {args.label_a}")
    ax[2].set_ylabel(f"chosen marker cost — {args.label_b}")

    fig.tight_layout()
    fig.savefig(out / "ik_determinism.png", dpi=140)

    worst = np.argsort(-c["d"][flip].max(0))[:8] if flip.sum() else []
    summary = dict(
        a=str(args.a),
        b=str(args.b),
        n_frames=int(len(solved)),
        n_solved=int(solved.sum()),
        n_bit_identical=zero,
        n_candidate_flips=int(flip.sum()),
        flip_rate=float(flip.sum() / max(solved.sum(), 1)),
        max_dqpos_on_agreeing_frames=(
            float(np.nanmax(c["per_frame"][agree])) if agree.sum() else None
        ),
        max_dqpos_on_flipped_frames=(
            float(np.nanmax(c["per_frame"][flip])) if flip.sum() else None
        ),
        cost_median_a=float(np.nanmedian(c["cost_a"][solved])),
        cost_median_b=float(np.nanmedian(c["cost_b"][solved])),
        b_cheaper_fraction=float(
            np.nansum(c["cost_b"][solved] < c["cost_a"][solved]) / solved.sum()
        ),
        root_xyz_max=float(np.abs(c["d"][solved][:, :3]).max()),
        top_flipped_dofs=[[a["names_qpos"][i], float(c["d"][flip][:, i].max())] for i in worst],
    )
    (out / "ik_determinism.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
