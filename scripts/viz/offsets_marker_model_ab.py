"""Compare two fitted marker models (two `offsets_*.h5`) on the same fly.

Marker offsets are the one per-individual constant the whole IK rests on, and
two offsets fits of the same data can disagree for reasons that no residual
number distinguishes: a different local optimum, a different alternation, or a
genuinely better marker model. This is the A/B that tells them apart.

Each candidate offsets set gets its OWN full pose solve over the sample, from
the same warm start, so neither is scored on the other's poses. Then:

  A  per-keypoint mean marker error (fitted site vs observed keypoint), both
     sets side by side, with the keypoints whose offsets differ most shaded.
  B  a RIGID invariant the pose solve cannot touch: two markers on the SAME
     body are a fixed distance apart, determined by the offsets alone. Compare
     that model distance to the observed median distance in the data. Whichever
     set sits closer to the black dot is the better marker model, and no pose
     solve, local optimum or solver tuning can flatter it.
  C  the two frames the FIRST candidate fits worst, in the x-z view: observed
     keypoints against both fitted marker sets.

**State the expectation before reading it.** If candidate 2 is the better
marker model, A's bars for it are at or below candidate 1's and the gap is
largest on the shaded keypoints; B's marks for it sit closer to the observed
medians, especially for the largest-offset pairs (the wing veins); and in C its
markers sit ON the observed keypoints where the other stands off. An
improvement concentrated in one keypoint, or a B that goes the other way, means
the residual number moved for a reason other than the marker model.

Written for, and first run on, the finding in
`.superpowers/sdd/2026-09-10-preprocess-ik-postprocess/task-8-fix-report.md`:
`stac_mjx`'s pose solver caches the marker offsets into its compiled problem,
so `Stac.fit_offsets`' pose solves never see the offsets it is fitting, and
every `offsets_fly{N}.h5` it wrote is half an alternation.

Usage
-----
    python scripts/viz/offsets_marker_model_ab.py \
        --reference-run <dir with offsets_fly0.h5>  --fly 0 \
        --offsets "stac_mjx reference"=<a.h5> --offsets "this port"=<b.h5> \
        --out figures/<date>-<topic>

The first `--offsets` is the baseline: ratios and panel C's worst frames are
taken from it. Keypoint data, anatomy config and solver tuning all come from
the reference run's own `offsets_fly{N}.h5`, so the A/B is run on exactly what
was fit, not on a re-derived sample.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tracking.inverse_kinematics import offsets_fit as offsets_fit_mod  # noqa: E402
from tracking.inverse_kinematics.anatomy import load_anatomy  # noqa: E402
from tracking.inverse_kinematics.solver import (  # noqa: E402
    PerFrameSolver,
    forward,
    get_site_xpos,
    mjx_load,
    set_site_pos,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_COLOR = "#e08a2e"  # orange: the baseline / first candidate
NEW_COLOR = "#2e9e5b"  # green: the fit under test  (viz/core/colors.py's fit green)


def _parse_offsets(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--offsets wants NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def _pose_solve(anatomy, cfg, kp_flat, offsets):
    """One full pose solve of `kp_flat` against `offsets`. Returns `(T, K, 3)`
    fitted marker-site world positions."""
    settings = offsets_fit_mod._solver_settings_from_cfg(cfg)
    root_kp_idx, _ = offsets_fit_mod._root_and_trunk(anatomy)
    orientation_idx = offsets_fit_mod._orientation_indices(anatomy)
    site_idxs = jnp.asarray(anatomy.site_idxs)

    mjx_model, mjx_data = mjx_load(anatomy.mj_model)
    mjx_model = set_site_pos(mjx_model, jnp.asarray(offsets, jnp.float32), site_idxs)
    mjx_data = forward(mjx_model, mjx_data)
    _data, qpos, _sites = offsets_fit_mod.pose_optimization(
        anatomy,
        mjx_model,
        mjx_data,
        kp_flat,
        root_kp_idx=root_kp_idx,
        orientation_idx=orientation_idx,
        settings=settings,
        solver=PerFrameSolver(settings),
    )

    def one(q):
        return get_site_xpos(forward(mjx_model, mjx_data.replace(qpos=q)), site_idxs)

    return np.asarray(jax.vmap(one)(jnp.asarray(qpos)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", type=Path, required=True)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--offsets", type=_parse_offsets, action="append", required=True)
    ap.add_argument("--anatomy", type=Path, default=REPO_ROOT / "configs/anatomy/v1.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.offsets) < 2:
        ap.error("pass at least two --offsets NAME=PATH; an A/B needs a B")
    args.out.mkdir(parents=True, exist_ok=True)

    src = args.reference_run / f"offsets_fly{args.fly}.h5"
    with h5py.File(src, "r") as f:
        kp_flat = np.asarray(f["kp_data"][:], np.float64)
        cfg = OmegaConf.to_container(
            OmegaConf.create(f["config"][()].decode()).model, resolve=False
        )
    anatomy = load_anatomy(OmegaConf.load(args.anatomy))
    names = list(anatomy.kp_order.names)
    n_kp = len(names)
    obs = kp_flat.reshape(-1, n_kp, 3)
    body_of = cfg["KEYPOINT_MODEL_PAIRS"]

    offsets, sites, err = {}, {}, {}
    for name, path in args.offsets:
        with h5py.File(path, "r") as f:
            stored = [n.decode() if isinstance(n, bytes) else str(n) for n in f["kp_names"][:]]
            if stored != names:
                raise SystemExit(f"{path}: kp_names disagree with the anatomy's keypoint order")
            offsets[name] = np.asarray(f["offsets"][:], np.float64).reshape(-1, 3)
        sites[name] = _pose_solve(anatomy, cfg, kp_flat, offsets[name])
        err[name] = np.linalg.norm(sites[name] - obs, axis=2)  # (T, K)
        total = float(np.nansum((sites[name] - obs) ** 2))
        print(f"{name}: total marker residual {total:.6f}  mean/frame {total / obs.shape[0]:.6f}")

    base = args.offsets[0][0]
    others = [n for n, _ in args.offsets[1:]]
    # The keypoints the two marker models disagree about most: what panel A shades.
    spread = np.abs(offsets[others[0]] - offsets[base]).max(axis=1)
    shaded = [names[i] for i in np.argsort(-spread)[:8]]

    np.savez_compressed(
        args.out / f"offsets_ab_fly{args.fly}.npz",
        obs=obs,
        kp_names=np.array(names, dtype="S"),
        **{f"sites::{n}": v for n, v in sites.items()},
        **{f"offsets::{n}": v for n, v in offsets.items()},
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.9, 1.2], hspace=0.55, wspace=0.18)
    colors = {base: BASE_COLOR, **{n: NEW_COLOR for n in others}}

    ax = fig.add_subplot(gs[0, :])
    x = np.arange(n_kp)
    width = 0.8 / len(sites)
    for k, (name, e) in enumerate(err.items()):
        ax.bar(
            x + (k - (len(err) - 1) / 2) * width,
            e.mean(axis=0),
            width,
            color=colors[name],
            label=f"{name} offsets",
        )
    for kp_name in shaded:
        ax.axvspan(
            names.index(kp_name) - 0.5,
            names.index(kp_name) + 0.5,
            color="#d0d7e6",
            alpha=0.55,
            zorder=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("mean |fitted site - observed kp|  (model units)")
    ax.set_title(
        f"A  Per-keypoint marker error, each offsets set on its OWN {obs.shape[0]}-frame pose "
        f"solve (fly{args.fly}).  Shaded = the 8 keypoints whose offsets differ most.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    rows = []
    for i in range(n_kp):
        for j in range(i + 1, n_kp):
            if body_of[names[i]] != body_of[names[j]]:
                continue
            rows.append(
                (
                    f"{names[i]}\n-{names[j]}",
                    float(np.nanmedian(np.linalg.norm(obs[:, i] - obs[:, j], axis=1))),
                    {n: float(np.linalg.norm(o[i] - o[j])) for n, o in offsets.items()},
                )
            )
    rows.sort(key=lambda r: -r[1])
    ax = fig.add_subplot(gs[1, :])
    xo = np.arange(len(rows))
    ax.plot(xo, [r[1] for r in rows], "ko", ms=7, label="observed median (the invariant)")
    for name in sites:
        ax.plot(
            xo,
            [r[2][name] for r in rows],
            "_",
            color=colors[name],
            ms=18,
            mew=3,
            label=f"{name} offsets",
        )
    ax.set_xticks(xo)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylabel("marker-pair distance (model units)")
    ax.set_title(
        "B  RIGID invariant the pose solve cannot touch: two markers on the same body are a "
        "fixed distance apart, set by the offsets alone.  Closer to the black dot is better.",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for name in sites:
        dev = float(np.mean([abs(r[2][name] - r[1]) for r in rows]))
        print(f"{name}: rigid-invariant mean |model - observed| = {dev:.6f}")

    for col, frame in enumerate(np.argsort(-err[base].mean(axis=1))[:2]):
        ax = fig.add_subplot(gs[2, col])
        centre = obs[frame].mean(axis=0)
        ax.scatter(
            obs[frame, :, 0] - centre[0],
            obs[frame, :, 2] - centre[2],
            s=46,
            facecolors="none",
            edgecolors="k",
            linewidths=1.3,
            label="observed keypoint",
            zorder=3,
        )
        for name, v in sites.items():
            ax.scatter(
                v[frame, :, 0] - centre[0],
                v[frame, :, 2] - centre[2],
                s=16,
                color=colors[name],
                label=f"fitted, {name} offsets",
                zorder=4,
            )
        for kp_name in shaded:
            i = names.index(kp_name)
            ax.annotate(
                kp_name,
                (obs[frame, i, 0] - centre[0], obs[frame, i, 2] - centre[2]),
                fontsize=6,
                color="#333",
            )
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (model units)")
        ax.set_ylabel("z (model units)")
        # Two lines: one long title per column runs into its neighbour's.
        errs = "   ".join(f"{n} {err[n][frame].mean():.4f}" for n in err)
        ax.set_title(
            f"C{col + 1}  frame {frame} (worst for {base}), x-z view\nmean marker err: {errs}",
            fontsize=9,
        )
        if col == 0:
            ax.legend(fontsize=8, loc="best")

    totals = {n: float(np.nansum((v - obs) ** 2)) for n, v in sites.items()}
    ratios = "   ".join(f"{n} {totals[n]:.4f} (x{totals[n] / totals[base]:.3f})" for n in totals)
    fig.suptitle(f"Marker-model A/B, fly{args.fly}.  Total marker residual: {ratios}", fontsize=12)
    out_png = args.out / f"offsets_ab_fly{args.fly}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print("wrote", out_png)


if __name__ == "__main__":
    main()
