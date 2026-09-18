"""Cohort evaluation for the maskless mvq model.

Port of `$SRC/train/train_mvq.py:209-498` (jarvis_jax). Maskless deviations:
there is no prompt mask, so only the unprompted mode exists (no
prompted/unprompted double forward, no `mask_containment`); `evaluate`
returns one flat metrics dict instead of `{"prompted", "unprompted"}`.
Everything instance-selection related is imported from
`tracking.detector.mvq.policy`, never reimplemented, so this metric and
inference's own figures cannot diverge.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from tracking.detector.mvq.policy import EXIST_THRESH, policy_instance
from tracking.train.common.prefetch import prefetch
from tracking.train.data.windows import window_batches
from tracking.train.mvq.losses import LossWeights, mvq_loss
from tracking.train.mvq.matching import assign_slots, slot_ignore
from tracking.train.mvq.step import _batch_to_model

MM_PER_UNIT = 0.1
CONTACT_UNITS = 30.0  # 3 mm; real mounting pairs have centroid gaps of ~24-30 units


def cohorts(ds) -> dict[str, np.ndarray]:
    """`{name: bool array over ds indices}` for the five acceptance cohorts."""
    n = len(ds)
    c = {
        "female": np.array([ds.is_female(i) for i in range(n)]),
        "two_fly": np.array([ds.n_flies(i) > 1 for i in range(n)]),
        "single_fly": np.array([ds.n_flies(i) == 1 for i in range(n)]),
    }
    for g in sorted({ds.calib_group(i) for i in range(n)}):
        c[f"group_{g}"] = np.array([ds.calib_group(i) == g for i in range(n)])

    def _contact(i):
        if ds.n_flies(i) < 2:
            return False
        cen = ds.fly_centroids(i)
        return bool(np.isfinite(cen).all() and np.linalg.norm(cen[0] - cen[1]) < CONTACT_UNITS)

    c["contact_pair"] = np.array([_contact(i) for i in range(n)])
    return c


@nnx.jit
def _fwd(model, batch, prompt_on):
    return model(**_batch_to_model(batch), prompt_on=prompt_on)


def _padded_batches(ds, batch_size, num_workers):
    batches = window_batches(ds, batch_size, shuffle=False, drop_last=False,
                              num_workers=num_workers)
    for b in batches:
        b0 = b["crops"].shape[0]
        if b0 < batch_size:
            pad = batch_size - b0
            b = {k: np.concatenate([v, np.repeat(v[-1:], pad, axis=0)], axis=0)
                 for k, v in b.items()}
        yield b


def evaluate(model, ds, batch_size, *, cohort_masks, part_of_k, weights: LossWeights, mesh,
             num_workers=8, kp_weight=None) -> dict:
    """One unprompted pass over `ds` (JPEGs decoded once via `window_batches`).

    `batch_size` should be `tcfg.batch_size` (same as training): a smaller
    eval batch changes both the sharding layout and the batch composition of
    the reproj/uv2d/head_vs_reproj aggregates below. `cohort_masks`: `cohorts(ds)`
    output, name -> bool array over ds indices (`window_batches`' shuffle=False,
    drop_last=False order matches dataset-index order, so the sample at running
    position `offset+bi` is ds index `offset+bi`). The ragged last batch is
    padded up to `batch_size` (repeating its final row) so every batch shards
    evenly; the per-sample loop still only reads the first `b0` real rows.

    `kp_weight` is passed straight through to `mvq_loss` so eval scores the
    same objective training optimises; it moves none of the per-sample
    statistics below, which use unweighted L2 distances.

    `reproj_px`/`uv2d_px`/`head_vs_reproj_px` are batch-level means (from
    `mvq_loss`), weighted by each batch's own real (non-padded) valid-entry
    count when combined across batches, so the result is
    batch-grouping-independent.
    """
    names = ds.kp_order
    ii = lambda n: names.index(n)  # noqa: E731
    pairs = [("EyeL", "EyeR")] + [(f"T{i}{s}_Tro", f"T{i}{s}_FeTi")
                                   for i in (1, 2, 3) for s in "LR"]
    seg_names = [(a, c) for a, c in pairs if a in names and c in names]
    seg_idx = [(ii(a), ii(c)) for a, c in seg_names]

    # per_sample rows: (mpjpe_units, n_joints_with_gt, n_exist_pred, n_flies_true,
    # seg_lengths_pred|None, seg_lengths_gt|None, ds_index, is_two_fly_window,
    # mpjpe_policy_units|nan, is_policy_miss, cross_fly_frac|nan)
    per_sample = []
    batch_stats = []  # rows: (reproj_px, uv2d_px, head_vs_reproj_px, valid_entry_count)
    slot_counts = None  # (I,3) int TP/FP/FN of per-slot existence, sized on first batch
    cross_counts = None  # (I,2) int [n_keypoints_on_the_other_fly, n_keypoints_scored]
    sex_hits = []

    n = len(ds)
    offset = 0
    for jb in prefetch(_padded_batches(ds, batch_size, num_workers), mesh):
        b0 = min(batch_size, n - offset)
        B = batch_size
        vis2d = np.asarray(jb["vis2d"])
        fly_valid = np.asarray(jb["fly_valid"])
        weight = int((vis2d[:b0] & fly_valid[:b0, :, None, None, None]).sum())
        on = jnp.zeros((B,), bool)  # maskless: prompting never happens
        out = _fwd(model, jb, on)
        jb_m = dict(jb, prompt_on=on)
        _, m = mvq_loss(out, jb_m, weights, part_of_k, kp_weight)
        xyz = np.asarray(out["xyz"])  # (B,I,T,K,3)
        I = xyz.shape[1]  # noqa: E741 -- I = n_instances, as in the port source
        if slot_counts is None:
            slot_counts = np.zeros((I, 3), int)
            cross_counts = np.zeros((I, 2), int)
        has3d = np.asarray(jb["has3d"])
        kp3d_local = np.asarray(jb["kp3d_local"])
        fly_sex = np.asarray(jb["fly_sex"])
        unlabelled_sex = np.asarray(jb["unlabelled_sex"])
        has_f = has3d.astype(np.float32)
        cen = ((kp3d_local * has_f[..., None]).sum((2, 3))
               / np.maximum(has_f.sum((2, 3)), 1.0)[..., None])
        dist = np.linalg.norm(cen, axis=-1)
        assign, slot_t = assign_slots(jnp.asarray(fly_sex), jnp.asarray(fly_valid), on,
                                       jnp.asarray(dist), I)
        assign, slot_t = np.asarray(assign), np.asarray(slot_t)
        # `& ~slot_t`: a slot holding a labelled fly certainly exists, never ignored.
        ignore = np.asarray(slot_ignore(jnp.asarray(unlabelled_sex), I)) & ~slot_t
        sex_logit = np.asarray(out["sex_logit"])
        batch_stats.append((float(m["match_reproj_px"]), float(m["uv2d_px"]),
                             float(m["head_vs_reproj_px"]), weight))
        for bi in range(b0):  # only the real rows -- padding never enters a per-sample stat
            i_ds = offset + bi
            gt = kp3d_local[bi, 0]
            has = has3d[bi, 0]
            d = np.linalg.norm(xyz[bi] - gt[None], axis=-1)  # (I,T,K)
            # oracle instance choice (nearest GT); mpjpe3d_policy_units below uses the
            # POLICY a real inference call would actually use to pick an instance.
            inst_oracle = int(np.argmin(np.where(has[None], d, 0).sum((1, 2)) / max(has.sum(), 1)))
            e = d[inst_oracle][has]
            L_pred = ([np.linalg.norm(xyz[bi, inst_oracle, 0, a] - xyz[bi, inst_oracle, 0, c])
                       for a, c in seg_idx] if has.sum() > 0 else None)
            L_gt = ([np.linalg.norm(gt[0, a] - gt[0, c]) if has[0, a] and has[0, c] else np.nan
                     for a, c in seg_idx] if has.sum() > 0 else None)
            exist_probs = 1 / (1 + np.exp(-np.asarray(out["exist_logit"][bi])))  # (I,)
            exist = exist_probs >= EXIST_THRESH
            two_fly = ds.n_flies(i_ds) > 1
            # shared policy (tracking.detector.mvq.policy) -- unprompted, no mask exists.
            inst_policy = policy_instance(exist_probs, xyz[bi], prompted=False, has_mask=False)
            is_miss = inst_policy is None
            if is_miss:
                mpjpe_policy = np.nan
            else:
                e_pol = d[inst_policy][has]
                mpjpe_policy = float(e_pol.mean()) if e_pol.size else np.nan
            for s in range(I):
                if ignore[bi, s]:
                    continue
                tp_fp_fn = [exist[s] and slot_t[bi, s], exist[s] and not slot_t[bi, s],
                            (not exist[s]) and slot_t[bi, s]]
                slot_counts[s] += np.array(tp_fp_fn, int)
            for f in range(fly_valid.shape[1]):
                if fly_valid[bi, f] and fly_sex[bi, f] >= 0 and assign[bi, f] >= 0:
                    sex_hits.append((i_ds, int((sex_logit[bi, assign[bi, f]] > 0)
                                                == (fly_sex[bi, f] == 0))))
            # cross_fly_frac: on a two-labelled-fly window, is this slot's prediction on
            # the right ANIMAL? NaN unless both flies are labelled, assigned and have a centroid.
            fl = [f for f in range(fly_valid.shape[1])
                  if fly_valid[bi, f] and assign[bi, f] >= 0 and has_f[bi, f].sum() > 0]
            cross = np.nan
            if len(fl) >= 2:
                vals = []
                for f in fl:
                    sl = int(assign[bi, f])
                    d_own = np.linalg.norm(xyz[bi, sl] - cen[bi, f], axis=-1)
                    d_oth = np.min([np.linalg.norm(xyz[bi, sl] - cen[bi, o], axis=-1)
                                    for o in fl if o != f], axis=0)
                    wrong = d_own > d_oth
                    cross_counts[sl] += np.array([int(wrong.sum()), int(wrong.size)], int)
                    vals.append(float(wrong.mean()))
                cross = float(np.mean(vals))
            per_sample.append((float(e.mean()) if e.size else np.nan, int(e.size), int(exist.sum()),
                                int(fly_valid[bi].sum()), L_pred, L_gt, i_ds, two_fly, mpjpe_policy,
                                is_miss, cross))
        offset += b0

    return _finish(per_sample, batch_stats, slot_counts, cross_counts, sex_hits, seg_names,
                   cohort_masks)


def _finish(per_sample, batch_stats, slot_counts, cross_counts, sex_hits, seg_names, cohort_masks):
    samples = per_sample
    mp = np.array([p[0] for p in samples])
    n = np.array([p[1] for p in samples])
    ok = np.isfinite(mp)
    # np.zeros((0, 4)) (not np.array([])) when there were no batches at all, so
    # the [:, k] column slices below stay 2-D instead of raising IndexError.
    bstats = np.array(batch_stats) if batch_stats else np.zeros((0, 4))
    w = bstats[:, 3]

    def _wmean(col):
        if col.size == 0:
            return float("nan")
        return float(np.average(col, weights=w)) if w.sum() > 0 else float(np.mean(col))

    res = {"mpjpe3d_units": float(np.average(mp[ok], weights=n[ok])) if ok.any() else float("nan"),
           "reproj_px": _wmean(bstats[:, 0]), "uv2d_px": _wmean(bstats[:, 1]),
           "head_vs_reproj_px": _wmean(bstats[:, 2])}
    res["mpjpe3d_mm"] = res["mpjpe3d_units"] * MM_PER_UNIT
    mp_policy = np.array([p[8] for p in samples])
    ok_policy = np.isfinite(mp_policy)
    miss = np.array([p[9] for p in samples], bool)
    res["mpjpe3d_policy_units"] = (float(np.average(mp_policy[ok_policy], weights=n[ok_policy]))
                                    if ok_policy.any() else float("nan"))
    res["mpjpe3d_policy_mm"] = res["mpjpe3d_policy_units"] * MM_PER_UNIT
    res["policy_miss_frac"] = float(miss.mean()) if len(samples) else 0.0
    # per-slot existence precision/recall from the label-driven TP/FP/FN counted above
    # (non-ignored slots only -- an unlabelled fly legitimately present gets no
    # existence loss/credit for its would-be slot).
    sc = slot_counts if slot_counts is not None else np.zeros((0, 3), int)
    for s in range(sc.shape[0]):
        tp, fp, fn = (int(x) for x in sc[s])
        res[f"exist_prec_slot{s}"] = float(tp) / max(tp + fp, 1)
        res[f"exist_rec_slot{s}"] = (float(tp) / (tp + fn)) if (tp + fn) > 0 else float("nan")
    # legacy exist_prec/exist_rec: same TP/FP/FN, summed over the TYPED slots (1..3) only.
    tp = int(sc[1:, 0].sum())
    fp = int(sc[1:, 1].sum())
    fn = int(sc[1:, 2].sum())
    res["exist_prec"] = float(tp) / max(tp + fp, 1)
    res["exist_rec"] = float(tp) / max(tp + fn, 1)
    # sex_hits rows are (ds_index, hit), so the same hits can be sliced per cohort below.
    res["sex_acc"] = float(np.mean([h for _, h in sex_hits])) if sex_hits else float("nan")
    cross = np.array([p[10] for p in samples], float) if samples else np.zeros(0)
    res["cross_fly_frac"] = float(np.nanmean(cross)) if np.isfinite(cross).any() else float("nan")
    cc = cross_counts if cross_counts is not None else np.zeros((0, 2), int)
    for s_ in range(cc.shape[0]):
        res[f"cross_fly_frac_slot{s_}"] = (
            float(cc[s_, 0]) / cc[s_, 1]) if cc[s_, 1] else float("nan")
    miss_arr = np.array([p[9] for p in samples], bool) if samples else np.zeros(0, bool)
    Ls = [p[4] for p in samples if p[4] is not None]  # skip samples with no 3D-labelled joint
    Ls_gt = [p[5] for p in samples if p[5] is not None]  # same gate as Ls
    if Ls:
        Ls = np.array(Ls)
        Ls_gt = np.array(Ls_gt)
        for j, (a, c) in enumerate(seg_names):
            res[f"rigid_spread_mm/{a}-{c}"] = float(np.std(Ls[:, j]) * MM_PER_UNIT)
            gt_col = Ls_gt[:, j]
            gt_ok = np.isfinite(gt_col)
            ratio = np.mean(Ls[gt_ok, j]) / np.mean(gt_col[gt_ok]) if gt_ok.any() else np.nan
            res[f"rigid_len_ratio/{a}-{c}"] = float(ratio)
    for name, mask in cohort_masks.items():
        in_cohort = np.array([mask[p[6]] for p in samples], bool) if samples else np.zeros(0, bool)
        sel = in_cohort & ok
        cohort_mp = float(np.average(mp[sel], weights=n[sel])) if sel.any() else float("nan")
        res[f"cohort_{name}"] = cohort_mp
        cx = cross[in_cohort] if in_cohort.any() else np.zeros(0)
        res[f"cross_fly_frac_{name}"] = (float(np.nanmean(cx))
                                          if cx.size and np.isfinite(cx).any() else float("nan"))
        pmiss = float(miss_arr[in_cohort].mean()) if in_cohort.any() else float("nan")
        res[f"policy_miss_frac_{name}"] = pmiss
        ch_hits = [h for i_ds, h in sex_hits if mask[i_ds]]
        res[f"sex_acc_{name}"] = float(np.mean(ch_hits)) if ch_hits else float("nan")
    return res
