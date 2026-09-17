"""Window enumeration and metadata over the unified training root, for the mvq model.

A window is (recording, host fly, start frame f0, spacing delta): T frames of
the host spaced `delta` apart, all cameras, to be cropped around one
window-level 3D center by `_build`. Camera
order and keypoint order are resolved BY NAME (`tracking.io.names.Order`),
never by position. Sex is resolved PER WINDOW, annotation-first: a
(recording, fly) pair can carry framesets from different annotation subsets
that label different animals under the same fly id, so collapsing to one
value per (recording, fly) is wrong (see `_resolve_fs_sex`).

The unified root can carry more than one frameset for the exact same
(recording, fly, frame) -- e.g. a human export and a pseudo "partner" export
both landing on the same real video frame. Every frameset still yields
exactly one window (`test_every_frameset_becomes_one_window_at_T1`), so a
window's own metadata (`calib_group`, `source`, `weight`, `role`, sex) is
read from the frameset THAT BUILT IT (`_win_fs[i]`), never re-derived by
looking the key back up -- a lookup could silently return the other
frameset's data. `window_index` raises rather than picking one silently
when a key is ambiguous.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from tracking.curate import schema
from tracking.curate.masks.store import MaskStore
from tracking.detector.mvq.geometry import affine_rows
from tracking.detector.mvq.slots import SEX_FEMALE, SEX_MALE, SEX_PRESENT_UNKNOWN, SEX_UNKNOWN
from tracking.geometry.rig import CameraRig
from tracking.train.data.copypaste import body_plane_axes, composite, sample_offset
from tracking.train.data.transforms import crop_origin

CROP = 448
WINDOW_KEYS = (
    "crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
    "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "donor_mask", "crop_origin",
    "fly_sex", "unlabelled_sex", "sample_weight", "is_negative",
)  # fmt: skip

_SEX_CODE = {"female": SEX_FEMALE, "male": SEX_MALE}


def _resolved_slots(frameset):
    """(image id, annotation id) for the cameras of `frameset` that resolved to a fly.

    >>> list(_resolved_slots({"frames": [1, 2], "ann_ids": [7, None]}))
    [(1, 7)]
    """
    for img_id, ann_id in zip(frameset["frames"], frameset["ann_ids"], strict=True):
        if ann_id is not None:
            yield img_id, ann_id


def _frameset_own_sex(fsv, id2ann):
    """The frameset's own annotation-level `sex`, from its first resolved slot.

    >>> _frameset_own_sex({"frames": [1], "ann_ids": [1]}, {1: {"sex": "male"}})
    'male'
    """
    for _, ann_id in _resolved_slots(fsv):
        return id2ann[ann_id].get("sex", "unknown")
    return "unknown"


def _resolve_sex(ann_sex, fly_id, rec_meta):
    """Sex fallback chain: annotation, then per-fly manifest, then recording, then unknown.

    >>> _resolve_sex("unknown", 0, {"fly_sex": {"fly0": "female"}})
    'female'
    """
    if ann_sex and ann_sex != "unknown":
        return ann_sex
    per_fly = (rec_meta.get("fly_sex") or {}).get(f"fly{fly_id}")
    if per_fly and per_fly != "unknown":
        return per_fly
    rec_sex = rec_meta.get("sex")
    if rec_sex and rec_sex != "unknown":
        return rec_sex
    return "unknown"


def _parse_sex_overrides(overrides):
    """{recording: {fly: "female"|"male"}}, keys/values normalized; raises on a bad sex.

    >>> _parse_sex_overrides({"rec": {"fly0": "Female"}})
    {'rec': {0: 'female'}}
    """
    out = {}
    for rec, m in (overrides or {}).items():
        per = {}
        for k, v in dict(m).items():
            fly = int(str(k).lower().replace("fly", ""))
            sex = str(v).lower()
            if sex not in _SEX_CODE:
                raise ValueError(
                    f"sex_overrides[{rec!r}][{k!r}] = {v!r}: expected 'female' or 'male'"
                )
            per[fly] = sex
        out[str(rec)] = per
    return out


def _parse_key(key, fsv):
    """(recording, frame, fly id) of a frameset key `<source_id>/<rec>/Frame_<n>/<tail>`.

    >>> _parse_key("src/rec/Frame_5/fly0", {"recording": "rec", "fly_id": 0})
    ('rec', 5, 0)
    """
    frame = int(key.rsplit("/", 2)[1].split("_")[1])
    return fsv["recording"], frame, int(fsv["fly_id"])


class WindowDataset:
    """Enumerates training windows over the unified root and exposes their metadata.

    >>> ds = WindowDataset(root, "train", T=1)          # doctest: +SKIP
    >>> len(ds), ds.calib_group(0), ds.camera_names(0)  # doctest: +SKIP
    """

    def __init__(
        self,
        root,
        split,
        T=1,
        *,
        pair_deltas=(1,),
        max_flies=2,
        jitter_units=3.0,
        seed=0,
        train=True,
        recordings=None,
        copy_paste=None,
        center_shift_units=0.0,
        sex_overrides=None,
        masks_root=None,
    ):
        self.root, self.split, self.T = root, split, int(T)
        self.max_flies, self.jitter, self.train = int(max_flies), float(jitter_units), bool(train)
        self.seed = int(seed)
        self.center_shift = float(center_shift_units)
        self.sex_overrides = _parse_sex_overrides(sex_overrides)
        self.copy_paste = copy_paste
        self.masks_root = masks_root
        self.epoch = 0

        coco = schema.load_instances(root, split)
        self.manifest_root = schema.load_manifest(root)
        self.manifest = self.manifest_root.get("recordings", {})
        self.kp_order = schema.keypoint_order(root)
        self._img = {im["id"]: im for im in coco["images"]}
        self._ann = {a["id"]: a for a in coco["annotations"]}

        self._fs, all_entries = {}, []
        for key, fsv in coco["framesets"].items():
            rec, frame, fly = _parse_key(key, fsv)
            if recordings is not None and rec not in recordings:
                continue
            self._fs[(rec, frame, fly)] = fsv
            all_entries.append((key, rec, frame, fly, fsv))

        self.pair_deltas = tuple(dict.fromkeys(int(d) for d in (pair_deltas or (1,))))
        self.windows, self.win_delta, self._win_fs = [], [], []
        self._win_index = collections.defaultdict(list)
        if self.T == 1:
            for _key, rec, frame, fly, fsv in sorted(all_entries, key=lambda e: e[0]):
                self._add_window(rec, fly, frame, 0, fsv, dedup=False)
        else:
            for rec, frame, fly in sorted(self._fs):
                fsv = self._fs[(rec, frame, fly)]
                if fly < 0:
                    for d, start in self._negative_pairs(rec, frame, fly):
                        start_fsv = self._fs.get((rec, start, fly), fsv)
                        self._add_window(rec, fly, start, d, start_fsv, dedup=True)
                    continue
                for d in self.pair_deltas:
                    if all((rec, frame + k * d, fly) in self._fs for k in range(self.T)):
                        self._add_window(rec, fly, frame, d, fsv, dedup=True)

        self._win_sex = [
            self._resolve_fs_sex(rec, f0, fly, self._win_fs[i])
            for i, (rec, fly, f0) in enumerate(self.windows)
        ]
        self._warn_sex_disagreements()
        for rec, m in sorted(self.sex_overrides.items()):
            n = sum(1 for r, fly, _ in self.windows if r == rec and fly in m)
            print(
                f"[windows] sex_label_overrides {rec}: {dict(sorted(m.items()))} -- overriding "
                f"the annotation/manifest chain on {n} window(s) of split {self.split!r}",
                flush=True,
            )
        self._donors = collections.defaultdict(list)
        if self.copy_paste is not None:
            for i in range(len(self.windows)):
                if self.is_negative(i):
                    continue
                sex = _SEX_CODE.get(self._win_sex[i], SEX_UNKNOWN)
                self._donors[(self.calib_group(i), sex, self.delta(i))].append(i)

        self._rigs = {}
        self._affine_mt = {}

    def _add_window(self, rec, fly, f0, d, fsv, *, dedup):
        """Append window (rec, fly, f0) at spacing d, owned by `fsv`.

        `dedup` skips an exact repeat key -- a T > 1 negative pair named from
        both its endpoints is one window either way. T = 1 never dedups: the
        unified root can carry two distinct framesets for the same (rec, fly,
        frame), and each still yields its own window.
        """
        key = (rec, int(fly), int(f0), int(d))
        if dedup and key in self._win_index:
            return
        self._win_index[key].append(len(self.windows))
        self.windows.append((rec, int(fly), int(f0)))
        self.win_delta.append(int(d))
        self._win_fs.append(fsv)

    def _negative_pairs(self, rec, frame, fly):
        """[(delta, start_frame)] this negative frameset supports, from its own `partners` map.

        A partner not `d` frames away is a broken link and raises; no entry for
        a spacing yields no window there (never a frozen same-frame pair).
        """
        partners = self._fs[(rec, frame, fly)].get("partners") or {}
        out = []
        for d in self.pair_deltas:
            pf = partners.get(str(int(d)), partners.get(int(d)))
            if pf is None:
                continue
            pf = int(pf)
            if abs(pf - frame) != int(d):
                raise ValueError(
                    f"negative frameset {rec}/Frame_{frame} (fly {fly}): partners[{d!r}] = {pf}, "
                    f"{abs(pf - frame)} frames away, not {d}"
                )
            start = min(frame, pf)
            if all((rec, start + k * int(d), fly) in self._fs for k in range(self.T)):
                out.append((int(d), start))
        return out

    def _warn_sex_disagreements(self):
        """One warning per (recording, fly) whose framesets disagree on annotation sex."""
        census = collections.defaultdict(collections.Counter)
        for (rec, _frame, fly), fsv in self._fs.items():
            if fly < 0:
                continue
            s = _frameset_own_sex(fsv, self._ann)
            if s and s != "unknown":
                census[(rec, fly)][s] += 1
        for (rec, fly), c in sorted(census.items()):
            if len(c) > 1:
                man_sex = (self.manifest.get(rec, {}).get("fly_sex") or {}).get(f"fly{fly}")
                print(
                    f"[windows] {rec} fly{fly}: framesets disagree on annotation sex "
                    f"{dict(sorted(c.items()))} -- the annotation wins per window; manifest "
                    f"fly_sex={man_sex!r} is not used for these framesets",
                    flush=True,
                )

    def __len__(self):
        return len(self.windows)

    def worker_spec(self):
        """Picklable description of this dataset for a loader worker process
        (`tracking.train.data.loaders`). Built from the ATTRIBUTES, not from
        a stashed copy of the constructor arguments, so it cannot drift from
        what the object actually is.

        `recordings` is recovered as the set of recordings that survived the
        constructor's filter, which selects exactly the same framesets
        whether the caller passed None or that same set.

        Refused for a subclass: the spec carries no class, so
        `loaders.build_dataset` would rebuild a plain `WindowDataset` and the
        subclass's `__getitem__` would silently not run in the worker.
        """
        if type(self) is not WindowDataset:
            raise TypeError(
                f"{type(self).__name__} subclasses WindowDataset, but a WindowSpec carries "
                f"no class: tracking.train.data.loaders.build_dataset would rebuild a plain "
                f"WindowDataset in each worker and {type(self).__name__}'s overrides would "
                f"never run. Use workers='threads', or teach build_dataset about the subclass."
            )
        from tracking.train.data.loaders import WindowSpec

        return WindowSpec(
            root=self.root,
            split=self.split,
            T=int(self.T),
            pair_deltas=tuple(self.pair_deltas),
            max_flies=int(self.max_flies),
            jitter_units=float(self.jitter),
            seed=int(self.seed),
            train=bool(self.train),
            recordings=tuple(sorted({rec for rec, _frame, _fly in self._fs})),
            copy_paste=self.copy_paste,
            center_shift_units=float(self.center_shift),
            sex_overrides={r: dict(m) for r, m in self.sex_overrides.items()},
            masks_root=self.masks_root,
        )

    def window_index(self, rec, fly, f0, delta=None):
        """Index of ONE window BY KEY -- never by a position in `windows`.

        `delta=None` means "whatever spacing was built for (rec, fly, f0)",
        ambiguous when several exist (several pair_deltas, or a duplicate
        frameset for the same key); ambiguity raises ValueError rather than
        picking one silently, and an unmatched key raises KeyError.
        """
        if delta is not None:
            hits = self._win_index.get((rec, int(fly), int(f0), int(delta)), [])
            if not hits:
                raise KeyError((rec, int(fly), int(f0), int(delta)))
            if len(hits) > 1:
                raise ValueError(
                    f"{(rec, int(fly), int(f0), int(delta))} matches {len(hits)} windows "
                    f"-- duplicate frameset for this key"
                )
            return hits[0]
        hits = [
            i
            for (r, f, s, _d), idxs in self._win_index.items()
            if (r, f, s) == (rec, int(fly), int(f0))
            for i in idxs
        ]
        if not hits:
            raise KeyError((rec, int(fly), int(f0)))
        if len(hits) > 1:
            raise ValueError(
                f"{(rec, int(fly), int(f0))} matches {len(hits)} windows at spacings "
                f"{sorted(self.delta(i) for i in hits)} -- pass delta="
            )
        return hits[0]

    def delta(self, i):
        """Frame spacing of window i: one of `pair_deltas` for T > 1, else 0."""
        return int(self.win_delta[i])

    def _frames(self, i):
        """The T video frames window i spans: `f0 + k * delta(i)`."""
        f0 = self.windows[i][2]
        return [f0 + k * self.delta(i) for k in range(self.T)]

    def calib_group(self, i):
        """Calibration group of window i, from ITS OWN frameset, never the manifest.

        A recording can span more than one group (the 2026-04-02 mid-day
        recalibration); a window whose calibration cannot be resolved raises
        rather than defaulting to one.
        """
        rec = self.windows[i][0]
        grp = schema.frameset_field(self.manifest_root, self._win_fs[i], rec, "calib_group", None)
        if grp is None:
            raise ValueError(f"window {i} ({rec}): no calib_group resolvable from its frameset")
        return grp

    def _fs_field(self, i, key, default):
        """Value of `key` for window i: its own frameset, else the recording, else root default."""
        rec = self.windows[i][0]
        return schema.frameset_field(self.manifest_root, self._win_fs[i], rec, key, default)

    def source(self, i):
        """`"real"` | `"pseudo"`: this window's provenance."""
        return str(self._fs_field(i, "source", "real"))

    def weight(self, i):
        """Loss weight for this window."""
        return float(self._fs_field(i, "weight", 1.0))

    def role(self, i):
        """`"anchor"` | `"partner"` | `"negative"`: the Delta-pairing side of this window.

        A negative window can be either side of its own pair (`_negative_pairs`)
        -- `"negative"` and `"partner"` both occur among negatives -- so this is
        read straight from the frameset, never inferred from `is_negative`.
        """
        return str(self._fs_field(i, "role", "anchor"))

    def is_negative(self, i):
        """True for an empty window (fly id -1): real pixels, no labelled fly."""
        return self.windows[i][1] < 0

    def is_female(self, i):
        """Host sex of THIS window (`_resolve_fs_sex`, annotation-first)."""
        return self._win_sex[i] == "female"

    def n_flies(self, i):
        """Labelled flies in window i (host + others, capped at max_flies); 0 for a negative."""
        return len(self._window_flies(i)[1])

    def _resolve_fs_sex(self, rec, frame, fly, fsv=None):
        """Resolved sex string of one window's host, annotation-first.

        `fsv` is the window's OWN frameset when known (its own host data); it
        falls back to a (rec, frame, fly) lookup for another fly's sex.
        """
        if int(fly) < 0:
            return "unknown"
        ov = self.sex_overrides.get(rec, {}).get(int(fly))
        if ov is not None:
            return ov
        if fsv is None:
            fsv = self._fs.get((rec, frame, fly))
        own = _frameset_own_sex(fsv, self._ann) if fsv is not None else "unknown"
        return _resolve_sex(own, fly, self.manifest.get(rec, {}))

    def fly_sex_code(self, rec, fly, frame):
        """Sex code (0 female, 1 male, -1 unknown) of one fly at one frame."""
        return _SEX_CODE.get(self._resolve_fs_sex(rec, frame, fly), SEX_UNKNOWN)

    def _window_flies(self, i):
        """Labelled fly ids in window i, host first, capped at max_flies; empty for a negative."""
        rec, host, _ = self.windows[i]
        if host < 0:
            return rec, []
        frames = self._frames(i)
        others = sorted(
            {
                k[2]
                for k in self._fs
                if k[0] == rec and k[1] in frames and k[2] != host and k[2] >= 0
            }
        )
        return rec, [host] + others[: self.max_flies - 1]

    def unlabelled_sex(self, i):
        """Sex code of the one unlabelled animal, or SEX_UNKNOWN if none / all labelled."""
        if self.windows[i][1] < 0:
            return SEX_UNKNOWN
        rec, flies = self._window_flies(i)
        meta = self.manifest.get(rec, {})
        n_present = int(meta.get("n_flies", len(flies)))
        if n_present <= len(flies):
            return SEX_UNKNOWN
        if meta.get("sex") == "mixed" and n_present == 2 and len(flies) == 1:
            host = self.fly_sex_code(rec, flies[0], self.windows[i][2])
            return (1 - host) if host in (SEX_FEMALE, SEX_MALE) else SEX_PRESENT_UNKNOWN
        named = {int(k[3:]): v for k, v in (meta.get("fly_sex") or {}).items()}
        missing = [fid for fid in sorted(named) if fid not in flies]
        if not missing:
            return SEX_PRESENT_UNKNOWN
        return _SEX_CODE.get(named[missing[0]], SEX_PRESENT_UNKNOWN)

    def _rig(self, group):
        if group not in self._rigs:
            self._rigs[group] = CameraRig.from_calib_dir(
                Path(self.root) / "calibrations" / str(group)
            )
        return self._rigs[group]

    def _affine(self, group):
        """(C,2,3) M, (C,2) t (float64) of calib group `group`; raises if it is not affine."""
        if group not in self._affine_mt:
            M, t = affine_rows(self._rig(group).matrices_f32)
            self._affine_mt[group] = (np.asarray(M, np.float64), np.asarray(t, np.float64))
        return self._affine_mt[group]

    def camera_names(self, i):
        """The window's camera-axis names, in the SAME order as a future `crops`/`kp2d`/`M`."""
        return self._rig(self.calib_group(i)).cameras

    def _full_labels(self, fsv, rig):
        """Per camera (BY NAME) full-frame (K, 3) labels for one frameset's resolved slots."""
        return self._frame_labels(fsv, rig)[0]

    def fly_centroids(self, i):
        """(n_labelled_flies, 3) world centroid of each fly's DLT-able labels at frame 0.

        Labels only, no image decode -- cheap enough for cohort summaries.
        """
        rec, flies = self._window_flies(i)
        rig = self._rig(self.calib_group(i))
        f0 = self.windows[i][2]
        out = np.full((len(flies), 3), np.nan, np.float32)
        for fi, fly in enumerate(flies):
            fsv = self._fs.get((rec, f0, fly))
            if fsv is None:
                continue
            kp = self._full_labels(fsv, rig)
            pts = np.full((len(self.kp_order), 3), np.nan, np.float64)
            for j in range(len(self.kp_order)):
                valid = kp[:, j, 2] > 0
                if valid.sum() >= 2:
                    pts[j] = rig.reconstruct(kp[:, j, :2], valid)
            has = ~np.isnan(pts).any(-1)
            if has.any():
                out[fi] = pts[has].mean(0)
        return out

    def _frame_labels(self, fsv, rig):
        """Per camera (BY NAME) full-frame (K,3) labels and (image info, annotation)."""
        cam_row = {n: idx for idx, n in enumerate(rig.cameras)}
        K = len(self.kp_order)
        kp = np.zeros((rig.n_cameras, K, 3), np.float32)
        infos = [None] * rig.n_cameras
        for img_id, ann_id in _resolved_slots(fsv):
            info, ann = self._img[img_id], self._ann[ann_id]
            c = cam_row.get(info["file_name"].split("/")[1])
            if c is None:
                continue
            k = np.asarray(ann["keypoints"], np.float32)
            if k.size == K * 3:
                kp[c] = k.reshape(-1, 3)
            infos[c] = (info, ann)
        return kp, infos

    def _dlt(self, kp, rig):
        """(K,3) DLT triangulation and (K,) has3d from `kp` (C,K,3) via `rig.reconstruct_batch`.

        A keypoint with fewer than 2 cameras scoring `kp[..., 2] > 0` gets
        `has3d = False` and a zero point rather than the NaN `reconstruct_batch` returns.
        """
        valid = kp[:, :, 2] > 0
        xyz = rig.reconstruct_batch(kp[:, :, :2].transpose(1, 0, 2), valid.T)
        has = ~np.isnan(xyz).any(-1)
        return np.where(has[:, None], xyz, 0.0).astype(np.float32), has

    def _decode(self, info):
        with Image.open(Path(self.root) / "images" / info["file_name"]) as im:
            return np.asarray(im.convert("RGB"), np.uint8)

    def _build(self, i):
        """One sample: crops, geometry and labels for window i (Task 5 -- see module docstring)."""
        rec, host, f0 = self.windows[i]
        negative = host < 0
        rig = self._rig(self.calib_group(i))
        cam_names = list(rig.cameras)
        C, T, K, F = rig.n_cameras, self.T, len(self.kp_order), self.max_flies
        M, t = self._affine(self.calib_group(i))

        frames = self._frames(i)
        _, flies = self._window_flies(i)
        kp_full = np.zeros((F, T, C, K, 3), np.float32)
        X3 = np.zeros((F, T, K, 3), np.float32)
        has3d = np.zeros((F, T, K), bool)
        infos = {}
        if negative:
            for ti, f in enumerate(frames):
                _, inf = self._frame_labels(self._fs[(rec, f, host)], rig)
                for c in range(C):
                    if inf[c] is not None:
                        infos.setdefault((ti, c), inf[c])
        for fi, fly in enumerate(flies):
            for ti, f in enumerate(frames):
                fsv = self._fs.get((rec, f, fly))
                if fsv is None:
                    continue
                kp, inf = self._frame_labels(fsv, rig)
                kp_full[fi, ti] = kp
                X3[fi, ti], has3d[fi, ti] = self._dlt(kp, rig)
                for c in range(C):
                    if inf[c] is not None:
                        infos.setdefault((ti, c), inf[c])

        cam_valid = np.zeros((T, C), bool)
        for ti, c in infos:
            cam_valid[ti, c] = True
        for ti, f in enumerate(frames):
            present = {
                self._img[img]["file_name"].split("/")[1]
                for img, _ in _resolved_slots(self._fs[(rec, f, host)])
            }
            for c, name in enumerate(cam_names):
                if name not in present:
                    cam_valid[ti, c] = False

        if negative:
            c3 = self._fs[(rec, f0, host)].get("center3D")
            if c3 is None:
                raise ValueError(
                    f"negative frameset {rec}/Frame_{f0}: no 'center3D'. An empty window has no "
                    f"labels to place itself with, so the exporter must store the centre it "
                    f"sampled (spec 2026-09-05 §3.5)"
                )
            center = np.asarray(c3, np.float64)
        else:
            vis0 = has3d[0, 0]
            pts = X3[0, 0][vis0] if vis0.any() else np.zeros((1, 3), np.float32)
            center = 0.5 * (pts.max(0).astype(np.float64) + pts.min(0).astype(np.float64))
        if self.train and self.jitter > 0:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(i), int(self.epoch)])
            )
            center = center + rng.uniform(-self.jitter, self.jitter, size=3)
        if not self.train and self.center_shift > 0:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), 99]))
            ang = rng.uniform(0, 2 * np.pi)
            center = center + self.center_shift * np.array([np.cos(ang), np.sin(ang), 0.0])
        center = center.astype(np.float32)

        origin = np.zeros((C, 2), np.int32)
        for c in range(C):
            info = next((infos[(ti, c)][0] for ti in range(T) if (ti, c) in infos), None)
            w, h = (info["width"], info["height"]) if info else (1936, 448)
            u, v = M[c] @ center + t[c]
            origin[c] = crop_origin([u, v, 0, 0], w, h, CROP)

        crops = np.zeros((T, C, CROP, CROP, 3), np.uint8)
        donor_mask = np.zeros((T, C, CROP, CROP), bool)
        mask_store = MaskStore(self.masks_root) if self.masks_root is not None else None
        for (ti, c), (info, _ann) in infos.items():
            if not cam_valid[ti, c]:
                continue
            img = self._decode(info)
            x0, y0 = origin[c]
            crops[ti, c] = img[y0 : y0 + CROP, x0 : x0 + CROP]
            if negative or mask_store is None:
                continue
            for img_id, ann_id in _resolved_slots(self._fs[(rec, frames[ti], host)]):
                if self._img[img_id]["file_name"] == info["file_name"]:
                    m = mask_store.load(rec, cam_names[c], frames[ti], ann_id)
                    if m is not None:
                        donor_mask[ti, c] = m[y0 : y0 + CROP, x0 : x0 + CROP].astype(bool)

        t_local = np.zeros((T, C, 2), np.float32)
        for ti in range(T):
            t_local[ti] = (M @ center + t - origin).astype(np.float32)
        kp2d = kp_full[..., :2] - origin[None, None, :, None, :]
        inside = ((kp2d >= 0) & (kp2d <= CROP - 1)).all(-1)
        vis2d = (kp_full[..., 2] > 0) & inside & cam_valid[None, :, :, None]
        fly_valid = np.array([fi < len(flies) and vis2d[fi].any() for fi in range(F)])
        if not negative:
            fly_valid[0] = True
        px_scale = float(np.mean(np.sqrt((M**2).sum((1, 2)) / 2.0)))
        return {
            "crops": crops,
            "cam_valid": cam_valid,
            "M": M.astype(np.float32),
            "t_local": t_local,
            "center3D": center,
            "kp3d_local": (X3 - center).astype(np.float32) * has3d[..., None],
            "has3d": has3d,
            "kp2d": kp2d.astype(np.float32),
            "vis2d": vis2d,
            "fly_valid": fly_valid,
            "px_scale": np.float32(px_scale),
            "is_female": np.bool_(self.is_female(i)),
            "donor_mask": donor_mask,
            "crop_origin": origin,
            "fly_sex": np.array(
                [
                    self.fly_sex_code(rec, flies[fi], f0) if fi < len(flies) else SEX_UNKNOWN
                    for fi in range(F)
                ],
                np.int8,
            ),
            "unlabelled_sex": np.int8(self.unlabelled_sex(i)),
            "sample_weight": np.float32(self.weight(i)),
            "is_negative": np.bool_(negative),
        }

    def paste_window(self, i, rng):
        """Copy-paste a donor fly into every frame of window i, or `None`.

        `None` for a negative window (no host to paste onto), for a host
        with no donor pool at its own `(calib_group, sex, delta)` or the
        opposite one, or when `composite` rejects every donor drawn within
        `copy_paste.max_tries` (see `tracking.train.data.copypaste.composite`
        for the rejection conditions). The donor pool is keyed by spacing
        because a donor is composited with its own per-frame motion, so it
        must span the same `delta` as the host.
        """
        if self.copy_paste is None or self.is_negative(i):
            return None
        p = self.copy_paste
        grp, d_i = self.calib_group(i), self.delta(i)
        host_sex = _SEX_CODE.get(self._win_sex[i], SEX_UNKNOWN)
        tgt = self._build(i)
        axes = body_plane_axes(tgt["kp3d_local"][0, 0], tgt["has3d"][0, 0])
        for _ in range(p.max_tries):
            opposite = host_sex in (SEX_FEMALE, SEX_MALE) and rng.uniform() < p.opposite_sex_p
            want = (1 - host_sex) if opposite else host_sex
            other = (1 - want) if want in (SEX_FEMALE, SEX_MALE) else host_sex
            pool = [j for j in self._donors.get((grp, want, d_i), []) if j != i]
            pool = pool or [j for j in self._donors.get((grp, other, d_i), []) if j != i]
            if not pool:
                return None
            j = int(pool[rng.integers(len(pool))])
            D = sample_offset(rng, axes, p)
            out = composite(tgt, self._build(j), D, p)
            if out is not None:
                return out
        return None

    def __getitem__(self, i):
        """One training sample (`WINDOW_KEYS`) for window i.

        Draws a copy-paste sample with probability `copy_paste.p` when
        training, window i has exactly one labelled fly (so a negative,
        which has zero, is structurally excluded), and it has no
        unlabelled animal; every other case, and a `paste_window` that
        rejects every donor, falls through to `_build(i)`. The RNG seed's
        trailing `7` separates this draw from `_build`'s own jitter draw
        (same `seed`/`i`/`epoch`) so the two do not correlate.
        """
        p = self.copy_paste
        if (
            p is not None
            and self.train
            and self.n_flies(i) == 1
            and self.unlabelled_sex(i) == SEX_UNKNOWN
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(i), int(self.epoch), 7])
            )
            if rng.uniform() < p.p:
                out = self.paste_window(i, rng)
                if out is not None:
                    return out
        return self._build(i)


def window_batches(
    ds,
    batch_size,
    *,
    shuffle=True,
    seed=0,
    weights=None,
    num_workers=8,
    drop_last=True,
    workers="threads",
    pool=None,
    pool_key=None,
):
    """Batches of `batch_size` windows, sampled in the PARENT and assembled by
    `num_workers` workers.

    `workers`:
      "threads"   -- a `ThreadPoolExecutor` inside this process. Fine while
                     the Python half of `__getitem__` is not the bottleneck.
      "processes" -- a spawn `ProcessSampleLoader`
                     (`tracking.train.data.loaders`), for when sample
                     assembly is GIL-bound. Batches are byte-identical to the
                     thread path for the same (indices, seed) -- the epoch
                     travels with every task and every per-sample RNG is a
                     pure function of (dataset seed, index, epoch).

    `pool` is an already-built `ProcessSampleLoader` to reuse (rebuilding one
    per epoch would re-parse every root's instances json in every worker);
    when it is None a pool is built on first use and cached on `ds`.
    `pool_key` picks which dataset inside that pool this call addresses.

    On the thread path, `tpool.shutdown` joins only on a clean drain of
    `starts`; a consumer that stops mid-epoch releases the threads without
    blocking, instead of waiting for the abandoned generator to be
    garbage-collected.
    """
    ds.epoch = int(seed)
    rng = np.random.default_rng(seed)
    n = len(ds)
    if weights is not None:
        w = np.asarray(weights, np.float64)
        w = w / w.sum()
        idx = rng.choice(n, size=n, replace=True, p=w)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
    stop = (n // batch_size) * batch_size if drop_last else n
    starts = range(0, stop, batch_size)
    if workers == "processes":
        yield from _process_batches(
            ds,
            [[int(i) for i in idx[s : s + batch_size]] for s in starts],
            seed=int(seed),
            num_workers=num_workers,
            pool=pool,
            pool_key=pool_key,
        )
        return
    if workers != "threads":
        raise ValueError(f"workers must be 'threads' or 'processes', got {workers!r}")
    tpool = ThreadPoolExecutor(max_workers=max(1, num_workers))
    drained = False
    try:
        for s in starts:
            samples = list(tpool.map(ds.__getitem__, [int(i) for i in idx[s : s + batch_size]]))
            yield {k: np.stack([smp[k] for smp in samples]) for k in WINDOW_KEYS}
        drained = True
    finally:
        tpool.shutdown(wait=drained, cancel_futures=True)


def _process_batches(ds, index_lists, *, seed, num_workers, pool, pool_key):
    """The `workers="processes"` half of `window_batches`."""
    from tracking.train.data.loaders import ProcessSampleLoader, dataset_spec

    owned = None
    if pool is None:
        pool = getattr(ds, "_loader_pool", None)
        if pool is None or pool.num_workers != max(1, int(num_workers)):
            if pool is not None:
                pool.close()
            pool = ProcessSampleLoader({pool_key: dataset_spec(ds)}, num_workers)
            try:
                ds._loader_pool = pool
            except AttributeError:
                owned = pool
    note = getattr(ds, "note_drawn", None)
    try:
        for bidx, samples in pool.map_batches(index_lists, epoch=int(seed), key=pool_key):
            if note is not None:
                for i in bidx:
                    note(i)
            yield {k: np.stack([smp[k] for smp in samples]) for k in WINDOW_KEYS}
    finally:
        if owned is not None:
            owned.close()
