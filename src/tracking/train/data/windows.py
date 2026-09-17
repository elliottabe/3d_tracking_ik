"""Window enumeration and metadata over the unified training root, for the mvq model.

A window is (recording, host fly, start frame f0, spacing delta): T frames of
the host spaced `delta` apart, all cameras, to be cropped around one
window-level 3D center by `_build` (Task 5, not implemented here). Camera
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
from pathlib import Path

import numpy as np

from tracking.curate import schema
from tracking.detector.mvq.slots import SEX_FEMALE, SEX_MALE, SEX_PRESENT_UNKNOWN, SEX_UNKNOWN
from tracking.geometry.rig import CameraRig

CROP = 448
WINDOW_KEYS = (
    "crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
    "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "prompt_mask", "crop_origin",
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
            all_entries.append((rec, frame, fly, fsv))

        self.pair_deltas = tuple(dict.fromkeys(int(d) for d in (pair_deltas or (1,))))
        self.windows, self.win_delta, self._win_fs = [], [], []
        self._win_index = collections.defaultdict(list)
        if self.T == 1:
            for rec, frame, fly, fsv in all_entries:
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
        self._rigs = {}

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
        """`"anchor"` | `"partner"` | `"negative"`: the pairing role of this window.

        Always `"negative"` for an empty window, regardless of which side of
        its own delta-pair the frameset's own `role` field names it as -- that
        finer distinction is `_negative_pairs`' bookkeeping, not this window's
        public role.
        """
        if self.is_negative(i):
            return "negative"
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

    def camera_names(self, i):
        """The window's camera-axis names, in the SAME order as a future `crops`/`kp2d`/`M`."""
        return self._rig(self.calib_group(i)).cameras

    def _full_labels(self, fsv, rig):
        """Per camera (BY NAME) full-frame (K, 3) labels for one frameset's resolved slots."""
        cam_row = {n: idx for idx, n in enumerate(rig.cameras)}
        kp = np.zeros((rig.n_cameras, len(self.kp_order), 3), np.float32)
        for img_id, ann_id in _resolved_slots(fsv):
            info, ann = self._img[img_id], self._ann[ann_id]
            c = cam_row.get(info["file_name"].split("/")[1])
            if c is None:
                continue
            k = np.asarray(ann["keypoints"], np.float32)
            if k.size == len(self.kp_order) * 3:
                kp[c] = k.reshape(-1, 3)
        return kp

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
