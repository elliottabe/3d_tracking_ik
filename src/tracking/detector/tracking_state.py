"""Tracked window placement: where each fly's crop is centred, over time."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

DEFAULT_MAX_REUSE_FRAMES = 8

DEFAULT_PLACEMENT_LAG = 8

WINDOW_SRC_TRACK = "track"
WINDOW_SRC_CENTERDETECT = "centerdetect"
WINDOW_SRC_REUSED = "reused"
WINDOW_SOURCE_CODE = {WINDOW_SRC_TRACK: 0, WINDOW_SRC_CENTERDETECT: 1, WINDOW_SRC_REUSED: 2}
WINDOW_SOURCE_NAME = {v: k for k, v in WINDOW_SOURCE_CODE.items()}
WINDOW_SOURCE_NONE = -1  # this fly had no window of its own this frame


def resolved_placement_lag(placement_lag) -> int:
    """`placement_lag` as a positive int (`None` -> `DEFAULT_PLACEMENT_LAG`)."""
    lag = DEFAULT_PLACEMENT_LAG if placement_lag is None else int(placement_lag)
    if lag < 1:
        raise ValueError(
            f"placement_lag must be >= 1 (1 == the serial, one-forward-per-frame "
            f"behaviour), got {lag}"
        )
    return lag


class TrackedWindowPlan(NamedTuple):
    """`plan_windows_tracked`'s return."""

    centres: np.ndarray
    assignment: np.ndarray
    source: list
    was_tracked: np.ndarray


def plan_windows_tracked(
    prev_centroids,
    cd_centres,
    *,
    merge_dist_units: float = 30.0,
    no_merge: bool = True,
    max_reuse_frames: int = DEFAULT_MAX_REUSE_FRAMES,
    reuse_frames=None,
) -> TrackedWindowPlan:
    """One frame's window plan, from TRACKING each fly's own last-known
    position rather than blind CenterDetect clustering.

    Args:
        prev_centroids: (F,3) each fly's last KNOWN-GOOD 3D centroid --
            carried forward across misses by the caller (`TrackedPlacementState`).
            A NaN row means this fly has never been tracked (recording start)
            or its track has expired (see `max_reuse_frames`).
        cd_centres: (W,3) this frame's CenterDetect-derived 3D centres
            (already through `lift_peaks_to_centres`/`cluster_centres`), or
            empty. These seed a window ONLY for a fly with no current track;
            a candidate within `merge_dist_units` of an already-tracked fly
            is dropped as a re-detection of that same fly, never a second
            window (see the module docstring: ONE threshold for both jobs).
        merge_dist_units: (1) a CenterDetect candidate within this distance
            of an already-tracked fly is a re-detection of it, not a new
            fly (dropped, never a second window); (2) when `no_merge=False`,
            two surviving windows within this distance of each other are
            merged into one shared crop. Both uses of the SAME threshold --
            see the module docstring.
        no_merge: `True` (the default) never merges two windows,
            even two different tracked flies in contact -- each keeps its
            own crop, centred on its own tracked position. `False` merges
            windows within `merge_dist_units` into one, at their mean.
        max_reuse_frames: `reuse_frames[fi] >= max_reuse_frames` expires fly
            `fi`'s track -- treated exactly like "never tracked", which lets
            a CenterDetect centre near its last position re-acquire it
            instead of being discarded as a duplicate forever.
        reuse_frames: (F,) int, how many consecutive frames (as of just
            BEFORE this one) each fly has been carried with no passing read
            since its last one. `None` == all-zero (a stateless single-frame
            call, e.g. tests) -- this function does not update it; the
            actual read only happens after this plan is built, so the
            caller (`TrackedPlacementState`) owns the counter.

    Returns:
        `TrackedWindowPlan` -- see its docstring.
    """
    prev = np.atleast_2d(np.asarray(prev_centroids, np.float64))
    n_flies = prev.shape[0]
    age = (
        np.zeros(n_flies, np.int64)
        if reuse_frames is None
        else np.asarray(reuse_frames, np.int64).reshape(n_flies)
    )
    cd = np.asarray(cd_centres, np.float64)
    cd = cd.reshape(-1, 3) if cd.size else np.zeros((0, 3))

    was_tracked = np.isfinite(prev).all(axis=1) & (age < int(max_reuse_frames))

    centres: list[np.ndarray] = []
    src: list[str] = []
    fly_lists: list[list[int]] = []
    for fi in range(n_flies):
        if was_tracked[fi]:
            centres.append(prev[fi])
            src.append(WINDOW_SRC_TRACK if int(age[fi]) == 0 else WINDOW_SRC_REUSED)
            fly_lists.append([fi])

    active = [prev[fi] for fi in range(n_flies) if was_tracked[fi]]
    for c in cd:
        if not np.isfinite(c).all():
            continue
        if active and min(float(np.linalg.norm(c - a)) for a in active) < float(merge_dist_units):
            continue  # within merge_dist_units of an active track -- the SAME fly
        centres.append(c)
        src.append(WINDOW_SRC_CENTERDETECT)
        fly_lists.append([])

    if not centres:
        return TrackedWindowPlan(
            np.zeros((0, 3), np.float32), np.full(n_flies, -1, np.int64), [], was_tracked
        )

    wc = np.asarray(centres, np.float64)
    if not no_merge and wc.shape[0] > 1:
        n = wc.shape[0]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(wc[i] - wc[j]) <= float(merge_dist_units):
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        _prec = {WINDOW_SRC_TRACK: 0, WINDOW_SRC_REUSED: 1, WINDOW_SRC_CENTERDETECT: 2}
        new_wc, new_src, new_fly_lists = [], [], []
        for members in groups.values():
            new_wc.append(wc[members].mean(axis=0))
            new_src.append(min((src[m] for m in members), key=lambda s: _prec[s]))
            fl: list[int] = []
            for m in members:
                fl.extend(fly_lists[m])
            new_fly_lists.append(fl)
        wc, src, fly_lists = np.asarray(new_wc), new_src, new_fly_lists

    assignment = np.full(n_flies, -1, np.int64)
    for widx, fl in enumerate(fly_lists):
        for fi in fl:
            assignment[fi] = widx
    return TrackedWindowPlan(wc.astype(np.float32), assignment, src, was_tracked)


class TrackedPlacementState:
    """One pass's tracked-placement state: each fly's last KNOWN-GOOD 3D
    centroid and how many consecutive frames it has been carried since.
    """

    def __init__(
        self,
        n_flies: int,
        *,
        merge_dist_units: float = 30.0,
        max_reuse_frames: int = DEFAULT_MAX_REUSE_FRAMES,
        no_merge: bool = True,
        init_centroid=None,
    ):
        self.n_flies = int(n_flies)
        self.merge_dist_units = float(merge_dist_units)
        self.max_reuse_frames = int(max_reuse_frames)
        self.no_merge = bool(no_merge)
        self.pos = np.full((self.n_flies, 3), np.nan, np.float64)
        if init_centroid is not None:
            ic = np.atleast_2d(np.asarray(init_centroid, np.float64))
            n = min(self.n_flies, ic.shape[0])
            self.pos[:n] = ic[:n]
        self.age = np.zeros(self.n_flies, np.int64)

    def centroids(self) -> np.ndarray:
        """(F,3) each fly's current last-known-good centroid (NaN if none)."""
        return self.pos.copy()

    def was_tracked(self) -> np.ndarray:
        """(F,) bool -- which flies have a LIVE track right now, i.e. before
        this frame's plan is built."""
        return np.isfinite(self.pos).all(axis=1) & (self.age < self.max_reuse_frames)

    def plan(self, cd_centres) -> TrackedWindowPlan:
        """This frame's `plan_windows_tracked` plan from the current state."""
        return plan_windows_tracked(
            self.pos,
            cd_centres,
            merge_dist_units=self.merge_dist_units,
            max_reuse_frames=self.max_reuse_frames,
            no_merge=self.no_merge,
            reuse_frames=self.age,
        )

    def note_read(self, fi: int, centroid) -> None:
        """Fly `fi`'s read PASSED at `centroid` (world units): the track
        moves there and its age resets."""
        self.pos[int(fi)] = np.asarray(centroid, np.float64)
        self.age[int(fi)] = 0

    def note_miss(self, fi: int) -> None:
        """Fly `fi`'s read MISSED: the position is kept, the age advances."""
        self.age[int(fi)] = self.age[int(fi)] + 1

    def update(self, t, centroids) -> None:
        """Batch convenience over `note_read`/`note_miss`: for each fly, a
        finite row in `centroids` is a passing read (`note_read`), a
        non-finite row is a miss (`note_miss`).
        """
        del t  # see docstring
        centroids = np.atleast_2d(np.asarray(centroids, np.float64))
        if centroids.shape[0] != self.n_flies:
            raise ValueError(
                f"centroids has {centroids.shape[0]} rows, expected {self.n_flies} (n_flies)"
            )
        for fi in range(self.n_flies):
            c = centroids[fi]
            if np.isfinite(c).all():
                self.note_read(fi, c)
            else:
                self.note_miss(fi)
