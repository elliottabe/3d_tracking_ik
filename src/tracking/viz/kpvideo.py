"""The kp-video render: any per-bout keypoint artifact drawn on the raw video.

This is the pipeline's primary verification instrument (see CLAUDE.md: "a
change that moves numbers is not done until a figure shows what it did on
real frames"), and it is designed to run automatically after every bout's
lift. It renders one panel per camera plus a 3D world panel, with each fly's
lifted 3D keypoints reprojected onto its own camera's frame.

EXPECTATION -- what a correct render looks like: every drawn keypoint sits ON
the fly's body in every camera panel at every frame, leg-chain lines follow
one leg's segments proximal to distal with no line jumping between legs or
between cameras, and fly0/fly1 (cyan/orange, or their sex label when
`sex.json`/`fly_sex` is available) never swap identity across the bout. A
keypoint that looks plausible but sits on the WRONG camera's image, or a
"leg" segment that bridges two different legs, is exactly the class of bug
CLAUDE.md documents (keypoint-order and camera-order traps) -- both survive a
quick glance and only fail a check against a rigid/anatomical invariant.

SCHEMA. `<bout_dir>/fly<f>/{kp3d,kp3d_filt,fitted}.npz` (this plan produces
only `kp3d.npz`; the other two names are accepted now so this render does not
need to change when they arrive) plus `<bout_dir>/mvq_meta.json`
(`bout_start_frame`, `n_frames`, `fly_sex`), describing the BOUT, not one fly.

ORDERING. Every array is read through `tracking.io.artifacts` (`BoutArtifact`
for `kp3d`, `load_npz` directly for the other two names -- both refuse an
artifact with no order stamp; there is no "fall back and announce" path,
because an unstamped artifact is refused at read time). Keypoints are indexed
by
NAME (`tracking.viz.colors.keypoint_groups`/`leg_chains`, `Order.index`),
never by a bare integer. Because this render always REPROJECTS (there is no
2D source among `kp3d`/`kp3d_filt`/`fitted`), the rig's camera order is
checked against the cameras actually present in `video_dir` before a single
frame is drawn, and the render refuses to start on a mismatch -- one camera's
keypoints on another camera's image still looks almost plausible, which is
how this class of bug survives review.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from tracking.geometry.rig import CameraRig
from tracking.io.artifacts import BoutArtifact, load_npz
from tracking.io.names import Order, as_order, require_same
from tracking.io.video import SlotReader
from tracking.viz.colors import PALETTE, fly_colour, keypoint_groups, leg_chains
from tracking.viz.video import write_video

SOURCES = ("kp3d", "kp3d_filt", "fitted")
_SOURCE_FILENAME = {"kp3d": "kp3d.npz", "kp3d_filt": "kp3d_filt.npz", "fitted": "fitted.npz"}

_COLS = 4
_CELL = 220
# These frames are 1936 x 448 -- long, shallow arena strips. The height is
# already small enough to keep nearly all of, so the crop tracks the flies in
# WIDTH only and takes a fixed slab of the height. Chasing them vertically in a
# 448 px strip buys no context and is most of the residual wobble.
_CROP_MARGIN = 1.5  # room around the flies' bounding box, horizontally
_CROP_KEEP_H = 1.0  # keep the FULL frame height; only width tracks the flies
_CELL_MIN_CROP = 260  # never magnify a lone fly into blur
_CROP_SMOOTH = 61  # frames; follow the flies' drift, not their per-frame jitter
# Re-render the 3D axes only once the panning view has drifted this fraction of
# its span. Redrawing them every frame would hand back the 95x saving that
# projecting the points ourselves buys; see `_Panel3D`.
_VIEW3D_REDRAW = 0.04
_BANNER_H = 40


def _video_camera_order(video_dir) -> Order:
    """Camera names actually present in `video_dir`, glob order (`Cam*.mp4`)."""
    paths = sorted(Path(video_dir).glob("Cam*.mp4"))
    if not paths:
        raise FileNotFoundError(f"no Cam*.mp4 files under {video_dir}")
    return Order(p.stem for p in paths)


def _discover_flies(bout_dir: Path, source: str) -> list[int]:
    """`[0, 1, ...]` for every `fly<d>/` under `bout_dir` holding `source`'s file."""
    fname = _SOURCE_FILENAME[source]
    out = []
    for p in sorted(bout_dir.glob("fly*")):
        if p.is_dir() and p.name[3:].isdigit() and (p / fname).exists():
            out.append(int(p.name[3:]))
    if not out:
        raise FileNotFoundError(f"no fly<d>/{fname} under {bout_dir}")
    return out


def _load_fly_kp3d(bout_dir: Path, fly: int, source: str, kp: Order, rig: CameraRig):
    """One fly's `(T, K, 3)` kp3d array, read through the order-verifying path."""
    if source == "kp3d":
        art = BoutArtifact.open(bout_dir, fly, kp=kp, cams=rig.cameras)
        return np.asarray(art.kp3d.values)
    path = Path(bout_dir) / f"fly{fly}" / _SOURCE_FILENAME[source]
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist (source={source!r} names that file)")
    data = load_npz(path, kp=kp)
    return np.asarray(data["kp3d"])


class _Panel3D:
    """The 3D world panel: matplotlib axes for the frame, cv2 for the data.

    Replaces a hand-rolled isometric projection that drew the world UPSIDE
    DOWN -- `y3 = y2*cos(b) - z*sin(b)` made larger z smaller, and the panel
    fit flipped again, so the two flips did not cancel and every fly rendered
    dorsal-side-down. Nothing caught it because nothing asked which way was up.

    SPEED. Matplotlib draws the axes, grid, ticks and labels correctly; it is
    also slow -- a full `canvas.draw()` per frame measures 37.1 ms, i.e. 74.5 s
    on a 2007-frame bout. So the axes are rendered ONCE into a background
    image, and each frame projects the points with mplot3d's OWN projection
    matrix (`proj3d.proj_transform` + `ax.transData`) and draws them with cv2
    over a copy. Measured: 0.39 ms/frame, 0.8 s for the bout -- 95x.

    Blitting via `ax.draw_artist` was tried first and REJECTED: it is 15x
    faster but does not re-run the 3D projection, so markers keep the screen
    positions they had at the last full draw and stop tracking the data. The
    orientation test caught it. Projecting explicitly is the honest version of
    the same idea -- matplotlib's matrix, our rasteriser.

    The view PANS at a fixed span (see `_view3d_path`). Re-rendering the
    background on every pan would give the cost straight back, so it is
    re-rendered only when the centre has drifted past `_VIEW3D_REDRAW` of the
    span, and the points are always projected with the limits the cached
    background was drawn at -- so markers can never drift against their grid.
    """

    def __init__(self, kp: Order, chains, limits, size: int, labels):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.size = (size, size)
        self.chains = [[kp.index(n) for n in chain] for chain in chains]
        self.labels = dict(labels)
        self.fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasAgg(self.fig)
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = limits
        self.ax.set_xlim(xlo, xhi)
        self.ax.set_ylim(ylo, yhi)
        self.ax.set_zlim(zlo, zhi)
        self.ax.set_xlabel("x (units)", fontsize=6)
        self.ax.set_ylabel("y", fontsize=6)
        self.ax.set_zlabel("z", fontsize=6)
        self.ax.tick_params(labelsize=5)
        self.ax.set_title("3D world", fontsize=7)
        for key, colour_bgr in self.labels.items():
            b, g, r = colour_bgr
            self.ax.plot([], [], [], color=(r / 255.0, g / 255.0, b / 255.0), label=str(key))
        self.ax.legend(fontsize=6, loc="upper left")
        self._centre = None
        self._span = None
        self._render_background()

    def _render_background(self) -> None:
        """Rasterise the axes once; cache the image and the projection with it."""
        from mpl_toolkits.mplot3d import proj3d

        self._proj3d = proj3d
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        self._bg = rgba[:, :, :3][:, :, ::-1].copy()
        self._H = self._bg.shape[0]
        self._M = self.ax.get_proj()

    def set_centre(self, centre, span) -> None:
        """Pan to `centre` at fixed `span`, re-rendering the axes only on drift."""
        centre = np.asarray(centre, float)
        span = np.asarray(span, float)
        if not np.isfinite(centre).all():
            return
        if self._centre is not None and np.all(
            np.abs(centre - self._centre) < _VIEW3D_REDRAW * span
        ):
            return
        for setter, c, h in zip(
            (self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim), centre, span / 2.0, strict=True
        ):
            setter(c - h, c + h)
        self._centre, self._span = centre, span
        self._render_background()

    def _to_px(self, pts):
        """`(K, 3)` world -> `(K, 2)` pixel coords in the cached background."""
        x, y, z = np.asarray(pts, float).T
        xs, ys, _ = self._proj3d.proj_transform(x, y, z, self._M)
        disp = self.ax.transData.transform(np.column_stack([xs, ys]))
        return np.column_stack([disp[:, 0], self._H - disp[:, 1]])

    def draw(self, points_by_key) -> np.ndarray:
        """`{key: (K, 3) world points or None}` -> a `(size, size, 3)` BGR tile."""
        img = self._bg.copy()
        for key, colour in self.labels.items():
            p = points_by_key.get(key)
            if p is None:
                continue
            px = self._to_px(p)
            ok = np.isfinite(px).all(-1)
            for chain in self.chains:
                pts = [
                    (int(round(px[i, 0])), int(round(px[i, 1])))
                    for i in chain
                    if i < len(px) and ok[i]
                ]
                for a, b in zip(pts[:-1], pts[1:], strict=False):
                    cv2.line(img, a, b, colour, 1, cv2.LINE_AA)
            for i in np.flatnonzero(ok):
                cv2.circle(img, (int(round(px[i, 0])), int(round(px[i, 1]))), 2, colour, -1)
        return cv2.resize(img, self.size)


def _world_limits(arrays, pct: float = 1.0, pad_frac: float = 0.10):
    """Fixed x/y/z limits over the drawn points, ROBUST to outlier frames.

    Percentile bounds, not min/max. Measured on bout 28: min/max gives a z
    range of 90 units while the two flies actually span 25 -- a handful of
    frames where a fit went wrong (the female's end-of-bout tangle) stretch
    the axis more than three-fold and leave the flies as specks for the whole
    render. Clipping at the 1st/99th percentile keeps the view on the animals;
    a point outside simply draws outside the axes.

    Computed ONCE so the panel does not rescale frame to frame -- a panel whose
    axes move cannot be read as a trajectory.
    """
    pts = np.concatenate([a.reshape(-1, 3) for a in arrays], axis=0) if arrays else np.zeros((0, 3))
    finite = pts[np.isfinite(pts).all(-1)]
    if len(finite) == 0:
        return ((0.0, 1.0),) * 3
    lo = np.percentile(finite, pct, axis=0)
    hi = np.percentile(finite, 100.0 - pct, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * pad_frac
    return tuple((float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(3))


def _view3d_path(arrays, T: int, margin: float = 1.6, pct: float = 95.0):
    """`(centres (T,3), span (3,))` for a 3D view that PANS but never rescales.

    Fixed limits over a whole bout are unusable here: the flies roam the arena,
    so the union of 2007 frames spans ~75 units in z while the two animals span
    ~25 at any instant, leaving them as specks. Moving the limits per frame
    would fix the size and break the reading -- a panel that rescales cannot be
    compared across frames.

    So the SPAN is constant (the `pct`th percentile of per-frame extents, times
    `margin`) and only the CENTRE moves, smoothed like the 2D crop. The panel
    pans with the flies at a fixed scale.
    """
    cen = np.full((T, 3), np.nan)
    ext = np.full((T, 3), np.nan)
    for t in range(T):
        here = [a[t] for a in arrays if t < a.shape[0]]
        if not here:
            continue
        pts = np.concatenate(here, axis=0)
        fin = pts[np.isfinite(pts).all(-1)]
        if not len(fin):
            continue
        lo, hi = fin.min(0), fin.max(0)
        cen[t] = (lo + hi) / 2.0
        ext[t] = hi - lo
    for j in range(3):
        col = cen[:, j]
        good = np.flatnonzero(np.isfinite(col))
        if not len(good):
            col[:] = 0.0
            continue
        col[: good[0]] = col[good[0]]
        col[good[-1] + 1 :] = col[good[-1]]
        for i in range(good[0] + 1, good[-1] + 1):
            if not np.isfinite(col[i]):
                col[i] = col[i - 1]
        cen[:, j] = _smooth(col, _CROP_SMOOTH)
    finite_ext = ext[np.isfinite(ext).all(-1)]
    span = np.percentile(finite_ext, pct, axis=0) * margin if len(finite_ext) else np.ones(3)
    return cen, np.maximum(span, 1e-6)


def _skeleton_chains(kp: Order):
    """Leg chains PLUS the body axis, the head, and each wing.

    `leg_chains` alone gives six legs and nothing joining them, so the 3D
    panel reads as a cloud of dots with six spidery arms and no fly in the
    middle. These extra chains are what make it legible as an animal. Every
    name is filtered against `kp`, so an anatomy missing one simply loses that
    segment rather than raising.
    """
    extra = [
        ["Antenna_Base", "Scutellum", "Abd_A4", "Abd_tip"],  # body axis
        ["EyeL", "Antenna_Base", "EyeR"],  # head
        ["WingL_base", "WingL_V12", "WingL_V13"],  # left wing
        ["WingR_base", "WingR_V12", "WingR_V13"],  # right wing
        ["WingL_base", "Scutellum", "WingR_base"],  # wing bases to thorax
    ]
    out = list(leg_chains(kp))
    for chain in extra:
        present = [n for n in chain if n in kp]
        if len(present) >= 2:
            out.append(present)
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, over a `(T,)` path.

    Edges are padded by repetition rather than shrinking the window, so the
    first and last frames are not pulled toward the middle of the bout.
    """
    if window <= 1 or len(values) <= 2:
        return values
    w = int(min(window, len(values)))
    if w % 2 == 0:
        w -= 1
    pad = w // 2
    padded = np.concatenate([np.repeat(values[:1], pad), values, np.repeat(values[-1:], pad)])
    return np.convolve(padded, np.ones(w, float) / w, mode="valid")


def _crop_path(rig, kp3d_by_fly, cmp_kp3d, T: int, reader_hw=None):
    """`{camera_index: (cx, cy, half_w, half_h)}`, each `(T,)` and SMOOTHED.

    Computed for every frame up front so the render loop only looks it up:
    a window recomputed per frame chases the flies' jitter and the panel
    shakes. Measured on bout 28, camera 0: the worst frame-to-frame jump falls
    from 52.8 px to 1.21 px once smoothed.

    The window SIZE is constant for the whole render (the 90th percentile of
    what each frame needs) and only its CENTRE moves. A size that varies would
    vary the zoom, and with it the displayed height and the flies' apparent
    size. The height is the full frame (`_CROP_KEEP_H`).

    A frame with no finite point inherits the previous frame's box rather than
    snapping to the image centre -- a crop that jumps to the middle whenever
    tracking drops out is exactly the jitter this removes, and the female has
    a 472-frame gap in this bout.
    """
    arrays = [a for a in (*kp3d_by_fly.values(), *cmp_kp3d.values())]
    # Project each fly's WHOLE trajectory once, not once per (frame, camera):
    # `project` returns every camera anyway, so a per-camera loop repeats the
    # same einsum seven times over.
    projected = [np.asarray(rig.project(a[:T])) for a in arrays]
    out = {}
    for ci in range(len(rig.cameras)):
        cx = np.full(T, np.nan)
        cy = np.full(T, np.nan)
        hw = np.full(T, np.nan)
        hh = np.zeros(T)  # unused: height is a fixed slab
        for t in range(T):
            here = [projected[i][t][:, ci, :] for i, a in enumerate(arrays) if t < a.shape[0]]
            if not here:
                continue
            allp = np.concatenate(here, axis=0)
            fin = allp[np.isfinite(allp).all(-1)]
            if not len(fin):
                continue
            lo, hi = fin.min(0), fin.max(0)
            cx[t], cy[t] = (lo + hi) / 2.0
            hw[t] = max((hi - lo)[0] * _CROP_MARGIN, _CELL_MIN_CROP) / 2.0
        for arr in (cx, cy, hw):
            good = np.flatnonzero(np.isfinite(arr))
            if not len(good):
                arr[:] = 0.0
                continue
            arr[: good[0]] = arr[good[0]]
            arr[good[-1] + 1 :] = arr[good[-1]]
            for i in range(good[0] + 1, good[-1] + 1):
                if not np.isfinite(arr[i]):
                    arr[i] = arr[i - 1]
        # ONE half-width for the whole render, not a smoothed per-frame one.
        # The window's SIZE sets the zoom (`_CELL / crop_width`), the zoom sets
        # the displayed height (`frame_height * zoom`), and so a width that
        # changes frame to frame makes the panel breathe vertically and the
        # flies change size -- even with the full frame height kept, which is
        # what made the height look like it was still being cropped. Fixed
        # size, moving centre: it pans, exactly like the 3D view.
        finite_hw = hw[np.isfinite(hw)]
        fixed_hw = float(np.percentile(finite_hw, 90.0)) if len(finite_hw) else _CELL_MIN_CROP / 2
        out[ci] = (
            _smooth(cx, _CROP_SMOOTH),
            _smooth(cy, _CROP_SMOOTH),
            np.full(T, fixed_hw),
            hh,
        )
    return out


def _group_colour_fn(kp: Order):
    groups = keypoint_groups(kp)
    name_to_group = {n: g for g, names in groups.items() for n in names}
    group_colour = {g: PALETTE[g] for g in groups}

    def f(name: str) -> tuple[int, int, int]:
        return group_colour[name_to_group[name]]

    return f


def _finite_px(coords2d: np.ndarray, kp: Order, bounds: tuple[int, int]) -> dict:
    """`name -> (x, y)` int pixel for every finite, in-bounds keypoint."""
    h, w = bounds
    shown = {}
    for k, name in enumerate(kp.names):
        p = coords2d[k]
        if not np.isfinite(p).all():
            continue
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if 0 <= x < w and 0 <= y < h:
            shown[name] = (x, y)
    return shown


def _draw_skeleton(
    img: np.ndarray,
    shown: dict,
    chains: list[list[str]],
    *,
    colour_by: str,
    base_colour: tuple[int, int, int],
    group_colour_of,
    override: tuple[int, int, int] | None = None,
    radius: int = 2,
) -> None:
    """Draw leg-chain lines only between ADJACENT PRESENT joints, then points.

    A missing joint never draws a fabricated segment -- see the module and
    CLAUDE.md's warning that an invented segment in a verification figure is
    worse than a gap.
    """
    line_colour = (
        override
        if override is not None
        else (PALETTE["legs"] if colour_by == "group" else base_colour)
    )
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:], strict=False):
            if a in shown and b in shown:
                cv2.line(img, shown[a], shown[b], line_colour, 1, cv2.LINE_AA)
    for name, p in shown.items():
        colour = (
            override
            if override is not None
            else (group_colour_of(name) if colour_by == "group" else base_colour)
        )
        cv2.circle(img, p, radius, colour, -1)


def render_kpvideo(
    bout_dir,
    *,
    recording=None,
    rig: CameraRig,
    kp_order,
    video_dir=None,
    source: str = "kp3d",
    out_path=None,
    compare=None,
    colour_by: str = "fly",
    fps: float = 30,
) -> str:
    """Render the standard kp-video for one bout: N camera panels + 3D panel.

    `video_dir` holds the `Cam*.mp4` files (`recording.session_dir` in
    production); `recording` may be `None` when `video_dir` is given
    explicitly. `source` picks which per-fly array is drawn (`kp3d`,
    `kp3d_filt`, or `fitted`); `compare=<other bout_dir>` draws that run's
    same-source arm underneath in white. `colour_by="fly"` (default) colours
    each fly cyan/orange; `colour_by="group"` colours by anatomy instead.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")

    bout_dir = Path(bout_dir)
    if video_dir is None:
        if recording is None:
            raise ValueError("render_kpvideo needs video_dir, or recording to derive it from")
        video_dir = recording.session_dir
    video_dir = Path(video_dir)

    kp = as_order(kp_order)

    # This render always reprojects (kp3d/kp3d_filt/fitted are 3D-only), so
    # the rig's camera order must equal the cameras actually on disk -- see
    # the module docstring's camera-order-trap warning.
    require_same(rig.cameras, _video_camera_order(video_dir), what="camera")

    meta_path = bout_dir / "mvq_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    # `bout_start_frame` says WHERE IN THE RECORDING this bout's frame 0 sits.
    # Defaulting it to 0 does not degrade the render -- it silently plays the
    # wrong footage. A bout starting at frame 446306 rendered from 0 shows the
    # flies as they were 446306 frames earlier, with this bout's keypoints
    # drawn confidently on top of them, and the overlay still LOOKS like a
    # skeleton. That shipped once; it is a hard failure now.
    if "bout_start_frame" not in meta:
        raise FileNotFoundError(
            f"{meta_path} is missing or has no `bout_start_frame`, so this render "
            f"cannot know which video frames belong to this bout. Refusing rather "
            f"than defaulting to 0, which would draw this bout's keypoints over a "
            f"different part of the recording and look plausible doing it."
        )
    start = int(meta["bout_start_frame"])
    fly_sex = meta.get("fly_sex", {}) or {}

    flies = _discover_flies(bout_dir, source)
    kp3d_by_fly = {fly: _load_fly_kp3d(bout_dir, fly, source, kp, rig) for fly in flies}

    T = min(arr.shape[0] for arr in kp3d_by_fly.values())
    if meta.get("n_frames") is not None:
        T = min(T, int(meta["n_frames"]))

    cmp_kp3d: dict[int, np.ndarray] = {}
    if compare is not None:
        cmp_bout_dir = Path(compare)
        for fly in _discover_flies(cmp_bout_dir, source):
            cmp_kp3d[fly] = _load_fly_kp3d(cmp_bout_dir, fly, source, kp, rig)

    labels = {
        fly: (f"fly{fly} {fly_sex[f'fly{fly}']}" if f"fly{fly}" in fly_sex else f"fly{fly}")
        for fly in flies
    }

    chains = leg_chains(kp)
    group_colour_of = _group_colour_fn(kp)

    # Fix the 3D panel's axes once, over every frame/fly that will be drawn,
    # so the panel does not rescale frame to frame.
    all_pts = [arr[:T] for arr in kp3d_by_fly.values()]
    all_pts += [arr[:T] for arr in cmp_kp3d.values()]
    panel_labels = {f"fly{f}": fly_colour(f) for f in sorted(kp3d_by_fly)}
    panel_labels.update({f"cmp{f}": PALETTE["white"] for f in sorted(cmp_kp3d)})
    panel3d_renderer = _Panel3D(
        kp, _skeleton_chains(kp), _world_limits(all_pts), _CELL, panel_labels
    )

    # Crop path, computed for EVERY frame up front and then smoothed.
    #
    # Computing it per frame inside the loop makes the window chase each
    # frame's noise, and the panel shakes. Smoothing needs the whole path, so
    # it cannot be done streaming -- hence the precompute. Cost is one
    # projection per camera per frame, which is trivial beside decoding.
    crop_path = _crop_path(rig, kp3d_by_fly, cmp_kp3d, T, reader_hw=None)
    view3d_centres, view3d_span = _view3d_path([*kp3d_by_fly.values(), *cmp_kp3d.values()], T)

    n_tiles = len(rig.cameras) + 1
    rows = (n_tiles + _COLS - 1) // _COLS
    # _BANNER_H and _CELL are both even, so W and H are always even -- safe
    # for macro_block_size=1 below (see its ValueError on odd dimensions).
    W, H = _COLS * _CELL, _BANNER_H + rows * _CELL

    if out_path is None:
        out_path = bout_dir / f"kpvideo_{source}.mp4"

    def frames():
        with SlotReader(video_dir, rig.cameras, start_slot=start, stride=1) as reader:
            for t in range(T):
                slot = start + t
                frames_rgb, present = reader(slot)
                canvas = np.zeros((H, W, 3), np.uint8)
                tiles = []
                for ci, cam in enumerate(rig.cameras.names):
                    frame_rgb = frames_rgb[ci]
                    frame_bgr = (
                        frame_rgb[:, :, ::-1]
                        if present[ci]
                        else np.zeros((reader.H, reader.W, 3), np.uint8)
                    )
                    h0, w0 = frame_bgr.shape[:2]

                    # The crop for this camera/frame comes from the SMOOTHED
                    # path computed before the loop: a window that recomputes
                    # itself each frame chases the flies' jitter and the panel
                    # shakes. Wider than tall, because these arenas are long and
                    # the flies travel along them.
                    pcx, _pcy, phw, _phh = crop_path[ci]
                    cx = float(pcx[t])
                    half_w = float(np.clip(phw[t], _CELL_MIN_CROP / 2, w0 / 2))
                    # Height: the full frame (see _CROP_KEEP_H), never tracked.
                    cy = h0 / 2.0
                    half_h = h0 * _CROP_KEEP_H / 2.0
                    # Clamp the window's ORIGIN and keep its SIZE. Clipping
                    # the far edge instead makes
                    # the crop narrower whenever the flies approach a frame
                    # boundary; the zoom is `_CELL / crop_width`, so a narrower
                    # crop zooms IN and the displayed height grows with it --
                    # measured 145 px -> 152 px between two frames, which reads
                    # as the panel height changing even though the height is
                    # never cropped.
                    crop_w = int(min(round(2 * half_w), w0))
                    crop_h = int(min(round(2 * half_h), h0))
                    x0 = int(np.clip(round(cx - half_w), 0, max(w0 - crop_w, 0)))
                    y0 = int(np.clip(round(cy - half_h), 0, max(h0 - crop_h, 0)))
                    x1, y1 = x0 + crop_w, y0 + crop_h
                    crop = frame_bgr[y0:y1, x0:x1]
                    tile = np.zeros((_CELL, _CELL, 3), np.uint8)
                    if crop.size:
                        zoom = min(_CELL / crop.shape[1], _CELL / crop.shape[0])
                        disp = cv2.resize(
                            crop,
                            (
                                max(int(round(crop.shape[1] * zoom)), 1),
                                max(int(round(crop.shape[0] * zoom)), 1),
                            ),
                        )
                        tile[: disp.shape[0], : disp.shape[1]] = disp
                        frame_bounds = disp.shape[:2]
                    else:
                        zoom, frame_bounds = 1.0, (0, 0)

                    def _to_tile(uv, _x0=x0, _y0=y0, _z=zoom):
                        return (np.asarray(uv, float) - np.array([_x0, _y0], float)) * _z

                    for _fly, arr in sorted(cmp_kp3d.items()):
                        if t >= arr.shape[0]:
                            continue
                        uv = _to_tile(np.asarray(rig.project(arr[t]))[:, ci, :])
                        shown = _finite_px(uv, kp, frame_bounds)
                        _draw_skeleton(
                            tile,
                            shown,
                            chains,
                            colour_by=colour_by,
                            base_colour=PALETTE["white"],
                            group_colour_of=group_colour_of,
                            override=PALETTE["white"],
                        )
                    for fly, arr in sorted(kp3d_by_fly.items()):
                        uv = _to_tile(np.asarray(rig.project(arr[t]))[:, ci, :])
                        shown = _finite_px(uv, kp, frame_bounds)
                        _draw_skeleton(
                            tile,
                            shown,
                            chains,
                            colour_by=colour_by,
                            base_colour=fly_colour(fly),
                            group_colour_of=group_colour_of,
                        )
                    cv2.putText(
                        tile,
                        cam,
                        (6, _CELL - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        PALETTE["white"],
                        1,
                        cv2.LINE_AA,
                    )
                    tiles.append(tile)

                pts = {
                    f"fly{fly}": (arr[t] if t < arr.shape[0] else None)
                    for fly, arr in sorted(kp3d_by_fly.items())
                }
                pts.update(
                    {
                        f"cmp{fly}": (arr[t] if t < arr.shape[0] else None)
                        for fly, arr in sorted(cmp_kp3d.items())
                    }
                )
                panel3d_renderer.set_centre(view3d_centres[t], view3d_span)
                tiles.append(panel3d_renderer.draw(pts))

                for i, tile in enumerate(tiles):
                    r, c = divmod(i, _COLS)
                    canvas[
                        _BANNER_H + r * _CELL : _BANNER_H + (r + 1) * _CELL,
                        c * _CELL : (c + 1) * _CELL,
                    ] = tile

                statuses = []
                for fly in flies:
                    arr = kp3d_by_fly[fly]
                    ok = t < arr.shape[0] and np.isfinite(arr[t]).any()
                    statuses.append(f"{labels[fly]}: {'ok' if ok else 'MISSING'}")
                present_str = "  ".join(statuses)
                header = f"{bout_dir.name}  source={source}  colour-by={colour_by}" + (
                    f"  white=compare({source})" if cmp_kp3d else ""
                )
                cv2.putText(
                    canvas,
                    f"frame {slot}  (local {t}/{T})   {present_str}",
                    (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    header,
                    (8, 33),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (170, 200, 235),
                    1,
                    cv2.LINE_AA,
                )
                yield canvas

    # macro_block_size=1: this render's whole point is pixel-accurate
    # correspondence between world/2D keypoints and the encoded frame -- the
    # default block-aligned padding does not merely pad, imageio/ffmpeg
    # RESIZES (stretches) a non-block-aligned frame to hit the target size,
    # which silently shifts every drawn pixel a few percent off the position
    # this function computed. W/H are always even (see above), so this never
    # hits write_video's odd-dimension guard.
    return write_video(out_path, frames(), fps=fps, macro_block_size=1)
