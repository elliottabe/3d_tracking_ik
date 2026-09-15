"""Forward-only threaded mp4 reading, and sync-plan slot mapping.

This is the pipeline's single largest speed decision: `SlotReader` decodes
each camera's mp4 with one dedicated thread, walking it FORWARD ONLY --
`grab()` (decode, discard) through the frames a stride skips, `retrieve()`
only at the stride hit that is wanted. After the one initial seek to the
start slot, every mp4 frame is decoded at most once, ever.

The alternative -- a kept-open capture re-seeked per call -- was measured at
0.39 coarse frames/s (a ~22-hour ETA on one real recording), ~470% CPU and
~0% GPU: CPU-bound on repeated keyframe-seek + GOP redecode, not GPU-bound in
whatever forward pass the caller runs per frame. Do not go back to that
design.

`SlotReader` is therefore NOT a random-access reader: it is constructed with
a stride and a start slot, and must be called with a strictly increasing
slot sequence (`start_slot, start_slot + stride, start_slot + 2*stride, ...`)
or it raises `ValueError`. For that sequence its frames are byte-identical
to `read_window`'s -- the random-access reference reader below, which any
single slot can be read from via a seek, at the cost of being unusable for a
whole-recording pass.

`preprocess`, when given, runs per camera INSIDE each decode thread (not on
the consumer) -- moving per-camera preprocessing there instead of onto the
single consumer thread that also has to issue every model forward was
measured worth 1.87x; that placement must not move.

Camera order and canonical-slot -> mp4-position mapping follow the same
contract as the rest of `tracking.io`: cameras are named via `Order`, and
`load_sync_plan` reads `sync_plan.json` (present <-> a recording had dropped
frames on some camera) mapping canonical slots to per-camera mp4 positions;
with no plan the mapping is positional.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from tracking.io.names import Order


def load_sync_plan(session_dir) -> dict[str, Any] | None:
    """Read `<session_dir>/sync_plan.json`, or `None` if it does not exist.

    Session dirs with no dropped frames on any camera have no sync plan; the
    mapping from canonical slot to per-camera mp4 position is then purely
    positional (`slot_positions` below).

    The plan's shape (when present) is `{"cameras": {cam_name: {"positions":
    [...] } | [...] , ...}}`: for canonical slot `s`, camera `cam_name`'s mp4
    position is `positions[s]`, or the camera dropped that slot if `s` is
    beyond the end of its `positions` list (or its position entry is
    `null`).
    """
    path = os.path.join(str(session_dir), "sync_plan.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _cam_positions(plan: dict[str, Any] | None, camera: str) -> list | None:
    """The raw per-camera position list from a loaded plan, or `None` for a
    plan with no positional mapping at all (mapping is then positional)."""
    if plan is None:
        return None
    cams = plan.get("cameras", plan)
    entry = cams.get(camera)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("positions")
    return entry


def slot_positions(plan: dict[str, Any] | None, camera: str, slots: np.ndarray) -> np.ndarray:
    """Canonical slots -> this camera's mp4 frame positions.

    Returns an `int64` array the same shape as `slots`; a slot the camera
    dropped (only possible with a real plan) is `-1`, meaning "not present at
    this slot" -- callers check that against the plan's own present mask,
    they do not treat `-1` as a real frame index.

    With `plan is None`, the mapping is positional: `positions[i] ==
    slots[i]`.
    """
    slots = np.asarray(slots)
    positions = _cam_positions(plan, camera)
    if positions is None:
        return slots.astype(np.int64)
    positions = np.asarray(positions, dtype=object)
    out = np.full(slots.shape, -1, dtype=np.int64)
    flat_slots = slots.reshape(-1)
    flat_out = out.reshape(-1)
    for i, s in enumerate(flat_slots):
        s = int(s)
        if 0 <= s < len(positions) and positions[s] is not None:
            flat_out[i] = int(positions[s])
    return out


def _slot_present(plan: dict[str, Any] | None, camera: str, slot: int) -> bool:
    if plan is None:
        return True
    positions = _cam_positions(plan, camera)
    if positions is None:
        return True
    return 0 <= slot < len(positions) and positions[slot] is not None


def _slot_position(plan: dict[str, Any] | None, camera: str, slot: int) -> int:
    """One canonical slot -> this camera's mp4 frame position (positional if
    `plan is None`); caller must have already checked `_slot_present`."""
    if plan is None:
        return int(slot)
    positions = _cam_positions(plan, camera)
    return int(positions[slot])


def _cap_get(cap, prop_id: int) -> float:
    """`cap.get(prop_id)`, indirected through this module-level function.

    `_CamStream._run` uses this ONLY for the post-retrieve frame-index drift
    check (below) -- the one read a test needs to falsify to exercise that
    check. `SlotReader` runs one decoder thread PER CAMERA, all concurrently
    calling `.get()` on their own `cv2.VideoCapture` instances; a test that
    wants to force a drift must NOT monkeypatch `cv2.VideoCapture.get` on the
    class itself, because that patches every instance out from under every
    other camera's thread at once, and unpatching at teardown races threads
    that may still be leaked from an earlier, differently-timed run of the
    same test. Patching this function instead only affects the single call
    site below.
    """
    return cap.get(prop_id)


class _CamStream:
    """One camera's forward-only decode thread.

    Called with a strictly increasing sequence of canonical slots
    `start_slot, start_slot + stride, ...`. Walks the mp4 FORWARD ONLY:
    `cap.grab()` (decode, discard) through every frame the stride skips,
    `cap.retrieve()` only at the wanted stride hit -- so each mp4 frame is
    decoded at most once for the whole pass, after the one initial seek (to
    `start_slot`).

    A slot this camera's plan drops yields `frame=None, present=False`
    without touching the decode cursor; the next present slot's absolute
    target position naturally catches the cursor up across the gap via
    `grab()`, not a seek.
    """

    def __init__(
        self,
        session_dir: str,
        cam: str,
        plan: dict[str, Any] | None,
        start_slot: int,
        stride: int,
        queue_size: int = 4,
        preprocess=None,
    ) -> None:
        self.session_dir, self.cam, self.plan = str(session_dir), cam, plan
        self.preprocess = preprocess
        self.start_slot, self.stride = int(start_slot), int(stride)
        self.q: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self.H = self.W = None
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camstream-{cam}")
        self._thread.start()
        self._ready.wait()
        if self.error is not None:
            # `_ready` is set right after opening the file, before the main
            # decode loop -- so this thread can already be past its FIRST
            # drift check (start_slot) by the time we get here, not just past
            # the isOpened() check (both set `self.error` the same way, and
            # `_run` exits on its own right after -- there is no lingering
            # thread either way). This instance is never handed back to the
            # caller (the exception propagates out of this constructor), so
            # nobody else will ever call `close()` on it; calling it here
            # makes the thread's `cap.release()` a synchronous, waited-for
            # part of raising rather than an eventually-consistent side
            # effect racing whatever runs next in the caller.
            self.close()
            raise self.error

    def _run(self) -> None:
        import cv2

        cv2.setNumThreads(1)
        path = os.path.join(self.session_dir, f"{self.cam}.mp4")
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise FileNotFoundError(path)
            self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        except Exception as e:
            self.error = e
            self._ready.set()
            cap.release()
            return
        self._ready.set()
        cursor = None
        slot = self.start_slot
        try:
            while not self._stop.is_set():
                present_i = _slot_present(self.plan, self.cam, slot)
                pos_i = _slot_position(self.plan, self.cam, slot) if present_i else None
                frame = None
                if present_i:
                    if cursor is None or pos_i < cursor:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_i)  # the ONE seek (start slot)
                        cursor = pos_i
                    ok = True
                    while cursor < pos_i:  # discard the stride-skipped frames
                        ok = cap.grab()
                        if not ok:
                            break
                        cursor += 1
                    if ok:
                        ok = cap.grab()  # decode the WANTED frame (not yet retrieved)
                    if not ok:
                        cursor = None  # EOF
                    if cursor is not None:
                        ok, fr = cap.retrieve()
                        if ok:
                            # Checked on EVERY stride hit, not just the
                            # first few: a drift that only appears later in
                            # the decode (e.g. after a dropped/corrupt GOP)
                            # would otherwise run silently past whatever
                            # window this check used to stop at. A frame
                            # from the wrong time consumed downstream as if
                            # it were correct is exactly the confidently-
                            # wrong data this repo's conventions exist to
                            # catch.
                            actual = int(_cap_get(cap, cv2.CAP_PROP_POS_FRAMES)) - 1
                            if actual != pos_i:
                                raise RuntimeError(
                                    f"{self.cam}: frame-index drift at slot {slot} -- "
                                    f"expected mp4 frame {pos_i}, CAP_PROP_POS_FRAMES "
                                    f"reports {actual} (sequential grab/retrieve landed "
                                    f"on the wrong frame)"
                                )
                            frame = fr[:, :, ::-1]  # BGR -> RGB
                            cursor += 1
                        else:
                            cursor = None
                present_out = bool(present_i and frame is not None)
                pre = (
                    self.preprocess(frame)
                    if (self.preprocess is not None and present_out)
                    else None
                )
                self._put((slot, frame, present_out, pre))
                slot += self.stride
        except Exception as e:
            self.error = e
            self._put((slot, None, None, None))  # unblock a waiting `get`
        finally:
            cap.release()

    def _put(self, item) -> None:
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def get(self, slot: int):
        """The frame for canonical `slot`, discarding any earlier queued slot
        the caller skipped over (a strictly increasing caller sequence need
        not hit every `start_slot + k*stride` grid point -- see
        `SlotReader`)."""
        while True:
            got_slot, frame, present, pre = self.q.get()
            if self.error is not None:
                raise RuntimeError(f"{self.cam} reader thread failed") from self.error
            if got_slot == slot:
                return frame, present, pre
            if got_slot > slot:
                raise ValueError(
                    f"{self.cam}: reader is already past slot {slot} (at {got_slot}) -- "
                    f"SlotReader must be called with a strictly increasing slot sequence"
                )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class SlotReader:
    """One `_CamStream` thread per camera, each decoding sequentially forward
    only (see `_CamStream`). NOT a random-access reader -- see module
    docstring.

    `__call__(slot)` returns `(frames, present)` for one canonical slot, in
    `cameras` order: `present` is `(C,)` bool, false for a camera whose slot
    is past the end of its file (or dropped by the sync plan). With no
    `preprocess`, `frames` is `(C, H, W, 3)` uint8 RGB. With `preprocess`
    given, it is applied per camera INSIDE that camera's decode thread (not
    on the consumer -- moving it there is worth 1.87x) and `frames` is the
    `(C, ...)` stack of `preprocess(frame)`; an absent camera's row is
    `preprocess(zeros)`, not zeros, computed once and cached.

    Must be called with a strictly increasing slot sequence on the
    `start_slot + k*stride` grid this reader was constructed with; anything
    else raises `ValueError`.

    Also usable as a context manager (`with SlotReader(...) as reader:`),
    which calls `close()` on exit -- including when the body raises. Without
    it, a consumer that raises mid-iteration and never calls `close()`
    leaves every one of its `cameras`-many decoder threads alive
    indefinitely; a leaked decoder thread per bout would exhaust the node
    over a real session run.
    """

    def __init__(
        self,
        session_dir,
        cameras: Order | Sequence[str],
        *,
        plan: dict[str, Any] | None = None,
        start_slot: int = 0,
        stride: int = 1,
        queue_size: int = 4,
        preprocess=None,
    ) -> None:
        self.session_dir = str(session_dir)
        self.cameras = cameras if isinstance(cameras, Order) else Order(cameras)
        self.plan = plan
        self.preprocess = preprocess
        self._absent_pre = None
        self._last_slot: int | None = None
        self.streams: list[_CamStream] = []
        try:
            for c in self.cameras:
                self.streams.append(
                    _CamStream(
                        self.session_dir,
                        c,
                        plan,
                        start_slot,
                        stride,
                        queue_size=queue_size,
                        preprocess=preprocess,
                    )
                )
        except Exception:
            for s in self.streams:  # one camera failed to open -- stop the rest
                s.close()
            raise
        self.H = self.streams[0].H
        self.W = self.streams[0].W
        self._start_slot, self._stride = int(start_slot), int(stride)

    def __call__(self, slot: int):
        slot = int(slot)
        if self._last_slot is not None and slot <= self._last_slot:
            raise ValueError(
                f"SlotReader must be called with a strictly increasing slot sequence; "
                f"got {slot} after {self._last_slot}"
            )
        if (slot - self._start_slot) % self._stride != 0:
            raise ValueError(
                f"slot {slot} is not on this reader's start_slot={self._start_slot}, "
                f"stride={self._stride} grid; SlotReader must be called with a strictly "
                f"increasing slot sequence on that grid"
            )
        self._last_slot = slot
        n = len(self.cameras)
        present = np.zeros(n, bool)
        rows: list = [None] * n
        for ci, stream in enumerate(self.streams):
            frame, pres, pre = stream.get(slot)
            present[ci] = bool(pres)
            if self.preprocess is None:
                rows[ci] = frame if pres else np.zeros((self.H, self.W, 3), np.uint8)
            else:
                # An absent camera contributes preprocess(zeros), not zeros
                # itself, so a missing camera's row still shape/dtype-matches
                # the rest of the stack under whatever `preprocess` does.
                if self._absent_pre is None:
                    self._absent_pre = self.preprocess(np.zeros((self.H, self.W, 3), np.uint8))
                rows[ci] = pre if pres else self._absent_pre
        frames = np.stack(rows).astype(np.uint8) if self.preprocess is None else np.stack(rows)
        return frames, present

    def close(self) -> None:
        for stream in self.streams:
            stream.close()

    def __enter__(self) -> SlotReader:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read_window(
    session_dir,
    cameras: Order | Sequence[str],
    slot: int,
    *,
    plan: dict[str, Any] | None = None,
):
    """Random-access reference reader: any single canonical `slot`, via a
    seek per camera. `(frames (C,H,W,3) uint8 RGB, present (C,) bool)`, in
    `cameras` order -- byte-identical to what `SlotReader` yields for the
    same slot when called in a strictly increasing sequence that includes
    it.

    Not for a whole-recording pass (a seek + GOP redecode per call); see the
    module docstring for why `SlotReader` exists.
    """
    import cv2

    cams = cameras if isinstance(cameras, Order) else Order(cameras)
    slot = int(slot)
    caps = []
    H = W = None
    try:
        for c in cams:
            cap = cv2.VideoCapture(os.path.join(str(session_dir), f"{c}.mp4"))
            caps.append(cap)
            if H is None and cap.isOpened():
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if h > 0 and w > 0:
                    H, W = h, w
        if H is None or W is None:
            raise FileNotFoundError(
                f"no readable camera videos in {session_dir} for cameras {list(cams)}"
            )
        out = np.zeros((len(cams), H, W, 3), np.uint8)
        present = np.zeros(len(cams), bool)
        for ci, (c, cap) in enumerate(zip(cams, caps, strict=True)):
            if not _slot_present(plan, c, slot):
                continue
            pos = _slot_position(plan, c, slot)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, fr = cap.read()
            if ok:
                out[ci] = fr[:, :, ::-1]  # BGR -> RGB
                present[ci] = True
        return out, present
    finally:
        for cap in caps:
            cap.release()
