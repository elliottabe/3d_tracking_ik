"""Forward-only threaded mp4 reading, and sync-plan slot mapping."""

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
    """Read `<session_dir>/sync_plan.json`, or `None` if it does not exist."""
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
    """Canonical slots -> this camera's mp4 frame positions."""
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
    """`cap.get(prop_id)`, indirected through this module-level function."""
    return cap.get(prop_id)


class _CamStream:
    """One camera's forward-only decode thread."""

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
