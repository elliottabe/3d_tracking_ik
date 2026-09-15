"""The single video writer used across the repo.

Writes an iterable/generator of BGR uint8 frames to an H.264 mp4 that plays
everywhere (browsers, VS Code's preview, Artifacts): libx264 + yuv420p +
``+faststart``. Encodes via imageio-ffmpeg's bundled ffmpeg; frames are BGR
(cv2 convention) and converted to RGB for the encoder. Size is taken from the
first frame; dimensions not divisible by ``macro_block_size`` are padded up to
the nearest multiple by imageio (e.g. 1080 -> 1088 at the default 16) — pass
``macro_block_size=1`` when the caller needs the output's exact pixel
dimensions preserved (verified e.g. by a downstream ffprobe check). Because
``macro_block_size=1`` disables that padding, an ODD frame dimension would
otherwise reach libx264/yuv420p directly: ffmpeg rejects it inside its own
subprocess, but imageio does not propagate that as a Python exception --
``append_data``/``close`` both return normally and the "video" is a 0-byte
file. So ``write_video`` validates evenness itself before opening the writer
whenever ``macro_block_size == 1``, and raises ``ValueError`` naming the
offending dimensions instead of leaving that silent corruption on disk.

No silent codec fallback: any mid-encode failure (a bad frame, a full disk,
whatever) propagates as-is. The cv2 ``mp4v`` fallback produces MPEG-4 Part 2,
which Chromium-based players (VS Code included) refuse to decode -- a render
that cannot produce a playable file must fail loudly rather than quietly
write one that looks like evidence and isn't. The fallback is therefore only
reachable when ffmpeg itself is genuinely unavailable (``_open_writer`` raises
``_FfmpegUnavailable`` before any frame is written) AND the caller opts in
with ``allow_cv2_fallback=True``; any other exception from opening the writer
propagates regardless of that flag.

Writes are atomic: every frame is encoded to ``<out_path stem>.tmp<ext>``
first, and only ``os.replace``d onto ``out_path`` after the LAST frame is
written successfully -- never straight to ``out_path`` (task 21, defect 2).
A 30-array-task campaign that (before task 21's defect-1 fix) raced 30
copies of this writer on the same ``out_path`` interleaved 30 H.264 streams
into one file -- every rendered video came back corrupt (``Invalid NAL unit
size``, frames stuck on the first picture). Worse, a stage's resume
predicate is typically ``exists``: a task killed mid-encode and requeued
(``ckpt-all`` is preemptible) would see that same corrupt, partial file
under the real name and SKIP re-rendering it, so the defect survives even
after the concurrency that caused it is fixed. Writing to a name nothing
downstream reads until the encode finishes -- and deleting that tmp file if
ANY exception interrupts the encode -- means ``out_path`` is either absent
or a complete, playable file, never a partial one visible under the real
name.
"""

import os

import numpy as np


class _FfmpegUnavailable(RuntimeError):
    """Raised by `_open_writer` only when no ffmpeg executable can be found at all.

    This is the ONE condition `write_video`'s `allow_cv2_fallback` path is
    allowed to catch and downgrade to a cv2 fallback -- anything else raised
    while opening or writing propagates even with the flag on.
    """


def _open_writer(out_path, fps, macro_block_size):
    """Construct the imageio/ffmpeg writer. Factored out so tests can monkeypatch it.

    imageio itself resolves/launches the ffmpeg binary LAZILY, on the first
    `append_data` call, not here -- so a missing binary would otherwise surface
    as an obscure `FileNotFoundError` out of the encode loop instead of a clear
    failure at construction time. `imageio_ffmpeg.get_ffmpeg_exe()` performs
    the same lookup imageio uses internally; calling it eagerly turns "no
    ffmpeg anywhere" into `_FfmpegUnavailable`, raised here before any frame is
    written, which is the only error `write_video`'s cv2 fallback may act on.
    """
    import imageio.v2 as imageio
    import imageio_ffmpeg

    try:
        imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as e:
        raise _FfmpegUnavailable(str(e)) from e

    return imageio.get_writer(
        out_path,
        format="FFMPEG",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=macro_block_size,
        output_params=["-movflags", "+faststart"],
    )


def _to_rgb(frame):
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame[:, :, ::-1]  # BGR (caller/cv2 convention) -> RGB (imageio)
    return np.ascontiguousarray(frame)


def write_video(out_path, frames_iter, *, fps=30, macro_block_size=16, allow_cv2_fallback=False):
    """Write BGR uint8 frames to an H.264/yuv420p mp4 at ``out_path``.

    ``frames_iter`` is any iterable of ``(H, W, 3)`` uint8 arrays. Raises
    ``ValueError`` if it yields nothing, or if ``macro_block_size == 1`` and
    the first frame's height or width is odd (libx264/yuv420p requires both
    dimensions even; with no padding to fall back on that combination
    otherwise produces a silent 0-byte file rather than an exception -- see
    the module docstring). Any failure while opening the writer or encoding a
    frame propagates, UNLESS ``allow_cv2_fallback=True`` and the failure is
    specifically ``_FfmpegUnavailable`` (no ffmpeg executable could be found
    at all) -- in that one case only, falls back to cv2's ``mp4v`` encoder,
    which may not play in browsers/VS Code. Never leaves a partial file
    visible under ``out_path`` -- see the module docstring's atomicity note
    (task 21, defect 2): a failure at any point after the tmp file is
    created removes it before propagating.
    """
    out_path = str(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames_iter = iter(frames_iter)
    try:
        first = np.ascontiguousarray(np.asarray(next(frames_iter)))
    except StopIteration:
        raise ValueError(
            f"write_video: frames_iter was empty; nothing written to {out_path}"
        ) from None

    if macro_block_size == 1:
        h, w = first.shape[0], first.shape[1]
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                f"write_video: macro_block_size=1 disables padding, but the first "
                f"frame is {w}x{h} (WxH) and libx264/yuv420p requires both dimensions "
                f"even. Pass even-sized frames, or use a macro_block_size (e.g. the "
                f"default 16) that pads up to an even multiple."
            )

    # `<stem>.tmp<ext>`, never `out_path` directly -- a resume predicate that
    # checks `exists` must never see a file here until the encode is whole
    # (module docstring, task 21 defect 2). A stray tmp from a previous
    # killed run is removed up front: `_open_writer`/cv2 both truncate on
    # open regardless, so this is defensive, not load-bearing.
    root, ext = os.path.splitext(out_path)
    tmp_path = f"{root}.tmp{ext}"
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        pass

    try:
        try:
            writer = _open_writer(tmp_path, fps, macro_block_size)
        except _FfmpegUnavailable as e:
            if not allow_cv2_fallback:
                raise
            print(
                f"[write_video] warning: ffmpeg unavailable ({e}); "
                f"falling back to cv2 mp4v -- may not play in VS Code/browsers."
            )
            _write_video_cv2_fallback(tmp_path, first, frames_iter, fps)
        else:
            try:
                writer.append_data(_to_rgb(first))
                for frame in frames_iter:
                    writer.append_data(_to_rgb(frame))
            finally:
                writer.close()
    except BaseException:
        # Mid-encode failure (bad frame, full disk, ...): the tmp file is
        # incomplete/corrupt exactly like the 30 raced writes were -- delete
        # it rather than leave it for a requeued task's `exists` check to
        # mistake for done, then propagate unchanged (no silent fallback).
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise

    os.replace(tmp_path, out_path)
    return out_path


def _write_video_cv2_fallback(out_path, first, frames_iter, fps):
    """cv2 ``mp4v`` fallback, reachable ONLY when ffmpeg itself is unavailable
    and the caller passed ``allow_cv2_fallback=True``. Produces MPEG-4 Part 2,
    which Chromium-based players (VS Code) refuse to decode.
    """
    import cv2

    h, w = first.shape[0], first.shape[1]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise OSError(f"cv2.VideoWriter failed to open {out_path}")
    try:
        writer.write(np.asarray(first))
        for frame in frames_iter:
            writer.write(np.asarray(frame))
    finally:
        writer.release()
    return str(out_path)
