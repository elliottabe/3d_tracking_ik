"""The single video writer used across the repo."""

import os

import numpy as np


class _FfmpegUnavailable(RuntimeError):
    """Raised by `_open_writer` only when no ffmpeg executable can be found at all."""


def _open_writer(out_path, fps, macro_block_size):
    """Construct the imageio/ffmpeg writer. Factored out so tests can monkeypatch it."""
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
    """Write BGR uint8 frames to an H.264/yuv420p mp4 at ``out_path``."""
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
