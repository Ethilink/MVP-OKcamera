"""Shared recording-mode test fakes — owned by TR1, imported by TR4/TR5/TR7.

Two things live here, mirroring T01's ``tests/fakes.py`` split (builders +
classes, no pytest fixtures — those stay in ``conftest.py`` for the T01
fakes; this module is imported directly):

- ``FakeEncoder`` — duck-types the encoder ``CaptureLoop.start_recording``
  depends on (frozen in TR1-capture-recording.md / TR2): ``write(frame)``,
  ``release()``, ``is_open``. Records every written frame IN ORDER plus an
  ordered event log, and can be told to raise on write (AC8).
- ``make_numbered_frames`` / ``decode_frame_index`` — a matched encode/decode
  pair so a frame's own 0-based index is recoverable from its pixels alone,
  with no external bookkeeping. That is what lets a test catch a dropped,
  duplicated, or reordered frame after the fact (off-by-one is otherwise
  silent — see RECORDING.md §Gotchas).
"""

from __future__ import annotations

import threading

import numpy as np


class FakeEncoder:
    """Stand-in for the encoder duck-type ``CaptureLoop.start_recording``
    consumes (``encoder.write(frame)``, ``encoder.release()``,
    ``encoder.is_open``).

    - ``written`` — every successfully written frame, in call order (a
      ``.copy()`` of what was passed in, so a caller that reuses/mutates a
      frame buffer after the call can never retroactively corrupt what this
      fake recorded — the copy is what makes ``written`` a trustworthy,
      independent record of "what was actually handed to the encoder").
    - ``events`` — ordered log of every call: ``("write", i)`` where ``i`` is
      the 0-based position of that write within ``written`` (i.e.
      ``len(written) - 1`` at the time of that call), ``("write_failed", i)``
      when ``raise_on_write`` caused a call to raise instead of recording,
      and ``("release",)``. Lets a test assert ``release()`` happened exactly
      once, and only after the last successful write (AC5).
    - ``is_open`` — ``True`` until ``release()`` is called, then ``False``.
      ``start_recording`` requires this to be ``True`` before it will accept
      the encoder (per the frozen interface).
    - ``raise_on_write`` — when truthy, ``write()`` raises ``RuntimeError``
      instead of recording the frame (the AC8 encoder-failure fake). A plain
      mutable attribute, so a test can flip it mid-recording if it wants a
      partial-success scenario, though no AC here requires that.
    """

    def __init__(self, raise_on_write: bool = False) -> None:
        self.written: list[np.ndarray] = []
        self.events: list[tuple] = []
        self.is_open = True
        self.raise_on_write = raise_on_write
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self.raise_on_write:
                self.events.append(("write_failed", len(self.written)))
                raise RuntimeError("FakeEncoder: simulated encoder write failure")
            self.written.append(frame.copy())
            self.events.append(("write", len(self.written) - 1))

    def release(self) -> None:
        with self._lock:
            self.is_open = False
            self.events.append(("release",))


# --- Numbered-frame encode/decode -------------------------------------------
#
# Encoding: frame i is filled with ONE constant BGR pixel value across the
# entire (h, w, 3) array — B = i % 256, G = (i // 256) % 256,
# R = (i // 65536) % 256 (little-endian base-256 across channels, so indices
# up to 256**3 - 1 ~= 16.7M are representable — far beyond any test's frame
# count). Because every pixel in the frame carries the identical value, ANY
# single pixel (``frame[0, 0]`` is used) recovers the full index — no
# side-channel bookkeeping is needed to know "which frame is this", which is
# exactly what lets a test detect a dropped/duplicated/reordered frame after
# it has passed through FakeCapture -> CaptureLoop -> FakeEncoder /
# published Latest: decode whatever pixels arrived and compare.


def make_numbered_frames(n: int, w: int, h: int) -> list[np.ndarray]:
    """``n`` deterministic BGR uint8 ``(h, w, 3)`` frames whose pixels encode
    their own 0-based index (see module docstring / encoding note above for
    the scheme). Pair with ``decode_frame_index`` to recover a frame's index
    from its pixels alone.
    """
    frames = []
    for i in range(n):
        b = i % 256
        g = (i // 256) % 256
        r = (i // 65536) % 256
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = b
        frame[:, :, 1] = g
        frame[:, :, 2] = r
        frames.append(frame)
    return frames


def decode_frame_index(frame: np.ndarray) -> int:
    """Inverse of ``make_numbered_frames``'s encoding: recover the 0-based
    index a frame was built with, from its own pixels (reads the top-left
    pixel — every pixel in the frame carries the same value).
    """
    b, g, r = int(frame[0, 0, 0]), int(frame[0, 0, 1]), int(frame[0, 0, 2])
    return b + g * 256 + r * 65536
