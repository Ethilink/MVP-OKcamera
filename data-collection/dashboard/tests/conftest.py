"""Shared fixtures — the test vocabulary T02/T03/T05 build against.

Two access styles, same objects:

    def test_x(fake_frame, fake_dets, FakeDetector, FakeCapture):
        frame = fake_frame(1920, 1080)
        dets = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))
        det = FakeDetector(predictions=[dets])
        cap = FakeCapture(frames=[frame])

or import the builders/classes directly from ``tests.fakes``.
"""

from __future__ import annotations

import pytest

from tests.fakes import FakeCapture as _FakeCapture
from tests.fakes import FakeDetector as _FakeDetector
from tests.fakes import make_fake_dets, make_fake_frame


@pytest.fixture
def fake_frame():
    """Factory: ``fake_frame(w, h) -> BGR uint8 (h, w, 3)`` gradient."""
    return make_fake_frame


@pytest.fixture
def fake_dets():
    """Factory: ``fake_dets(boxes, masks=..., confidences=..., size=(1920,1080))``."""
    return make_fake_dets


@pytest.fixture
def FakeDetector():  # noqa: N802 - exposes the class under its own name
    return _FakeDetector


@pytest.fixture
def FakeCapture():  # noqa: N802 - exposes the class under its own name
    return _FakeCapture
