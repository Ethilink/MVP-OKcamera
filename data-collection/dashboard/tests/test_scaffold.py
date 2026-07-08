"""T01 scaffold ACs: project boots, orc_model resolves, index served, CLI strict."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import build_parser, create_app


def test_ac01_pytest_runs():
    """AC1: a trivial test executes (uv run pytest has ≥1 passing test)."""
    assert True


def test_ac02_orc_model_imports():
    """AC2: the path dep resolves — orc_model and its Detector import."""
    import orc_model  # noqa: F401
    from orc_model.components.detector import Detector  # noqa: F401


def test_ac03_index_served():
    """AC3: app boots and GET / returns the dashboard HTML.

    (T06 replaced the T01 placeholder with the real capture UI; AC3's intent —
    the app boots and serves the index — is unchanged, so this asserts on stable
    markup from the shipped page instead of the old placeholder string.)
    """
    app = create_app(detector=None, writer_factory=None, capture=None)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<title>ORB data-collection dashboard</title>" in resp.text


def test_ac04_cli_requires_weights_and_version():
    """AC4: starting without --weights / --model-version fails loudly (SystemExit)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--weights", "x.onnx"])  # missing --model-version
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-version", "rfdetr-2026-07-07"])  # missing --weights

    # Both present → parses, and the tag is carried through verbatim.
    args = parser.parse_args(
        ["--weights", "x.onnx", "--model-version", "rfdetr-2026-07-07"]
    )
    assert args.model_version == "rfdetr-2026-07-07"
