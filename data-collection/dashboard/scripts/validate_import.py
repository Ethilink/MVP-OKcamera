"""Import-format validator used by ``POST /validate`` (T05, spec §AC8).

The canonical implementation lives at
``data-collection/docs/scripts/validate_import.py``. Rather than copy ~200 lines
(and let the two drift), this module loads that file by path and re-exports its
``validate`` callable, so the dashboard's in-process ``/validate`` and the
standalone CLI validator can never disagree about "import-ready".

``/validate`` imports this lazily, so a resolution problem surfaces as a 500 on
that one endpoint rather than breaking app import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CANONICAL = (
    Path(__file__).resolve().parents[2] / "docs" / "scripts" / "validate_import.py"
)


def _load_canonical():
    spec = importlib.util.spec_from_file_location(
        "_orc_validate_import_canonical", _CANONICAL
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load canonical validator: {_CANONICAL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load_canonical().validate
