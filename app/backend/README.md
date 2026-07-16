# app/backend

Live runtime for the ORC demo: runs the camera capture-infer loop, holds the
Start/Stop phase state machine, computes the Usage/Completeness report, and
serves it all to `app/frontend` over the frozen HTTP API
([`../docs/api-contract.md`](../docs/api-contract.md)).

Consumes the model only through the `InstrumentTracker` seam from the `orc_model`
package (real tracker, or `ScenarioTracker`/`FakeInstrumentTracker` fakes) — it
never reaches past that seam. Runs headless with `--fake` (ScenarioTracker +
synthetic frames) so the frontend and tests need no camera. Stack (frozen —
DESIGN D1): **FastAPI + uvicorn**, `uv` project. See
[`../docs/DESIGN.md`](../docs/DESIGN.md) and tasks T01–T04.

## MVP settings

All tunable KU Leuven MVP behaviour lives in one operator-facing file at the
app root: [`../mvp.toml`](../mvp.toml). It groups and explains the detector,
Deep OC-SORT, identity matcher, session linker/Unknown re-detection, report
debounce, and camera/capture settings. The backend only loads and validates the
file; it does not own the chosen values. Invalid ranges, misspelled keys, and
incompatible tracker/linker timing fail clearly at startup.

The default is loaded automatically. A second profile can be selected with
`orc-demo --config /path/to/profile.toml ...`.
