# app/backend

Live runtime for the ORC demo: ingests the camera stream, loads `model`'s artifact, holds Start/Stop phase state, and computes the Usage/Completeness report for the dashboard. Talks to `app/frontend` over its own API; never imports from `model/` directly except the artifact.

Not yet scaffolded.
