"""T04: FastAPI app assembly, `--fake`/`--camera` CLI, and the `orc-demo`
console-script entrypoint (see pyproject.toml `[project.scripts]`).

Stub for T01 so `uv run orc-demo` resolves — filled in T04. Do not import
`load_tracker` at module top level here; T04 imports it lazily inside the
real-camera branch only (see DESIGN.md and T01 task notes).
"""


def main() -> None:
    raise NotImplementedError("orc-demo CLI is implemented in T04")


if __name__ == "__main__":
    main()
