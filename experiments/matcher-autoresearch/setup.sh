#!/usr/bin/env bash
# Re-runnable scaffold for the contained matcher-autoresearch sandbox.
# Run from the repo root:  bash experiments/matcher-autoresearch/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="$ROOT/experiments/matcher-autoresearch"
cd "$ROOT"

echo "→ sandbox: $DIR"
mkdir -p "$DIR/harness" "$DIR/method" "$DIR/runs" "$DIR/champion"

# frozen-harness base = the stage-1 research code (loader/eval/fusion/openset)
cp docs/wayfinder/session-linker/assets/T02-stage1/*.py "$DIR/harness/"
# the research brief travels with the sandbox (containment)
cp docs/wayfinder/session-linker/assets/T02-stage2/program.md "$DIR/program.md"

# sanity: the read-only inputs the run depends on
echo "→ checking read-only inputs:"
for p in model/data/instruments model/data/other_objects \
         matching/data/testing model/weights; do
  if [ -e "$ROOT/$p" ]; then echo "   ok   $p"; else echo "   MISSING  $p"; fi
done

echo "→ python deps (in the model venv or a fresh uv venv):"
echo "   uv pip install torch transformers pillow scikit-learn pycocotools psutil scipy"

echo "✓ scaffold ready. Next: open a fresh chat and follow README.md → 'How to start it'."
