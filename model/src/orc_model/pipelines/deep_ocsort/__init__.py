"""Deep-OC-SORT, vendored and adapted from
https://github.com/GerardMaggiolino/Deep-OC-SORT (MIT license).

The upstream repo is a MOT-benchmark research codebase, not a library: its
`OCSort` hard-constructs a CUDA-only, MOT-checkpoint-only embedder and a
file-cache-backed camera-motion-compensator from `argparse` args, and expects
detections already in a YOLOX tensor's coordinate space. None of that applies
to our cached-detection playground, so this package keeps `ocsort.py`'s
tracking algorithm (Kalman filter + OCM/OCR + embedding-fused association)
close to the original, but:
  - `embedder.py` / `cmc.py` are our own lightweight, CPU/MPS-friendly
    replacements (a generic torchvision backbone; on-the-fly sparse-flow CMC)
    instead of vendoring the original FastReID/torchreid/file-cache versions.
  - `OCSort.__init__` takes those two in directly rather than building them
    from `args`.
  - `OCSort.update()` drops the tensor-space rescale and MOT `tag` string,
    taking detections already in the frame's own pixel coordinates.
  - `kalmanfilter.py` keeps only the predict/update/affine-correction path
    actually exercised here, dropping the filterpy-dependent bits (steady
    state, RTS smoothing, log-likelihood) `ocsort.py` never calls.

See `DeepOCSortTracker` for the `sv.Detections`-in/-out wrapper used by the
playground notebook.
"""

from .tracker import DeepOCSortTracker

__all__ = ["DeepOCSortTracker"]
