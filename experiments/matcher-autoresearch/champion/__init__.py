"""The current champion — a clean, standalone copy of the winning method/
variant. Drops into the linker build behind build_gallery/score/accept
(linker-design.md §6). See PARAMS.md for hyperparameters + guarded metrics."""
from .champion import ChampionGallery, ChampionMethod  # noqa: F401
