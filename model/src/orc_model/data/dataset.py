"""Collection of `Clip`s discovered on disk.

`ClipDataset.from_data_dir` walks a directory of clip subdirectories (the
real `model/data/` layout: one subdirectory per clip, each containing
`annotations/annotations.json` and an `images/` subdirectory), skipping any
subdirectory that doesn't look like a complete clip rather than crashing —
`model/data/` is gitignored/machine-local and can vary.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

from orc_model.data.models import Clip


def _default_data_dir() -> Path:
    """`model/data`, resolved relative to this file's location on disk.

    This file lives at `model/src/orc_model/data/dataset.py`:
    parents[0] = model/src/orc_model/data
    parents[1] = model/src/orc_model
    parents[2] = model/src
    parents[3] = model
    """
    return Path(__file__).resolve().parents[3] / "data"


def _is_valid_clip_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "annotations" / "annotations.json").is_file()
        and (path / "images").is_dir()
    )


@dataclass(frozen=True)
class ClipDataset:
    clips: list[Clip]

    @classmethod
    def from_data_dir(cls, data_dir: Path | str | None = None) -> "ClipDataset":
        if data_dir is None:
            data_dir = _default_data_dir()
        data_dir = Path(data_dir)

        clips = []
        for entry in sorted(data_dir.iterdir()):
            if not entry.is_dir():
                continue
            if not _is_valid_clip_dir(entry):
                warnings.warn(
                    f"Skipping incomplete clip directory: {entry}", stacklevel=2
                )
                continue
            clips.append(Clip.from_directory(entry))

        clips.sort(key=lambda clip: clip.name)
        return cls(clips=clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        return iter(self.clips)

    def get_clip(self, name: str) -> Clip:
        for clip in self.clips:
            if clip.name == name:
                return clip
        available = ", ".join(clip.name for clip in self.clips)
        raise KeyError(f"No clip named {name!r}. Available clips: {available}")

    def __getitem__(self, key: str | int) -> Clip:
        if isinstance(key, str):
            return self.get_clip(key)
        return self.clips[key]
