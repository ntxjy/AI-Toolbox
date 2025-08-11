"""Dataset utilities for multi-view MOS prediction.

This file provides a :class:`MVSDataset` class that reads six images and
optionally six depth maps per sample.  The dataset expects a metadata
file in which each line contains paths to the images, optional depth
maps and the MOS score.  An example line is::

    l1.png r1.png l2.png r2.png l3.png r3.png dl1.png dr1.png ... dr3.png 75.0

If depth maps are not available the corresponding paths can be replaced
by ``-``.  Additional columns after the MOS value are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["MVSDataset", "load_image"]


@dataclass
class MVSSample:
    images: List[Path]
    depths: List[Optional[Path]]
    mos: float
    distortion_type: Optional[int] = None


def load_image(path: Path) -> torch.Tensor:
    """Read an image file into a ``torch.Tensor`` in the range [0, 1]."""
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


class MVSDataset(Dataset):
    """Dataset for multi-view stereoscopic quality assessment.

    Parameters
    ----------
    meta_file: str or Path
        Path to the text file containing the metadata.
    transform: callable, optional
        Transform applied to each loaded image.
    use_depth: bool, default=False
        Whether depth maps are present in the metadata.
    """

    def __init__(
        self,
        meta_file: Path | str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_depth: bool = False,
    ) -> None:
        self.meta_file = Path(meta_file)
        self.transform = transform
        self.use_depth = use_depth
        self.samples: List[MVSSample] = []

        with self.meta_file.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if self.use_depth:
                    if len(parts) < 13:
                        raise ValueError("Metadata line must contain 13+ columns with depth")
                    img_paths = [Path(p) for p in parts[:6]]
                    depth_paths = [None if p == "-" else Path(p) for p in parts[6:12]]
                    mos = float(parts[12])
                    dist_type = int(parts[13]) if len(parts) > 13 else None
                else:
                    if len(parts) < 7:
                        raise ValueError("Metadata line must contain 7+ columns without depth")
                    img_paths = [Path(p) for p in parts[:6]]
                    depth_paths = [None] * 6
                    mos = float(parts[6])
                    dist_type = int(parts[7]) if len(parts) > 7 else None
                self.samples.append(MVSSample(img_paths, depth_paths, mos, dist_type))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        imgs = []
        deps = []
        for ipath, dpath in zip(sample.images, sample.depths):
            img = load_image(ipath)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
            if dpath is not None:
                depth_img = Image.open(dpath).convert("L")
                depth = torch.from_numpy(np.array(depth_img)).unsqueeze(0).float() / 255.0
            else:
                depth = torch.zeros(1, *img.shape[1:])
            deps.append(depth)
        return {
            "images": imgs,
            "depths": deps if self.use_depth else None,
            "mos": torch.tensor([sample.mos], dtype=torch.float32),
            "distortion_type": sample.distortion_type,
        }
