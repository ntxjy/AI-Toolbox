"""Minimal training script for the multi-view MOS prediction model.

This script is intentionally lightweight and serves as an example of how
all pieces fit together.  It does not implement distributed training or
advanced logging but mirrors the structure described in the design
specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import MVSDataset
from .model import DFEModule, MultiViewIQAModel
from .losses import (
    smooth_l1,
    pairwise_rank_loss,
    zscore_mse,
    anchor_consistency_loss,
    anchor_relative_loss,
)


@dataclass
class Config:
    meta_file: Path
    use_depth: bool = False
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 1


def build_model(cfg: Config) -> MultiViewIQAModel:
    dfe = DFEModule(use_depth=cfg.use_depth)
    model = MultiViewIQAModel(dfe)
    return model


def train(cfg: Config) -> None:
    dataset = MVSDataset(cfg.meta_file, use_depth=cfg.use_depth)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = build_model(cfg)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    model.train()
    for epoch in range(cfg.epochs):
        for batch in loader:
            images = batch["images"]
            depths = batch.get("depths")
            target = batch["mos"]
            pred = model(images, depths)
            l1 = smooth_l1(pred, target)
            rank = pairwise_rank_loss(pred.view(-1), target.view(-1))
            zmse = zscore_mse(pred, target)
            ac = anchor_consistency_loss(pred, target, max_val=1.0, min_val=0.0)
            ar = anchor_relative_loss(pred, target, max_val=1.0, min_val=0.0)
            loss = l1 + 0.5 * rank + zmse + 0.2 * (ac + ar)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":  # pragma: no cover - script entry
    import argparse

    parser = argparse.ArgumentParser(description="Train multi-view MOS model")
    parser.add_argument("meta_file", type=Path, help="metadata file")
    parser.add_argument("--use-depth", action="store_true")
    args = parser.parse_args()
    cfg = Config(meta_file=args.meta_file, use_depth=args.use_depth)
    train(cfg)
