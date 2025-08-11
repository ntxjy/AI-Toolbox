"""Generic training and validation template for IQA models."""
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from common.group_sampler import GroupSampler
from common.losses import iqa_loss
from common.eval import logistic_fit_metrics


class DummyDataset(Dataset):
    """A minimal dataset example.

    Args:
        data: image tensors.
        labels: quality scores.
        scenes: scene identifiers for each sample.
        params: parameter identifiers for each sample.
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor, scenes, params):
        self.data = data
        self.labels = labels
        self.scenes = scenes
        self.params = params

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.data[idx], self.labels[idx]


def train_template(model: torch.nn.Module, train_ds: Dataset, val_ds: Dataset,
                   epochs: int = 1, batch_size: int = 8, lr: float = 1e-4):
    """Train ``model`` using provided datasets and evaluate each epoch."""
    sampler = GroupSampler(train_ds.scenes, train_ds.params, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            preds = model(imgs).squeeze()
            loss = iqa_loss(preds, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs).squeeze()
                preds_all.append(preds)
                labels_all.append(labels.float())
        preds_cat = torch.cat(preds_all)
        labels_cat = torch.cat(labels_all)
        plcc, rmse, gof = logistic_fit_metrics(preds_cat, labels_cat)
        print(f"Epoch {epoch + 1}: PLCC={plcc:.4f} RMSE={rmse:.4f} GoF={gof:.4f}")


__all__ = [
    "DummyDataset",
    "train_template",
]
