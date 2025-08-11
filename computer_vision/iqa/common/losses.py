import torch
import torch.nn.functional as F


def plcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson Linear Correlation Coefficient."""
    pred = pred.view(-1)
    target = target.view(-1)
    vx = pred - pred.mean()
    vy = target - target.mean()
    plcc = (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)
    return 1 - plcc


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def listwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Listwise pairwise ranking loss using logistic formulation."""
    pred = pred.view(-1)
    target = target.view(-1)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    mask = diff_target > 0
    if mask.any():
        loss = F.binary_cross_entropy_with_logits(diff_pred[mask], torch.ones_like(diff_pred[mask]))
    else:
        loss = torch.tensor(0.0, device=pred.device)
    return loss


def l_delta(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on pairwise score differences."""
    pred = pred.view(-1)
    target = target.view(-1)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    return torch.mean(torch.abs(diff_pred - diff_target))


def iqa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined IQA training loss."""
    return plcc_loss(pred, target) + l1_loss(pred, target) + listwise_rank_loss(pred, target) + l_delta(pred, target)
