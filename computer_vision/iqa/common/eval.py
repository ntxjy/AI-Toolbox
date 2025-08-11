import torch


def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(-1).double()
    y = y.view(-1).double()
    vx = x - x.mean()
    vy = y - y.mean()
    return (vx * vy).sum() / (torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8)


def logistic4(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    b1, b2, b3, b4 = beta
    return b1 + (b2 - b1) / (1.0 + torch.exp(-b3 * (x - b4)))


def logistic5(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    b1, b2, b3, b4, b5 = beta
    return b1 + (b2 - b1) / (1.0 + torch.exp(-b3 * (x - b4))) ** b5


def fit_logistic(x: torch.Tensor, y: torch.Tensor, kind: str = "5", max_iter: int = 500) -> torch.Tensor:
    x = x.view(-1).double()
    y = y.view(-1).double()
    if kind == "4":
        beta = torch.nn.Parameter(torch.tensor([y.min(), y.max(), 1.0, x.mean()], dtype=torch.double))
        func = logistic4
    else:
        beta = torch.nn.Parameter(torch.tensor([y.min(), y.max(), 1.0, x.mean(), 1.0], dtype=torch.double))
        func = logistic5

    optimizer = torch.optim.LBFGS([beta], max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        pred = func(x, beta)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    return beta.detach()


def logistic_fit_metrics(pred: torch.Tensor, target: torch.Tensor, kind: str = "5"):
    """Fit logistic curve then compute PLCC, RMSE and GoF."""
    pred = pred.view(-1).double()
    target = target.view(-1).double()
    beta = fit_logistic(pred, target, kind)
    func = logistic4 if kind == "4" else logistic5
    mapped = func(pred, beta)
    plcc = pearsonr(mapped, target)
    rmse = torch.sqrt(torch.mean((mapped - target) ** 2))
    sse = torch.sum((mapped - target) ** 2)
    sst = torch.sum((target - target.mean()) ** 2)
    gof = 1 - sse / sst
    return float(plcc), float(rmse), float(gof)
