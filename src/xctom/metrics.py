from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).cpu())


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).cpu())


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    m = mse(pred, target)
    if m <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)


@torch.no_grad()
def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Batch-level SSIM approximation using local average pooling.
    Input expected in [0, 1].
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=11, stride=1, padding=5) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )

    return float(ssim_map.mean().cpu())