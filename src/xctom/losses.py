from __future__ import annotations

import torch
import torch.nn as nn


class SSIMLoss(nn.Module):
    """
    Lightweight differentiable SSIM loss.

    Returns:
        1 - SSIM

    This is used as an auxiliary structure-aware loss.
    L1 remains the primary loss.
    """

    def __init__(self, window_size: int = 11, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2) -> None:
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
        self.avg_pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_x = self.avg_pool(pred)
        mu_y = self.avg_pool(target)

        sigma_x = self.avg_pool(pred * pred) - mu_x * mu_x
        sigma_y = self.avg_pool(target * target) - mu_y * mu_y
        sigma_xy = self.avg_pool(pred * target) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.c1) * (sigma_x + sigma_y + self.c2)
        )

        return 1.0 - ssim_map.mean()


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for XCT -> OM baseline.

    L1:
        robust pixel-wise fidelity.

    SSIM:
        structural similarity term.

    No adversarial term here.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.2) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1_value = self.l1(pred, target)
        ssim_value = self.ssim(pred, target)

        total = self.l1_weight * l1_value + self.ssim_weight * ssim_value

        parts = {
            "l1": float(l1_value.detach().cpu()),
            "ssim_loss": float(ssim_value.detach().cpu()),
            "total": float(total.detach().cpu()),
        }

        return total, parts