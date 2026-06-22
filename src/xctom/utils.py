from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility. May reduce speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert [C,H,W] tensor in [0,1] to displayable numpy image.
    """
    x = x.detach().cpu().clamp(0, 1)

    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")

    if x.shape[0] == 1:
        return x[0].numpy()

    return x.permute(1, 2, 0).numpy()


def save_prediction_figure(
    xct: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    path: str | Path,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    xct_np = tensor_to_numpy_image(xct)
    target_np = tensor_to_numpy_image(target)
    pred_np = tensor_to_numpy_image(pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(xct_np, cmap="gray")
    axes[0].set_title("Input XCT")

    axes[1].imshow(target_np, cmap="gray")
    axes[1].set_title("Real OM")

    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Predicted OM")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)