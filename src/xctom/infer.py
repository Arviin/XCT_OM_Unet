from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xctom.dataset import PairedImageDataset
from xctom.model_unet import UNet2D
from xctom.utils import ensure_dir, get_device, load_yaml, save_prediction_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = get_device()

    data_cfg = cfg["data"]

    test_ds = PairedImageDataset(
        data_cfg["test_xct_dir"],
        data_cfg["test_om_dir"],
        image_size=data_cfg["image_size"],
        input_channels=data_cfg["input_channels"],
        output_channels=data_cfg["output_channels"],
        normalize_mode=data_cfg["normalize_mode"],
        augment=False,
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model_cfg = cfg["model"]
    model = UNet2D(
        input_channels=int(data_cfg["input_channels"]),
        output_channels=int(data_cfg["output_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    out_dir = ensure_dir(Path(cfg["project"]["output_dir"]) / "test_predictions")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x = batch["xct"].to(device=device, dtype=torch.float32)
            y = batch["om"].to(device=device, dtype=torch.float32)

            pred = model(x)

            save_prediction_figure(
                xct=x[0],
                target=y[0],
                pred=pred[0],
                path=out_dir / f"test_{i:04d}.png",
            )

    print(f"Saved predictions to: {out_dir}")


if __name__ == "__main__":
    main()