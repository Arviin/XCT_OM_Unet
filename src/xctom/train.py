from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xctom.dataset import PairedImageDataset
from xctom.losses import ReconstructionLoss
from xctom.metrics import mae, mse, psnr, simple_ssim
from xctom.model_unet import UNet2D
from xctom.utils import (
    ensure_dir,
    get_device,
    load_yaml,
    save_json,
    save_prediction_figure,
    set_seed,
)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "ssim_loss": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    n_batches = 0

    for batch in tqdm(loader, leave=False):
        x = batch["xct"].to(device=device, dtype=torch.float32)
        y = batch["om"].to(device=device, dtype=torch.float32)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss, parts = criterion(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["l1"] += parts["l1"]
        totals["ssim_loss"] += parts["ssim_loss"]
        totals["mae"] += mae(pred, y)
        totals["mse"] += mse(pred, y)
        totals["psnr"] += psnr(pred, y)
        totals["ssim"] += simple_ssim(pred, y)

        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))

    output_dir = Path(cfg["project"]["output_dir"])
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")
    prediction_dir = ensure_dir(output_dir / "predictions")

    save_json(cfg, log_dir / "config_used.json")

    device = get_device()
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    train_ds = PairedImageDataset(
        data_cfg["train_xct_dir"],
        data_cfg["train_om_dir"],
        image_size=data_cfg["image_size"],
        input_channels=data_cfg["input_channels"],
        output_channels=data_cfg["output_channels"],
        normalize_mode=data_cfg["normalize_mode"],
        augment=True,
    )
    val_ds = PairedImageDataset(
        data_cfg["val_xct_dir"],
        data_cfg["val_om_dir"],
        image_size=data_cfg["image_size"],
        input_channels=data_cfg["input_channels"],
        output_channels=data_cfg["output_channels"],
        normalize_mode=data_cfg["normalize_mode"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model_cfg = cfg["model"]
    model = UNet2D(
        input_channels=int(data_cfg["input_channels"]),
        output_channels=int(data_cfg["output_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        depth=int(model_cfg["depth"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    loss_cfg = cfg["training"]["loss"]
    criterion = ReconstructionLoss(
        l1_weight=float(loss_cfg["l1_weight"]),
        ssim_weight=float(loss_cfg["ssim_weight"]),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    metrics_path = log_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_l1",
                "train_ssim_loss",
                "train_mae",
                "train_mse",
                "train_psnr",
                "train_ssim",
                "val_loss",
                "val_l1",
                "val_ssim_loss",
                "val_mae",
                "val_mse",
                "val_psnr",
                "val_ssim",
                "learning_rate",
            ]
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    n_epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["early_stopping_patience"])
    grad_clip_norm = float(cfg["training"]["grad_clip_norm"])

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"val_ssim={val_metrics['ssim']:.4f} | "
            f"val_psnr={val_metrics['psnr']:.2f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["l1"],
                    train_metrics["ssim_loss"],
                    train_metrics["mae"],
                    train_metrics["mse"],
                    train_metrics["psnr"],
                    train_metrics["ssim"],
                    val_metrics["loss"],
                    val_metrics["l1"],
                    val_metrics["ssim_loss"],
                    val_metrics["mae"],
                    val_metrics["mse"],
                    val_metrics["psnr"],
                    val_metrics["ssim"],
                    current_lr,
                ]
            )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_val_loss,
            cfg,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                cfg,
            )

            print(f"Saved new best checkpoint with val_loss={best_val_loss:.5f}")

            if cfg["validation"]["save_prediction_examples"]:
                model.eval()
                batch = next(iter(val_loader))
                x = batch["xct"].to(device=device, dtype=torch.float32)
                y = batch["om"].to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    pred = model(x)

                n = min(int(cfg["validation"]["num_prediction_examples"]), x.shape[0])
                for i in range(n):
                    save_prediction_figure(
                        xct=x[i],
                        target=y[i],
                        pred=pred[i],
                        path=prediction_dir / f"epoch_{epoch:04d}_example_{i:02d}.png",
                    )
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()