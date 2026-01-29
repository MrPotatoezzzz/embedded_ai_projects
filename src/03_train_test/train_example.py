#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import random
from contextlib import nullcontext

import torch
from torch import nn
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device, metric_fn):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_scores = metric_fn(outputs, targets)
            scores.append(batch_scores.detach().cpu())
    if not scores:
        return 0.0
    return torch.cat(scores).mean().item()


def plot_curve(values, title, path):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_example(
    data_root,
    output_dir,
    epochs=50,
    batch_size=4,
    lr=1e-3,
    size=256,
    num_workers=0,
    seed=7,
    min_iou=0.9,
    stop_on="eval",
    base=64,
    amp=False,
    weight_decay=0.0,
    early_stop_patience=20,
    early_stop_min_delta=1e-4,
):
    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "supervisely_persons_loader.py"
    model_path = root / "src" / "02_model" / "model_example.py"
    metrics_path = root / "src" / "03_train_test" / "metrics_example.py"

    loader_module = load_module("data_loader_example", loader_path)
    model_module = load_module("model_example", model_path)
    metrics_module = load_module("metrics_example", metrics_path)

    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    splits_json = Path(data_root) / "splits.json"

    train_loader = loader_module.create_dataloader(
        splits_json=splits_json,
        split="train",
        batch_size=batch_size,
        size=size,
        shuffle=True,
        num_workers=num_workers,
    )
    train_eval_loader = loader_module.create_dataloader(
        splits_json=splits_json,
        split="train",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
    )
    eval_loader = loader_module.create_dataloader(
        splits_json=splits_json,
        split="val",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_module.UNet3(out_channels=1, base=base).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    dice_weight = 0.5

    use_amp = amp and device.type in ("cuda", "mps")
    if amp and device.type == "cpu":
        print("AMP requested but running on CPU; AMP will be disabled.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    train_losses = []
    train_iou = []
    eval_iou = []
    best_iou = -1.0
    best_epoch = 0
    best_path = output_dir / "model_best.pt"
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        train_iter = train_loader
        if tqdm is not None:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for inputs, targets in train_iter:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                outputs = model(inputs)
                bce_loss = loss_fn(outputs, targets)
                dice_loss = metrics_module.dice_loss_from_logits(outputs, targets).mean()
                loss = bce_loss + dice_weight * dice_loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

            if tqdm is not None:
                train_iter.set_postfix(loss=loss.item())

        avg_loss = running_loss / max(sample_count, 1)
        train_losses.append(avg_loss)

        train_score = evaluate(
            model, train_eval_loader, device, lambda p, t: metrics_module.iou(torch.sigmoid(p), t)
        )
        train_iou.append(train_score)

        iou_score = evaluate(
            model, eval_loader, device, lambda p, t: metrics_module.iou(torch.sigmoid(p), t)
        )
        eval_iou.append(iou_score)

        if iou_score > best_iou:
            best_iou = iou_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            no_improve_epochs = 0
        else:
            if iou_score < best_iou + early_stop_min_delta:
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"train IoU: {train_score:.4f} - eval IoU: {iou_score:.4f}"
        )

        stop_score = iou_score if stop_on == "eval" else train_score
        stop_label = "eval" if stop_on == "eval" else "train"
        if stop_score >= min_iou:
            print(f"Reached {stop_label} IoU >= {min_iou:.2f}. Stopping early.")
            break
        if no_improve_epochs >= early_stop_patience:
            print(
                f"No eval IoU improvement >= {early_stop_min_delta:.1e} for "
                f"{early_stop_patience} epochs. Stopping early."
            )
            break

    torch.save(model.state_dict(), output_dir / "model_last.pt")

    plot_curve(train_losses, "Train Loss", output_dir / "loss_curve.png")
    plot_curve(eval_iou, "Eval IoU", output_dir / "iou_curve.png")

    metrics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": size,
        "train_loss": train_losses,
        "train_iou": train_iou,
        "eval_iou": eval_iou,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "min_iou": min_iou,
        "stop_on": stop_on,
        "base": base,
        "amp": amp,
        "weight_decay": weight_decay,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
    }

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "output_dir": str(output_dir),
        "best_path": str(best_path),
        "best_iou": best_iou,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the example segmentation model.")
    parser.add_argument("--data-root", default="data/supervisely_persons")
    parser.add_argument("--output-dir", default="outputs/supervisely_run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--base",
        type=int,
        default=64,
        help="Base channel width for UNet.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (faster on GPU/MPS).",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.9,
        help="Stop training once IoU meets this value.",
    )
    parser.add_argument(
        "--stop-on",
        choices=("train", "eval"),
        default="eval",
        help="Which IoU to use for early stopping.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop if eval IoU doesn't improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum eval IoU improvement to reset early stopping.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_example(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        size=args.size,
        num_workers=args.num_workers,
        seed=args.seed,
        min_iou=args.min_iou,
        stop_on=args.stop_on,
        base=args.base,
        amp=args.amp,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )


if __name__ == "__main__":
    main()
    use_amp = amp and device.type == "mps"
    if amp and device.type == "cpu":
        print("AMP requested but running on CPU; AMP will be disabled.")
    if amp and device.type == "cuda":
        use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
