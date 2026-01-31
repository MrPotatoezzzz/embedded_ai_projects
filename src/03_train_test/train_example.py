#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import random

import torch
from torch import nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
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


def _thresholded_iou(metrics_module, logits, targets, thr=0.5):
    # Ensure IoU is computed on binary predictions (much more reliable)
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    return metrics_module.iou(preds, targets)


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
    lr=3e-4,
    size=256,
    num_workers=0,
    seed=0,
    # new knobs
    base=64,
    dropout=0.2,
    weight_decay=1e-4,
    patience=8,
    iou_threshold=0.5,
    pos_weight=None,
    dice_weight=0.5,
):
    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "supervisely_persons_loader.py"
    model_path = root / "src" / "02_model" / "model_example.py"
    metrics_path = root / "src" / "03_train_test" / "metrics_example.py"

    loader_module = load_module("data_loader_example", loader_path)
    model_module = load_module("model_example", model_path)
    metrics_module = load_module("metrics_example", metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = model_module.UNet3(out_channels=1, base=base, dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if pos_weight is not None:
        pw = torch.tensor([float(pos_weight)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_losses = []
    train_iou = []
    eval_iou = []

    best_iou = -1.0
    best_epoch = 0
    best_path = output_dir / "model_best.pt"

    bad_epochs = 0

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

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)

            bce_loss = loss_fn(outputs, targets)
            dice_loss = metrics_module.dice_loss_from_logits(outputs, targets).mean()
            loss = bce_loss + dice_weight * dice_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

            if tqdm is not None:
                train_iter.set_postfix(loss=float(loss.item()))

        avg_loss = running_loss / max(sample_count, 1)
        train_losses.append(avg_loss)

        # IoU on binarized predictions
        train_score = evaluate(
            model,
            train_eval_loader,
            device,
            lambda p, t: _thresholded_iou(metrics_module, p, t, thr=iou_threshold),
        )
        train_iou.append(train_score)

        val_score = evaluate(
            model,
            eval_loader,
            device,
            lambda p, t: _thresholded_iou(metrics_module, p, t, thr=iou_threshold),
        )
        eval_iou.append(val_score)

        # checkpoint + early stopping on VAL
        if val_score > best_iou + 1e-4:
            best_iou = val_score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"train IoU: {train_score:.4f} - val IoU: {val_score:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping: val IoU didn't improve for {patience} epochs.")
            break

    torch.save(model.state_dict(), output_dir / "model_last.pt")

    plot_curve(train_losses, "Train Loss", output_dir / "loss_curve.png")
    plot_curve(eval_iou, "Val IoU", output_dir / "iou_curve.png")

    metrics = {
        "epochs_requested": epochs,
        "epochs_ran": len(train_losses),
        "batch_size": batch_size,
        "lr_init": lr,
        "weight_decay": weight_decay,
        "image_size": size,
        "base": base,
        "dropout": dropout,
        "patience": patience,
        "iou_threshold": iou_threshold,
        "pos_weight": pos_weight,
        "dice_weight": dice_weight,
        "train_loss": train_losses,
        "train_iou": train_iou,
        "val_iou": eval_iou,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
    }

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "output_dir": str(output_dir),
        "best_path": str(best_path),
        "best_iou": best_iou,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train example segmentation model.")
    parser.add_argument("--data-root", default="data/supervisely_persons")
    parser.add_argument("--output-dir", default="outputs/supervisely_run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # new args (safe defaults)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Weight for Dice loss term (BCE + dice_weight * DiceLoss).",
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
        base=args.base,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        iou_threshold=args.iou_threshold,
        pos_weight=args.pos_weight,
        dice_weight=args.dice_weight,
    )


if __name__ == "__main__":
    main()
