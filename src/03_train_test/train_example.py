#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
import random

import torch
from torch import nn


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


def evaluate(model, dataloader, device, ssim_fn):
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_scores = ssim_fn(outputs, targets)
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
    size=64,
    num_workers=0,
    seed=7,
    min_ssim=0.9,
    edge_weight=4.0,
    l1_weight=0.2,
):
    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "data_loader_example.py"
    model_path = root / "src" / "02_model" / "model_example.py"
    metrics_path = root / "src" / "03_train_test" / "metrics_example.py"

    loader_module = load_module("data_loader_example", loader_path)
    model_module = load_module("model_example", model_path)
    metrics_module = load_module("metrics_example", metrics_path)

    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = loader_module.create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        size=size,
        shuffle=True,
        num_workers=num_workers,
    )
    train_eval_loader = loader_module.create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
        augment=False,
    )
    eval_loader = loader_module.create_dataloader(
        data_root=data_root,
        split="eval",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_module.SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction="none")

    train_losses = []
    train_ssim = []
    eval_ssim = []
    best_ssim = -1.0
    best_epoch = 0
    best_path = output_dir / "model_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            pixel_loss = loss_fn(outputs, targets)
            weights = 1.0 + edge_weight * targets
            l1_loss = (pixel_loss * weights).mean()
            ssim_loss = (1.0 - metrics_module.ssim(outputs, targets)).mean()
            loss = l1_loss * l1_weight + ssim_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

        avg_loss = running_loss / max(sample_count, 1)
        train_losses.append(avg_loss)

        train_score = evaluate(model, train_eval_loader, device, metrics_module.ssim)
        train_ssim.append(train_score)

        ssim_score = evaluate(model, eval_loader, device, metrics_module.ssim)
        eval_ssim.append(ssim_score)

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch}/{epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"train SSIM: {train_score:.4f} - eval SSIM: {ssim_score:.4f}"
        )

        if train_score >= min_ssim:
            print(f"Reached train SSIM >= {min_ssim:.2f}. Stopping early.")
            break

    torch.save(model.state_dict(), output_dir / "model_last.pt")

    plot_curve(train_losses, "Train Loss", output_dir / "loss_curve.png")
    plot_curve(eval_ssim, "Eval SSIM", output_dir / "ssim_curve.png")

    metrics = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": size,
        "train_loss": train_losses,
        "train_ssim": train_ssim,
        "eval_ssim": eval_ssim,
        "best_ssim": best_ssim,
        "best_epoch": best_epoch,
        "min_ssim": min_ssim,
        "edge_weight": edge_weight,
        "l1_weight": l1_weight,
    }

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "output_dir": str(output_dir),
        "best_path": str(best_path),
        "best_ssim": best_ssim,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the example image-to-image model.")
    parser.add_argument("--data-root", default="data/dataset_example")
    parser.add_argument("--output-dir", default="outputs/example_run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=4.0,
        help="Emphasize edge pixels in the training loss.",
    )
    parser.add_argument(
        "--l1-weight",
        type=float,
        default=0.2,
        help="Scale the L1 component of the SSIM + L1 training loss.",
    )
    parser.add_argument(
        "--min-ssim",
        type=float,
        default=0.9,
        help="Stop training once train SSIM meets this value.",
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
        min_ssim=args.min_ssim,
        edge_weight=args.edge_weight,
        l1_weight=args.l1_weight,
    )


if __name__ == "__main__":
    main()
