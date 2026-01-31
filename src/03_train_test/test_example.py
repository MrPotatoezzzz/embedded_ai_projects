#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _thresholded_iou(metrics_module, logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    return metrics_module.iou(preds, targets)


def evaluate_model(data_root, weights_path, batch_size=4, size=256, num_workers=0, base=32, dropout=0.2, iou_threshold=0.5):
    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "supervisely_persons_loader.py"
    model_path = root / "src" / "02_model" / "model_example.py"
    metrics_path = root / "src" / "03_train_test" / "metrics_example.py"

    loader_module = load_module("data_loader_example", loader_path)
    model_module = load_module("model_example", model_path)
    metrics_module = load_module("metrics_example", metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_json = Path(data_root) / "splits.json"

    test_loader = loader_module.create_dataloader(
        splits_json=splits_json,
        split="test",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_module.UNet3(out_channels=1, base=base, dropout=dropout).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scores = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_scores = _thresholded_iou(metrics_module, outputs, targets, thr=iou_threshold)
            scores.append(batch_scores.detach().cpu())

    mean_iou = torch.cat(scores).mean().item() if scores else 0.0

    output_dir = Path(weights_path).parent
    out_path = output_dir / "test_metrics.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"test_iou": mean_iou, "iou_threshold": iou_threshold, "base": base, "dropout": dropout}, handle, indent=2)

    print(f"Test IoU: {mean_iou:.4f}")
    return mean_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Test the example segmentation model.")
    parser.add_argument("--data-root", default="data/supervisely_persons")
    parser.add_argument("--weights", default="outputs/supervisely_run/model_best.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)

    # keep test aligned with training config
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        data_root=args.data_root,
        weights_path=args.weights,
        batch_size=args.batch_size,
        size=args.size,
        num_workers=args.num_workers,
        base=args.base,
        dropout=args.dropout,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()