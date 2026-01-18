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


def evaluate_model(data_root, weights_path, batch_size=4, size=64, num_workers=0):
    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "data_loader_example.py"
    model_path = root / "src" / "02_model" / "model_example.py"
    metrics_path = root / "src" / "03_train_test" / "metrics_example.py"

    loader_module = load_module("data_loader_example", loader_path)
    model_module = load_module("model_example", model_path)
    metrics_module = load_module("metrics_example", metrics_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = loader_module.create_dataloader(
        data_root=data_root,
        split="test",
        batch_size=batch_size,
        size=size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = model_module.SmallUNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    scores = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_scores = metrics_module.ssim(outputs, targets)
            scores.append(batch_scores.detach().cpu())

    if not scores:
        mean_ssim = 0.0
    else:
        mean_ssim = torch.cat(scores).mean().item()

    output_dir = Path(weights_path).parent
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump({"test_ssim": mean_ssim}, handle, indent=2)

    print(f"Test SSIM: {mean_ssim:.4f}")
    return mean_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Test the example image-to-image model.")
    parser.add_argument("--data-root", default="data/dataset_example")
    parser.add_argument("--weights", default="outputs/example_run/model_best.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        data_root=args.data_root,
        weights_path=args.weights,
        batch_size=args.batch_size,
        size=args.size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
