#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import random

import numpy as np
from PIL import Image


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_to_rgb(tensor):
    array = tensor.permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def tensor_to_mask(tensor):
    array = tensor.squeeze(0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def overlay_mask(image_rgb, mask_gray, alpha=0.5, color=(255, 0, 0)):
    mask = (mask_gray > 0).astype(np.uint8)
    overlay = image_rgb.copy()
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def save_triplet(image_rgb, mask_gray, overlay_rgb, out_path):
    mask_rgb = np.stack([mask_gray] * 3, axis=-1)
    combined = np.concatenate([image_rgb, mask_rgb, overlay_rgb], axis=1)
    Image.fromarray(combined).save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Check image/mask alignment.")
    parser.add_argument("--data-root", default="data/supervisely_persons")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="outputs/alignment_check")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    loader_path = root / "src" / "01_data_loaders" / "supervisely_persons_loader.py"
    loader_module = load_module("supervisely_persons_loader", loader_path)

    splits_json = Path(args.data_root) / "splits.json"
    dataset = loader_module.SuperviselyPersonsDataset(
        splits_json=splits_json,
        split=args.split,
        size=args.size,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(args.num_samples, len(indices))]

    empty_count = 0
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_rgb = tensor_to_rgb(image)
        mask_gray = tensor_to_mask(mask)
        overlay_rgb = overlay_mask(image_rgb, mask_gray)

        coverage = float((mask_gray > 0).mean())
        if coverage < 1e-5:
            empty_count += 1

        out_path = out_dir / f"{args.split}_{i:02d}_cov{coverage:.3f}.png"
        save_triplet(image_rgb, mask_gray, overlay_rgb, out_path)

    print(f"Saved {len(indices)} samples to {out_dir}")
    if empty_count:
        print(f"Warning: {empty_count} masks were empty in the sample set.")


if __name__ == "__main__":
    main()
