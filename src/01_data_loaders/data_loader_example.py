#!/usr/bin/env python3
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is optional for the PPM-only example.
    Image = None


DEFAULT_IMAGE_SIZE = 64


def _read_ppm(path):
    with open(path, "rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Unsupported PPM format: {magic!r}")

        tokens = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())

        if len(tokens) < 3:
            raise ValueError(f"Invalid PPM header in {path}")

        width = int(tokens[0])
        height = int(tokens[1])
        maxval = int(tokens[2])
        if maxval != 255:
            raise ValueError(f"Unsupported maxval {maxval} in {path}")

        pixel_count = width * height * 3
        pixels = list(handle.read(pixel_count))
        if len(pixels) != pixel_count:
            raise ValueError(f"Unexpected pixel data length in {path}")

    tensor = torch.tensor(pixels, dtype=torch.float32)
    tensor = tensor.view(height, width, 3).permute(2, 0, 1) / 255.0
    return tensor


def _read_with_pil(path):
    image = Image.open(path).convert("RGB")
    data = torch.ByteTensor(list(image.tobytes()))
    tensor = data.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float() / 255.0


def _load_image(path, size):
    suffix = Path(path).suffix.lower()
    if Image is not None and suffix in {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}:
        tensor = _read_with_pil(path)
    elif suffix == ".ppm":
        tensor = _read_ppm(path)
    else:
        raise ValueError(
            f"Unsupported image format {suffix}. Install Pillow or use PPM files."
        )

    if size is not None and (tensor.shape[1] != size or tensor.shape[2] != size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


class ImageToImageDataset(Dataset):
    def __init__(self, data_root, split, size=DEFAULT_IMAGE_SIZE, augment=False):
        self.data_root = Path(data_root)
        self.split = split
        self.size = size
        self.augment = augment

        input_dir = self.data_root / split / "input"
        target_dir = self.data_root / split / "target"
        input_paths = sorted(input_dir.glob("*"))
        target_paths = {path.name: path for path in target_dir.glob("*")}

        self.pairs = []
        for input_path in input_paths:
            target_path = target_paths.get(input_path.name)
            if target_path:
                self.pairs.append((input_path, target_path))

        if not self.pairs:
            raise FileNotFoundError(
                f"No image pairs found in {input_dir} and {target_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        input_tensor = _load_image(input_path, self.size)
        target_tensor = _load_image(target_path, self.size)

        if self.augment and random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])

        return input_tensor, target_tensor


def create_dataloader(
    data_root,
    split,
    batch_size=4,
    size=DEFAULT_IMAGE_SIZE,
    shuffle=True,
    num_workers=0,
    augment=None,
):
    if augment is None:
        augment = split == "train"

    dataset = ImageToImageDataset(
        data_root=data_root,
        split=split,
        size=size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
