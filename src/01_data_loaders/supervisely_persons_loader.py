#!/usr/bin/env python3
import argparse
import base64
import io
import json
import random
import shutil
import warnings
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import dataset_tools as dtools
except ImportError:
    dtools = None

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
JSON_EXTS = {".json"}
IMAGE_DIR_NAMES = {"images", "image", "imgs", "img"}
MASK_DIR_NAMES = {"masks", "mask", "ann", "annotations", "labels", "gt"}


# ----------------------------
# Download + dataset discovery
# ----------------------------
def download_supervisely_persons(dst_dir):
    if dtools is None:
        raise ImportError("dataset_tools is not installed; pip install dataset-tools.")
    dst_path = Path(dst_dir).expanduser()
    dst_path.mkdir(parents=True, exist_ok=True)
    dtools.download(dataset="Supervisely Persons", dst_dir=str(dst_path))
    return _resolve_dataset_root(dst_path)


def _resolve_dataset_root(dst_path):
    candidates = [p for p in dst_path.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        name = candidate.name.lower()
        if "supervisely" in name and "person" in name:
            return candidate
    raise FileNotFoundError(
        "Could not resolve dataset root; pass --dataset-root explicitly."
    )


def _find_best_dirs(dataset_root):
    image_dirs = [
        p
        for p in dataset_root.rglob("*")
        if p.is_dir() and p.name.lower() in IMAGE_DIR_NAMES
    ]
    mask_dirs = [
        p
        for p in dataset_root.rglob("*")
        if p.is_dir() and p.name.lower() in MASK_DIR_NAMES
    ]

    best = (0, None, None)
    for image_dir in image_dirs:
        for mask_dir in mask_dirs:
            if image_dir.parent != mask_dir.parent:
                continue
            matches = _count_matches(image_dir, mask_dir)
            if matches > best[0]:
                best = (matches, image_dir, mask_dir)

    if best[1] is None or best[0] == 0:
        raise FileNotFoundError(
            "Could not find matching image/mask folders. "
            "Pass --image-dir and --mask-dir explicitly."
        )
    return best[1], best[2]


def _count_matches(image_dir, mask_dir):
    mask_map = _build_mask_maps(mask_dir)
    return sum(
        1
        for img in image_dir.glob("*")
        if img.suffix.lower() in IMAGE_EXTS and _match_mask(img, mask_map) is not None
    )


# ----------------------------
# IO: image + mask
# ----------------------------
def _load_image(path):
    if Image is None:
        raise ImportError("Pillow is required to load images.")
    image = Image.open(path).convert("RGB")
    data = torch.ByteTensor(list(image.tobytes()))
    tensor = data.view(image.height, image.width, 3).permute(2, 0, 1)
    return tensor.float() / 255.0


def _load_mask(path):
    if Image is None:
        raise ImportError("Pillow is required to load masks.")
    path = Path(path)
    if path.suffix.lower() in JSON_EXTS:
        mask = _mask_from_supervisely_json(path)
    else:
        mask = Image.open(path).convert("L")
    data = torch.ByteTensor(list(mask.tobytes()))
    tensor = data.view(mask.height, mask.width).unsqueeze(0)
    return (tensor.float() / 255.0).clamp(0, 1)


def _mask_from_supervisely_json(path):
    payload = json.loads(Path(path).read_text())
    size = payload.get("size", {})
    width = int(size.get("width", 0))
    height = int(size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size in {path}")

    mask = Image.new("L", (width, height), 0)
    objects = payload.get("objects", [])

    for obj in objects:
        geometry_type = obj.get("geometryType")
        if geometry_type == "bitmap":
            bitmap = obj.get("bitmap", {})
            data = bitmap.get("data")
            origin = bitmap.get("origin", [0, 0])
            if not data:
                continue
            submask = _decode_supervisely_bitmap(data)
            if submask.mode == "RGBA":
                submask = submask.split()[-1]
            else:
                submask = submask.convert("L")

            x, y = origin
            # handle swapped origin edge-case
            if (x > width or y > height) and (origin[0] <= height and origin[1] <= width):
                x, y = origin[1], origin[0]

            x = max(0, min(int(x), width - 1))
            y = max(0, min(int(y), height - 1))
            mask.paste(submask, (x, y), submask)

        elif geometry_type == "polygon":
            points = obj.get("points", {}).get("exterior")
            if points:
                draw = ImageDraw.Draw(mask)
                draw.polygon([tuple(p) for p in points], fill=255)
        else:
            warnings.warn(f"Unsupported geometryType {geometry_type} in {path}")

    return mask


def _decode_supervisely_bitmap(data):
    raw = base64.b64decode(data)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    return Image.open(io.BytesIO(raw))


# ----------------------------
# Splits
# ----------------------------
def create_splits(
    dataset_root,
    out_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=1337,
    image_dir=None,
    mask_dir=None,
):
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if image_dir and mask_dir:
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
    else:
        image_dir, mask_dir = _find_best_dirs(dataset_root)

    pairs = []
    mask_map = _build_mask_maps(mask_dir)
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        mask_path = _match_mask(image_path, mask_map)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found in {image_dir} and {mask_dir}."
        )

    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    payload = {"dataset_root": str(dataset_root), "splits": {}}
    for split_name, split_pairs in splits.items():
        payload["splits"][split_name] = [
            {"image": _rel_path(dataset_root, img_path), "mask": _rel_path(dataset_root, mask_path)}
            for img_path, mask_path in split_pairs
        ]

    splits_path = out_dir / "splits.json"
    splits_path.write_text(json.dumps(payload, indent=2))
    return splits_path


def _rel_path(root, path):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ----------------------------
# Augmentations (image+mask)
# ----------------------------
def _rand_bool(p, g):
    return torch.rand((), generator=g).item() < p


def _color_jitter(image, g, brightness=0.10, contrast=0.10):
    # image: [C,H,W] in [0,1]
    if brightness > 0:
        b = (1.0 + (torch.rand((), generator=g).item() * 2 - 1) * brightness)
        image = image * b
    if contrast > 0:
        c = (1.0 + (torch.rand((), generator=g).item() * 2 - 1) * contrast)
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - mean) * c + mean
    return image.clamp(0, 1)


def _add_noise(image, g, sigma=0.02):
    if sigma <= 0:
        return image
    noise = torch.randn(image.shape, generator=g, device=image.device) * sigma
    return (image + noise).clamp(0, 1)


def _resize_tensor(tensor, size, mode):
    if tensor.dim() != 3:
        raise ValueError("Expected tensor shape (C, H, W).")
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(size, size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    ).squeeze(0)


def _random_scale_and_crop(image, mask, out_size, g, scale_range=(0.75, 1.25)):
    # 1) scale
    _, h, w = image.shape
    scale = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand((), generator=g).item()
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image_s = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    mask_s = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)

    # 2) crop (or pad) to out_size
    _, hs, ws = image_s.shape

    if hs >= out_size:
        top = int(torch.randint(0, hs - out_size + 1, (1,), generator=g).item())
        image_s = image_s[:, top : top + out_size, :]
        mask_s = mask_s[:, top : top + out_size, :]
    else:
        pad = out_size - hs
        pad_top = int(torch.randint(0, pad + 1, (1,), generator=g).item())
        pad_bottom = pad - pad_top
        image_s = F.pad(image_s, (0, 0, pad_top, pad_bottom), mode="constant", value=0.0)
        mask_s = F.pad(mask_s, (0, 0, pad_top, pad_bottom), mode="constant", value=0.0)

    _, hs, ws = image_s.shape
    if ws >= out_size:
        left = int(torch.randint(0, ws - out_size + 1, (1,), generator=g).item())
        image_s = image_s[:, :, left : left + out_size]
        mask_s = mask_s[:, :, left : left + out_size]
    else:
        pad = out_size - ws
        pad_left = int(torch.randint(0, pad + 1, (1,), generator=g).item())
        pad_right = pad - pad_left
        image_s = F.pad(image_s, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)
        mask_s = F.pad(mask_s, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)

    return image_s, mask_s


def _random_rotate(image, mask, g, degrees=10):
    # small rotation around center, same transform on image+mask
    if degrees <= 0:
        return image, mask

    angle = (torch.rand((), generator=g).item() * 2 - 1) * degrees
    theta = torch.tensor([
        [torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0.0],
        [torch.sin(torch.deg2rad(torch.tensor(angle))),  torch.cos(torch.deg2rad(torch.tensor(angle))), 0.0],
    ], dtype=torch.float32).unsqueeze(0)

    grid = F.affine_grid(theta, size=(1, image.shape[0], image.shape[1], image.shape[2]), align_corners=False)
    img_r = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    msk_r = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False).squeeze(0)
    return img_r, msk_r


def apply_train_augs(image, mask, out_size, seed, idx):
    # Use per-sample deterministic generator (stable across workers)
    g = torch.Generator()
    g.manual_seed(int(seed) * 1_000_003 + int(idx))

    # Make sure mask is binary-ish
    mask = (mask > 0.5).float()

    # 1) flip
    if _rand_bool(0.5, g):
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[2])

    # 2) scale jitter + crop/pad to out_size
    image, mask = _random_scale_and_crop(image, mask, out_size=out_size, g=g)

    # 3) small rotation (optional but helpful)
    if _rand_bool(0.7, g):
        image, mask = _random_rotate(image, mask, g=g, degrees=8)

    # 4) photometric (image only)
    image = _color_jitter(image, g=g, brightness=0.12, contrast=0.12)

    # 5) noise (image only)
    if _rand_bool(0.3, g):
        image = _add_noise(image, g=g, sigma=0.02)

    # keep mask in {0,1}
    mask = (mask > 0.5).float()
    return image, mask


# ----------------------------
# Dataset + dataloader
# ----------------------------
class SuperviselyPersonsDataset(Dataset):
    def __init__(self, splits_json, split, size=None, augment=False, augment_seed=0):
        splits_json = Path(splits_json)
        payload = json.loads(splits_json.read_text())
        self.dataset_root = Path(payload["dataset_root"])
        self.items = payload["splits"][split]
        self.size = size
        self.augment = augment
        self.augment_seed = augment_seed

        if not self.items:
            raise ValueError(f"Split '{split}' is empty in {splits_json}.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_path = self.dataset_root / item["image"]
        mask_path = self.dataset_root / item["mask"]

        image = _load_image(image_path)
        mask = _load_mask(mask_path)

        # If size is given, training uses size as final output dimension.
        if self.size is not None:
            if self.augment:
                image, mask = apply_train_augs(image, mask, out_size=self.size, seed=self.augment_seed, idx=idx)
            else:
                image = _resize_tensor(image, self.size, mode="bilinear")
                mask = _resize_tensor(mask, self.size, mode="nearest")
                mask = (mask > 0.5).float()
        else:
            # If no resizing, still binarize mask
            mask = (mask > 0.5).float()

        return image, mask


def create_dataloader(
    splits_json,
    split,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    size=None,
    augment=None,        # if None: auto (train=True else False)
    augment_seed=0,
):
    if augment is None:
        augment = (split == "train")

    dataset = SuperviselyPersonsDataset(
        splits_json=splits_json,
        split=split,
        size=size,
        augment=augment,
        augment_seed=augment_seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ----------------------------
# Helpers for matching masks
# ----------------------------
def _build_mask_maps(mask_dir):
    candidates = [
        p for p in mask_dir.glob("*")
        if p.suffix.lower() in IMAGE_EXTS.union(JSON_EXTS)
    ]
    by_name = {p.name: p for p in candidates}
    by_stem = {p.stem: p for p in candidates}
    return {"by_name": by_name, "by_stem": by_stem}


def _match_mask(image_path, mask_map):
    by_name = mask_map["by_name"]
    by_stem = mask_map["by_stem"]
    direct = by_name.get(image_path.name)
    if direct is not None:
        return direct
    json_match = by_name.get(f"{image_path.name}.json")
    if json_match is not None:
        return json_match
    return by_stem.get(image_path.stem)


# ----------------------------
# Export utilities (unchanged)
# ----------------------------
def export_split_folders(splits_json, out_dir):
    splits_json = Path(splits_json)
    payload = json.loads(splits_json.read_text())
    dataset_root = Path(payload["dataset_root"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in payload["splits"].items():
        images_out = out_dir / split_name / "images"
        masks_out = out_dir / split_name / "masks"
        images_out.mkdir(parents=True, exist_ok=True)
        masks_out.mkdir(parents=True, exist_ok=True)

        for item in items:
            image_src = dataset_root / item["image"]
            mask_src = dataset_root / item["mask"]
            shutil.copy2(image_src, images_out / image_src.name)
            shutil.copy2(mask_src, masks_out / mask_src.name)


def export_example_layout(splits_json, out_dir):
    splits_json = Path(splits_json)
    payload = json.loads(splits_json.read_text())
    dataset_root = Path(payload["dataset_root"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in payload["splits"].items():
        mapped_split = "eval" if split_name == "val" else split_name
        input_out = out_dir / mapped_split / "input"
        target_out = out_dir / mapped_split / "target"
        input_out.mkdir(parents=True, exist_ok=True)
        target_out.mkdir(parents=True, exist_ok=True)

        for item in items:
            image_src = dataset_root / item["image"]
            mask_src = dataset_root / item["mask"]

            shutil.copy2(image_src, input_out / image_src.name)

            mask_tensor = _load_mask(mask_src)
            mask_img = Image.fromarray(
                (mask_tensor.squeeze(0).clamp(0, 1) * 255).byte().numpy()
            )
            mask_name = f"{image_src.stem}.png"
            mask_img.save(target_out / mask_name)


# ----------------------------
# CLI (unchanged)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download Supervisely Persons and create train/val/test splits."
    )
    parser.add_argument("--download", action="store_true", help="Download dataset.")
    parser.add_argument("--dst-dir", default="~/dataset-ninja/", help="Download root.")
    parser.add_argument("--dataset-root", default=None, help="Existing dataset root.")
    parser.add_argument("--out-dir", default="data/supervisely_persons")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument(
        "--export-data",
        default=None,
        help="Copy images/masks into train/val/test folders under this directory.",
    )
    parser.add_argument(
        "--export-example-layout",
        default=None,
        help="Export to train/eval/test input/target layout for data_loader_example.py.",
    )
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if args.download:
        dataset_root = download_supervisely_persons(args.dst_dir)

    if dataset_root is None:
        raise ValueError("Provide --dataset-root or use --download.")

    splits_path = create_splits(
        dataset_root=dataset_root,
        out_dir=args.out_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    print(f"Wrote splits to {splits_path}")

    if args.export_data:
        export_split_folders(splits_path, args.export_data)
        print(f"Copied split data to {Path(args.export_data).resolve()}")

    if args.export_example_layout:
        export_example_layout(splits_path, args.export_example_layout)
        print(f"Wrote example layout to {Path(args.export_example_layout).resolve()}")


if __name__ == "__main__":
    main()