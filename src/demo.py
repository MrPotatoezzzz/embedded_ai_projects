#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time split-screen edge demo.")
    parser.add_argument(
        "--weights",
        default="outputs/example_edge_model/model_best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Resize camera frames to this square size for inference.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g. 0) or a video/GStreamer pipeline string.",
    )
    parser.add_argument(
        "--window",
        default="Edge Demo",
        help="Window title.",
    )
    return parser.parse_args()


def prepare_frame(frame_bgr, size):
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def tensor_to_bgr(tensor):
    image = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    model_path = root / "02_model" / "model_example.py"
    model_module = load_module("model_example", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_module.SmallUNet().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera/source: {args.source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tensor = prepare_frame(frame, args.size)
            tensor = tensor.to(device)

            with torch.no_grad():
                output = model(tensor)

            processed = tensor_to_bgr(output)
            processed = cv2.resize(
                processed,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            combined = cv2.hconcat([frame, processed])
            cv2.imshow(args.window, combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
