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
    parser = argparse.ArgumentParser(description="Real-time split-screen segmentation demo.")
    parser.add_argument(
        "--weights",
        default="outputs/supervisely_run/model_best.pt",
        help="Path to trained model weights (.pt) or an ONNX export (.onnx).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLOv8-seg ONNX outputs.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS when using YOLOv8-seg ONNX outputs.",
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
        default="Segmentation Demo",
        help="Window title.",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Show a third panel with background blurred outside the mask.",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=15.0,
        help="Gaussian blur strength for background blur view.",
    )
    return parser.parse_args()


def prepare_frame(frame_bgr, size):
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor


def prepare_frame_numpy(frame_bgr, size):
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return array


def tensor_to_gray_bgr(tensor):
    image = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def mask_to_gray_bgr(mask_np):
    image = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def apply_background_blur(frame_bgr, mask_np, sigma):
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=sigma)
    mask_3 = (mask_np[..., None] > 0.5).astype(np.uint8)
    return np.where(mask_3 == 1, frame_bgr, blurred)


def logits_to_mask(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def create_onnx_session(weights_path, use_cuda):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if use_cuda:
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(str(weights_path), providers=providers)


def get_onnx_input_size(session):
    shape = session.get_inputs()[0].shape
    if len(shape) != 4:
        return None
    height = shape[2]
    width = shape[3]
    if isinstance(height, int) and isinstance(width, int):
        return height, width
    return None


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(image_bgr, new_size, color=(114, 114, 114)):
    height, width = image_bgr.shape[:2]
    scale = min(new_size / height, new_size / width)
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, (left, top)


def scale_boxes(boxes_xyxy, scale, pad, orig_shape):
    pad_x, pad_y = pad
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    h, w = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def unletterbox_mask(mask, input_size, orig_shape, scale, pad):
    pad_x, pad_y = pad
    h0, w0 = orig_shape
    x1 = int(pad_x)
    y1 = int(pad_y)
    x2 = int(min(input_size, pad_x + round(w0 * scale)))
    y2 = int(min(input_size, pad_y + round(h0 * scale)))
    cropped = mask[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((h0, w0), dtype=np.uint8)
    return cv2.resize(cropped, (w0, h0), interpolation=cv2.INTER_NEAREST)


def nms_xyxy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def yolo8seg_postprocess(
    output0,
    output1,
    input_size,
    orig_shape,
    scale,
    pad,
    conf_thres,
    iou_thres,
):
    preds = output0[0].transpose(1, 0)
    num_mask = 32
    num_classes = preds.shape[1] - 4 - num_mask
    if num_classes <= 0:
        return None

    boxes_xywh = preds[:, :4]
    scores = preds[:, 4 : 4 + num_classes]
    mask_coeffs = preds[:, 4 + num_classes :]

    if num_classes == 1:
        conf = scores[:, 0]
        class_ids = np.zeros_like(conf, dtype=np.int64)
    else:
        class_ids = np.argmax(scores, axis=1)
        conf = scores[np.arange(scores.shape[0]), class_ids]

    keep = conf >= conf_thres
    if not np.any(keep):
        return None

    boxes_xywh = boxes_xywh[keep]
    conf = conf[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    x = boxes_xywh[:, 0]
    y = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_indices = nms_xyxy(boxes_xyxy, conf, iou_thres)
    boxes_xyxy = boxes_xyxy[keep_indices]
    conf = conf[keep_indices]
    class_ids = class_ids[keep_indices]
    mask_coeffs = mask_coeffs[keep_indices]

    proto = output1[0]
    proto_h, proto_w = proto.shape[1], proto.shape[2]
    proto_flat = proto.reshape(proto.shape[0], -1)
    masks = sigmoid_np(mask_coeffs @ proto_flat)
    masks = masks.reshape(-1, proto_h, proto_w)

    input_h, input_w = input_size
    combined = np.zeros((input_h, input_w), dtype=np.uint8)
    for i, m in enumerate(masks):
        mask_up = cv2.resize(m, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        box = boxes_xyxy[i]
        x1, y1, x2, y2 = box
        x1 = int(max(0, np.floor(x1)))
        y1 = int(max(0, np.floor(y1)))
        x2 = int(min(input_w - 1, np.ceil(x2)))
        y2 = int(min(input_h - 1, np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        mask_bin = (mask_up > 0.5).astype(np.uint8)
        mask_bin[:y1, :] = 0
        mask_bin[y2:, :] = 0
        mask_bin[:, :x1] = 0
        mask_bin[:, x2:] = 0
        combined = np.maximum(combined, mask_bin)

    combined = unletterbox_mask(combined, input_w, orig_shape, scale, pad)
    return combined


def main():
    args = parse_args()
    weights_path = Path(args.weights)
    use_onnx = weights_path.suffix.lower() == ".onnx"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_onnx:
        session = create_onnx_session(weights_path, device.type == "cuda")
        input_name = session.get_inputs()[0].name
        fixed_size = get_onnx_input_size(session)
        if fixed_size is not None:
            if args.size != fixed_size[0] or args.size != fixed_size[1]:
                print(
                    f"ONNX model expects {fixed_size[0]}x{fixed_size[1]} inputs; "
                    f"overriding --size {args.size}."
                )
            args.size = fixed_size[0]
    else:
        root = Path(__file__).resolve().parent
        model_path = root / "02_model" / "model_example.py"
        model_module = load_module("model_example", model_path)
        model = model_module.UNet3(out_channels=1, base=64).to(device)
        state = torch.load(weights_path, map_location=device)
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

            if use_onnx:
                padded, scale, pad = letterbox(frame, args.size)
                array = prepare_frame_numpy(padded, args.size)
                outputs = session.run(None, {input_name: array})
                if len(outputs) == 1:
                    output_tensor = torch.from_numpy(outputs[0])
                    mask = logits_to_mask(output_tensor)
                else:
                    mask_np = yolo8seg_postprocess(
                        outputs[0],
                        outputs[1],
                        (args.size, args.size),
                        (frame.shape[0], frame.shape[1]),
                        scale,
                        pad,
                        args.conf,
                        args.iou,
                    )
                    if mask_np is None:
                        mask = torch.zeros(1, 1, frame.shape[0], frame.shape[1])
                    else:
                        mask = torch.from_numpy(mask_np)[None, None, ...].float()
            else:
                tensor = prepare_frame(frame, args.size)
                tensor = tensor.to(device)
                with torch.no_grad():
                    output = model(tensor)
                    mask = logits_to_mask(output)

            mask_np = (
                mask.squeeze(0)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if mask_np.shape[:2] != (frame.shape[0], frame.shape[1]):
                mask_np = cv2.resize(
                    mask_np,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            processed = mask_to_gray_bgr(mask_np)
            if args.blur:
                blurred = apply_background_blur(frame, mask_np, args.blur_sigma)
                combined = cv2.hconcat([frame, processed, blurred])
            else:
                combined = cv2.hconcat([frame, processed])
            cv2.imshow(args.window, combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
