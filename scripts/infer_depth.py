"""Depth Anything V2 (transformers) 단일 이미지/프레임 추론.

사용 예:
    python scripts/infer_depth.py --image videos/sample.jpg --output outputs/depth.png

라이브러리로 import 해서 쓰는 경우:
    model, processor = load_depth_model("depth-anything/Depth-Anything-V2-Small-hf", "cuda")
    depth = run_depth(model, processor, frame_bgr, "cuda")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_depth_model(
    repo: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cpu",
):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo)
    model.to(device).eval()
    return model, processor


@torch.no_grad()
def run_depth(model, processor, frame_bgr: np.ndarray, device: str) -> np.ndarray:
    """BGR 이미지를 받아 (H, W) float32 depth 맵을 반환."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    pred = outputs.predicted_depth  # [1, H, W]
    h, w = frame_bgr.shape[:2]
    depth = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return depth.astype(np.float32)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """float depth → 8bit colormap (BGR)."""
    d_min, d_max = float(depth.min()), float(depth.max())
    norm = (depth - d_min) / (d_max - d_min + 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--output", required=True, help="출력 colormap 이미지 경로")
    p.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HF 모델 ID (small/base)",
    )
    p.add_argument("--device", default=None, help="cpu | cuda (생략 시 자동)")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[depth] device={device} model={args.model}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    model, processor = load_depth_model(args.model, device)
    depth = run_depth(model, processor, frame, device)
    vis = colorize_depth(depth)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(f"[depth] saved {args.output}  range=({depth.min():.3f}, {depth.max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
