"""Depth Anything V2 (transformers) 단일 이미지/프레임 추론.

기본 모델은 Metric Outdoor Small (도시 보행 가정).
모델 ID에 "Metric" 이 들어있으면 출력은 미터 단위로 해석되고,
콘솔/이미지에 m 단위 캡션이 붙는다.

사용 예:
    python scripts/infer_depth.py --image videos/sample.jpg --output outputs/depth.png

라이브러리로 import 해서 쓰는 경우:
    model, processor = load_depth_model(
        "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "cuda")
    depth = run_depth(model, processor, frame_bgr, "cuda")   # (H,W) float32 meters
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_metric_model(repo: str) -> bool:
    return "metric" in repo.lower()


def load_depth_model(
    repo: str = DEFAULT_DEPTH_MODEL,
    device: str = "cpu",
):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForDepthEstimation.from_pretrained(repo)
    model.to(device).eval()
    return model, processor


@torch.no_grad()
def run_depth(model, processor, frame_bgr: np.ndarray, device: str) -> np.ndarray:
    """BGR 이미지를 받아 (H, W) float32 depth 맵을 반환.

    Metric 모델이면 단위는 meters, Relative 모델이면 임의 스케일.
    """
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


def colorize_depth(depth: np.ndarray, annotate_meters: bool = False) -> np.ndarray:
    """float depth → 8bit colormap (BGR). annotate_meters=True 면 좌상단에 미터 캡션."""
    d_min, d_max = float(depth.min()), float(depth.max())
    norm = (depth - d_min) / (d_max - d_min + 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

    if annotate_meters:
        text = f"depth: {d_min:.2f} m  -  {d_max:.2f} m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (0, 0), (tw + 16, th + 14), (0, 0, 0), -1)
        cv2.putText(
            vis, text, (8, th + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return vis


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="입력 이미지 경로")
    p.add_argument("--output", required=True, help="출력 colormap 이미지 경로")
    p.add_argument(
        "--model",
        default=DEFAULT_DEPTH_MODEL,
        help="HF 모델 ID (기본: Metric Outdoor Small)",
    )
    p.add_argument("--device", default=None, help="cpu | cuda (생략 시 자동)")
    args = p.parse_args()

    device = pick_device(args.device)
    metric = is_metric_model(args.model)
    unit = "m" if metric else "(relative)"
    print(f"[depth] device={device} model={args.model} unit={unit}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    model, processor = load_depth_model(args.model, device)
    depth = run_depth(model, processor, frame, device)
    vis = colorize_depth(depth, annotate_meters=metric)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    print(
        f"[depth] saved {args.output}  range=({depth.min():.3f}, {depth.max():.3f}) {unit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
