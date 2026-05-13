"""YOLOv8n-seg 단일 이미지/프레임 추론.

사용 예:
    python scripts/infer_seg.py --image videos/sample.jpg --output outputs/seg.png

라이브러리로:
    model = load_seg_model("checkpoints/yolov8n-seg.pt", "cuda")
    result = run_seg(model, frame_bgr, "cuda")
    vis = result.plot()  # BGR ndarray
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_seg_model(weights: str = "checkpoints/yolov8n-seg.pt", device: str = "cpu"):
    from ultralytics import YOLO

    # ultralytics는 파일 없으면 자동 다운로드. 경로가 없는 경우 모델 이름만 넘긴다.
    if not Path(weights).exists():
        print(f"[seg] {weights} 없음 -> ultralytics 기본 캐시에서 yolov8n-seg.pt 사용")
        weights = "yolov8n-seg.pt"
    model = YOLO(weights)
    model.to(device)
    return model


def run_seg(model, frame_bgr, device: str, conf: float = 0.25):
    results = model.predict(
        frame_bgr,
        device=device,
        conf=conf,
        verbose=False,
    )
    return results[0]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", default="checkpoints/yolov8n-seg.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[seg] device={device} weights={args.weights}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    model = load_seg_model(args.weights, device)
    result = run_seg(model, frame, device, conf=args.conf)
    vis = result.plot()  # BGR

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, vis)
    n = 0 if result.boxes is None else len(result.boxes)
    print(f"[seg] saved {args.output}  detections={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
