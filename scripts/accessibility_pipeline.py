"""End-to-end pipeline:
영상 → 프레임 → depth + segmentation → overlay 영상 저장.

사용:
    python scripts/accessibility_pipeline.py --input videos/sample.mp4 --output outputs/

옵션:
    --stride N        N프레임마다 1장 처리 (기본 1 = 모든 프레임)
    --max-frames K    K프레임까지만 처리 (디버그용, 기본 0 = 제한 없음)
    --depth-model …   HF 모델 ID (기본 small)
    --seg-weights …   YOLO 가중치 경로 (기본 checkpoints/yolov8n-seg.pt)
    --device …        cpu | cuda (생략 시 자동)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# scripts/ 디렉토리를 import 경로로 보장
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from infer_depth import (  # noqa: E402
    DEFAULT_DEPTH_MODEL,
    colorize_depth,
    is_metric_model,
    load_depth_model,
    run_depth,
)
from infer_seg import load_seg_model, run_seg  # noqa: E402


def pick_device(prefer: str | None) -> str:
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_panel(
    frame_bgr: np.ndarray, depth_color: np.ndarray, seg_vis: np.ndarray
) -> np.ndarray:
    """원본 | seg | depth 3분할 패널 (가로)."""
    h, w = frame_bgr.shape[:2]
    if depth_color.shape[:2] != (h, w):
        depth_color = cv2.resize(depth_color, (w, h))
    if seg_vis.shape[:2] != (h, w):
        seg_vis = cv2.resize(seg_vis, (w, h))
    return np.hstack([frame_bgr, seg_vis, depth_color])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="입력 영상 경로")
    p.add_argument("--output", required=True, help="출력 디렉토리")
    p.add_argument("--stride", type=int, default=1, help="N프레임마다 1장 처리")
    p.add_argument(
        "--max-frames", type=int, default=0, help="처리할 최대 프레임 수 (0=무제한)"
    )
    p.add_argument(
        "--depth-model",
        default=DEFAULT_DEPTH_MODEL,
        help="HF 모델 ID (기본: Metric Outdoor Small)",
    )
    p.add_argument("--seg-weights", default="checkpoints/yolov8n-seg.pt")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--save-sample",
        action="store_true",
        help="첫 처리 프레임을 outputs/sample.png 로 저장",
    )
    args = p.parse_args()

    device = pick_device(args.device)
    metric = is_metric_model(args.depth_model)
    unit = "m" if metric else "(relative)"
    print(f"[pipeline] device={device}  depth_unit={unit}  depth_model={args.depth_model}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.input}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[pipeline] in: {args.input}  {width}x{height} @ {fps_in:.2f}fps  frames={total_frames}")

    fps_out = max(1.0, fps_in / max(1, args.stride))
    panel_w = width * 3  # 원본 | seg | depth
    panel_h = height
    out_path = out_dir / "overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (panel_w, panel_h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"cannot open writer: {out_path}")

    print(f"[pipeline] out: {out_path}  panel={panel_w}x{panel_h} @ {fps_out:.2f}fps")

    print("[pipeline] loading models...")
    depth_model, depth_proc = load_depth_model(args.depth_model, device)
    seg_model = load_seg_model(args.seg_weights, device)

    processed = 0
    frame_idx = 0
    expected = (
        total_frames // max(1, args.stride) if total_frames else None
    )
    if args.max_frames > 0:
        expected = args.max_frames if expected is None else min(expected, args.max_frames)

    pbar = tqdm(total=expected, desc="frames", unit="f")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            depth = run_depth(depth_model, depth_proc, frame, device)
            depth_color = colorize_depth(depth, annotate_meters=metric)

            seg_result = run_seg(seg_model, frame, device)
            seg_vis = seg_result.plot()  # BGR

            panel = make_panel(frame, depth_color, seg_vis)
            writer.write(panel)

            if processed == 0:
                print(
                    f"[pipeline] frame0 depth range = "
                    f"({float(depth.min()):.2f}, {float(depth.max()):.2f}) {unit}"
                )
                if args.save_sample:
                    sample_path = out_dir / "sample.png"
                    cv2.imwrite(str(sample_path), panel)
                    print(f"[pipeline] sample frame -> {sample_path}")

            processed += 1
            pbar.update(1)
            frame_idx += 1

            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"[pipeline] done. processed={processed} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
