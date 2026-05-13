"""Download pretrained checkpoints used by the pipeline.

기본:
- yolov8n-seg.pt                              -> checkpoints/yolov8n-seg.pt
- Depth Anything V2 Metric Outdoor Small (HF) -> ~/.cache/huggingface

옵션:
- --depth-hf {relative-small,relative-base,
              metric-outdoor-small,metric-outdoor-large,
              metric-indoor-small,metric-indoor-large,none}
  여러 개를 미리 받고 싶으면 콤마로: --depth-hf metric-outdoor-small,metric-indoor-small
- --depth-pth {small,base,both,none}
  Depth Anything V2 원본 relative .pth 체크포인트도 checkpoints/ 에 받는다.
  (현재 파이프라인은 transformers 버전만 사용. 원본 repo로 갈아끼울 때 대비)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

YOLO_SEG_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"
)

DEPTH_PTH_URLS = {
    "small": (
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/"
        "resolve/main/depth_anything_v2_vits.pth"
    ),
    "base": (
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/"
        "resolve/main/depth_anything_v2_vitb.pth"
    ),
}

HF_DEPTH_REPOS = {
    "relative-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "relative-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "metric-outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "metric-outdoor-large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "metric-indoor-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "metric-indoor-large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}

DEPTH_HF_CHOICES = list(HF_DEPTH_REPOS.keys()) + ["none"]


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url}\n        -> {dst}")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        if dst.exists():
            dst.unlink(missing_ok=True)
        raise SystemExit(f"[error] download failed: {e}") from e


def _prefetch_hf_depth(key: str) -> None:
    """transformers/HF 캐시에 Depth Anything V2 가중치를 미리 받아둔다."""
    repo = HF_DEPTH_REPOS[key]
    print(f"[hf] prefetch {repo}")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        print("[warn] transformers 미설치. requirements 설치 후 다시 실행하라.")
        return
    AutoImageProcessor.from_pretrained(repo)
    AutoModelForDepthEstimation.from_pretrained(repo)


def _parse_depth_hf(value: str) -> list[str]:
    if value == "none":
        return []
    keys = [v.strip() for v in value.split(",") if v.strip()]
    bad = [k for k in keys if k not in HF_DEPTH_REPOS]
    if bad:
        raise SystemExit(
            f"[error] unknown --depth-hf key(s): {bad}. valid: {DEPTH_HF_CHOICES}"
        )
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrained model downloader")
    parser.add_argument(
        "--depth-hf",
        default="metric-outdoor-small",
        help=(
            "HF transformers 캐시에 미리 받아둘 Depth Anything V2 키 (콤마 구분 다중). "
            f"선택지: {DEPTH_HF_CHOICES} (default: metric-outdoor-small)"
        ),
    )
    parser.add_argument(
        "--depth-pth",
        choices=["small", "base", "both", "none"],
        default="none",
        help="원본 relative .pth 체크포인트도 checkpoints/ 에 다운로드 (default: none)",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="yolov8n-seg.pt 다운로드 생략",
    )
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_yolo:
        _download(YOLO_SEG_URL, CHECKPOINTS_DIR / "yolov8n-seg.pt")

    for key in _parse_depth_hf(args.depth_hf):
        _prefetch_hf_depth(key)

    if args.depth_pth != "none":
        sizes = ["small", "base"] if args.depth_pth == "both" else [args.depth_pth]
        for s in sizes:
            url = DEPTH_PTH_URLS[s]
            fname = Path(url).name
            _download(url, CHECKPOINTS_DIR / fname)

    print("[done] checkpoints ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
