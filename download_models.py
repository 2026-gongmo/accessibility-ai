"""Download pretrained checkpoints used by the pipeline.

기본:
- yolov8n-seg.pt           -> checkpoints/yolov8n-seg.pt
- Depth Anything V2 (HF)   -> ~/.cache/huggingface (transformers 캐시)

옵션:
- --depth-pth small|base|both
  Depth Anything V2 원본 .pth 체크포인트도 checkpoints/ 아래에 받는다.
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
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}


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


def _prefetch_hf_depth(size: str) -> None:
    """transformers/HF 캐시에 Depth Anything V2 가중치를 미리 받아둔다."""
    repo = HF_DEPTH_REPOS[size]
    print(f"[hf] prefetch {repo}")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError:
        print("[warn] transformers 미설치. requirements 설치 후 다시 실행하라.")
        return
    AutoImageProcessor.from_pretrained(repo)
    AutoModelForDepthEstimation.from_pretrained(repo)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrained model downloader")
    parser.add_argument(
        "--depth-hf",
        choices=["small", "base", "both", "none"],
        default="small",
        help="HF transformers 캐시에 미리 받아둘 Depth Anything V2 사이즈 (default: small)",
    )
    parser.add_argument(
        "--depth-pth",
        choices=["small", "base", "both", "none"],
        default="none",
        help="원본 .pth 체크포인트도 checkpoints/ 에 다운로드 (default: none)",
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

    if args.depth_hf != "none":
        sizes = ["small", "base"] if args.depth_hf == "both" else [args.depth_hf]
        for s in sizes:
            _prefetch_hf_depth(s)

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
