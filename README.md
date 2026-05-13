## accessibility-ai

영상 한 편을 입력하면 **Depth Anything V2**로 깊이를 추정하고
**Ultralytics YOLOv8-seg**로 인스턴스 세그멘테이션을 돌려
원본 / seg / depth 3분할 오버레이 영상을 만들어 주는 MVP.

- pretrained 모델만 사용 (fine-tuning 없음)
- CPU에서도 동작 (느림). CUDA 있으면 자동으로 GPU 사용
- Android / GPS / DB 없음

### 디렉토리

```
accessibility-ai/
├── requirements.txt
├── setup.sh
├── download_models.py
├── scripts/
│   ├── infer_depth.py
│   ├── infer_seg.py
│   └── accessibility_pipeline.py
├── checkpoints/      # 모델 가중치 (gitignore)
├── videos/           # 입력 영상 (gitignore)
├── outputs/          # 결과 영상 (gitignore)
└── models/           # 모델 관련 작업 폴더 (gitignore)
```

### 1. venv 생성

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. requirements 설치

#### CPU (기본)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### GPU (CUDA)

PyTorch는 CUDA 버전에 따라 별도 휠을 받아야 한다. 시스템 CUDA에 맞춰 골라 설치한다.

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 그 다음 나머지 패키지
pip install -r requirements.txt
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True`가 나오면 GPU 사용 가능.

#### 한 방에 (Linux/macOS)

```bash
bash setup.sh         # CPU
bash setup.sh cu121   # CUDA 12.1
bash setup.sh cu118   # CUDA 11.8
```

### 3. pretrained 모델 다운로드

```bash
python download_models.py
```

기본 동작:

- `checkpoints/yolov8n-seg.pt` 다운로드
- `transformers` HF 캐시에 `Depth-Anything-V2-Metric-Outdoor-Small-hf` 미리 받아둠
  (단위가 **미터**로 나오는 모델. 도시 보행 도메인 기본값)

옵션:

```bash
# 실내(쇼핑몰, 지하철 등) 데이터에 맞춰 Metric Indoor 추가
python download_models.py --depth-hf metric-indoor-small

# 정밀도가 더 필요하면 Large
python download_models.py --depth-hf metric-outdoor-large

# 여러 개 한 번에 (콤마 구분)
python download_models.py --depth-hf metric-outdoor-small,metric-indoor-small

# Relative depth 모델 (단위 없음, 시각화만 필요할 때)
python download_models.py --depth-hf relative-small

# 원본 repo .pth 파일도 받기 (현 파이프라인은 안 씀, 보존용)
python download_models.py --depth-pth small
python download_models.py --depth-pth both
```

사용 가능한 키:
`relative-small`, `relative-base`,
`metric-outdoor-small`, `metric-outdoor-large`,
`metric-indoor-small`, `metric-indoor-large`, `none`.

### 4. inference 실행

#### 전체 파이프라인 (영상 → 오버레이 영상)

```bash
python scripts/accessibility_pipeline.py \
  --input videos/sample.mp4 \
  --output outputs/
```

결과:

- `outputs/overlay.mp4` — 원본 | seg | depth 3분할
- `--save-sample` 주면 첫 프레임을 `outputs/sample.png`로 저장
- 콘솔에 `frame0 depth range = (a, b) m` 줄이 찍히면 metric depth 정상 동작
- depth 패널 좌상단에 `depth: a.aa m - b.bb m` 캡션이 박힘

자주 쓰는 옵션:

| 옵션 | 설명 | 기본 |
| --- | --- | --- |
| `--stride N` | N프레임마다 1장 처리 (CPU에서 권장: 5~10) | `1` |
| `--max-frames K` | 처음 K프레임만 처리 (디버그) | `0` (무제한) |
| `--device cpu` / `cuda` | 강제 지정 | 자동 |
| `--depth-model …` | HF 모델 ID 변경 | `…Metric-Outdoor-Small-hf` |
| `--seg-weights …` | YOLO 가중치 경로 | `checkpoints/yolov8n-seg.pt` |

모델 ID 안에 `Metric` 이 들어있으면 단위는 미터, 없으면 임의 스케일이다.
실내 데이터면 `--depth-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`.

#### 단일 이미지 테스트

```bash
# Depth
python scripts/infer_depth.py --image videos/sample.jpg --output outputs/depth.png

# Segmentation
python scripts/infer_seg.py --image videos/sample.jpg --output outputs/seg.png
```

### 5. CPU 환경 빠른 점검

CPU에서는 한 프레임당 수 초가 걸린다. 우선 동작 확인만 한다면:

```bash
python scripts/accessibility_pipeline.py \
  --input videos/sample.mp4 \
  --output outputs/ \
  --stride 30 \
  --max-frames 5 \
  --save-sample
```

`outputs/sample.png`와 `outputs/overlay.mp4`가 생성되면 환경 셋업 완료.

### 6. GPU 환경 설명

- `--device`를 따로 주지 않으면 `torch.cuda.is_available()`로 자동 선택.
- `depth` / `seg` 모두 GPU에 올라간다.
- 메모리 부족 시 `--depth-model` 그대로 두고(`small`이 가장 가벼움) `--stride`를 키워 처리량을 조절한다.

### 사용 오픈소스

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — `transformers` 통합 모델 사용
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — `yolov8n-seg`

### 제외 범위 (현재 단계)

- fine-tuning
- Android 앱 구현
- GPS 통합
- DB 연동
