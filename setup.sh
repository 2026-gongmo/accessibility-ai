#!/usr/bin/env bash
# venv 생성 + requirements 설치 + 모델 다운로드까지 한 번에.
# 사용: bash setup.sh        (CPU)
#       bash setup.sh cu121  (CUDA 12.1 torch 설치)

set -euo pipefail

VARIANT="${1:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] creating venv -> ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# venv activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

case "${VARIANT}" in
  cpu)
    echo "[setup] installing CPU torch + requirements"
    pip install -r requirements.txt
    ;;
  cu121)
    echo "[setup] installing CUDA 12.1 torch + requirements"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ;;
  cu118)
    echo "[setup] installing CUDA 11.8 torch + requirements"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ;;
  *)
    echo "unknown variant: ${VARIANT} (cpu | cu121 | cu118)"
    exit 1
    ;;
esac

echo "[setup] downloading pretrained models"
python download_models.py

echo "[setup] done. Activate:  source ${VENV_DIR}/bin/activate"
