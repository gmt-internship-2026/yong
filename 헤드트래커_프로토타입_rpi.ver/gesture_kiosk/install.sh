#!/usr/bin/env bash
# gesture_kiosk 설치 (라즈베리파이5, Python 3.11) — win.ver install.bat과 동일한 순서.
# 실행: bash install.sh   (또는 chmod +x install.sh 후 ./install.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "============================================================"
echo " gesture_kiosk 설치 (라즈베리파이5, aarch64)"
echo " 판정 엔진: MediaPipe FaceLandmarker — CPU 추론(GPU 불필요)"
echo "============================================================"

# ---- 1) Python 3.11 확인 -------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "[FAIL] python3을 찾지 못했습니다 — 라즈베리파이OS(Bookworm)는 기본 내장입니다"
  exit 1
fi
PY_VER=$(python3 --version | awk '{print $2}')
echo "[INFO] python3 ${PY_VER} 사용"
case "$PY_VER" in
  3.11.*) ;;
  *) echo "[경고] 배포 기준은 3.11.x 입니다 — 현재 ${PY_VER} (대체로 동작하나 기준과 다름)" ;;
esac

# ---- 2) 시스템 패키지 -----------------------------------------------------
# libcamera-apps/python3-picamera2: 공식 카메라 모듈(backend: picamera2) 쓸 때만 필요 —
#   최신 라즈베리파이OS 이미지엔 대개 이미 들어있어 설치가 즉시 끝난다.
# espeak-ng: pyttsx3의 리눅스 TTS 드라이버가 요구하는 시스템 패키지(pip 아님).
# libatlas-base-dev: numpy/opencv 선형대수 연산의 ARM 최적화 백엔드.
echo "[INFO] apt 패키지 확인/설치 (sudo 필요)..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip python3-dev \
  libatlas-base-dev espeak-ng v4l-utils \
  python3-picamera2 libcamera-apps || \
  echo "[경고] 일부 apt 패키지 설치 실패 — picamera2 백엔드를 안 쓸 거면 무시해도 됩니다"

# ---- 3) 가상환경 -----------------------------------------------------------
# --system-site-packages 필수: python3-picamera2는 apt가 시스템 파이썬에 까는
#   패키지라(libcamera 바인딩이 pip 휠로 안 나옴) venv가 이걸 봐야 backend: picamera2가
#   동작한다. v4l2만 쓸 거라면 없어도 되지만, 켜 둬도 해가 없다.
if [ ! -d venv_rpi ]; then
  echo "[INFO] 가상환경 생성 중..."
  python3 -m venv --system-site-packages venv_rpi
fi
# shellcheck disable=SC1091
source venv_rpi/bin/activate
pip install --upgrade pip -q

# ---- 4) 패키지 설치 ---------------------------------------------------------
echo "[INFO] pip 패키지 설치..."
if ! pip install -r requirements.txt; then
  echo "[FAIL] 패키지 설치 실패 — mediapipe/opencv-python의 aarch64 휠 존재 여부를 확인하세요"
  echo "       (requirements.txt 상단 주석 참고, 버전 핀을 범위로 완화 후 재시도)"
  exit 1
fi

# ---- 5) 모델 준비 -----------------------------------------------------------
python scripts/download_weights.py || {
  echo "[FAIL] 모델 다운로드 실패 — 오프라인이면 models/weights/face_landmarker.task를 직접 반입하세요"
  exit 1
}

# ---- 6) 스모크 테스트 --------------------------------------------------------
echo
echo "[INFO] 설치 검증 실행..."
python scripts/smoke_test.py || echo "[경고] 검증 실패 항목이 있습니다 — 설치가이드.md 문제 해결 참고"

echo
echo "[DONE] 설치 완료 — bash run.sh 로 실행하세요 (브라우저: http://<라즈베리파이IP>:5000)"
