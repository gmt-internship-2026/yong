#!/usr/bin/env bash
# gesture_kiosk 실행 (라즈베리파이5). 종료: Ctrl+C. UI 없이 이벤트만: bash run.sh --headless
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f venv_rpi/bin/activate ]; then
  echo "[FAIL] 설치가 안 되어 있습니다 — install.sh 를 먼저 실행하세요"
  exit 1
fi
# shellcheck disable=SC1091
source venv_rpi/bin/activate

echo "[INFO] 제스처 민원발급기 데모 시작 — 브라우저: http://<라즈베리파이IP>:5000"
echo "       종료: 이 창에서 Ctrl+C"
python scripts/run_demo.py "$@"
