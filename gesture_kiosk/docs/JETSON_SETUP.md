# Jetson Orin Nano 개발자 키트 — 클론 후 셋업 가이드

작성: 2026-07-08. JetPack 6.x (Ubuntu 22.04 / Python 3.10 / CUDA 12.x) 기준.

## 0. 성능 모드 켜기 (30 FPS 목표라면 필수)

```bash
sudo nvpmodel -m 0        # 최대 전력 모드 (MAXN)
sudo jetson_clocks        # 클럭 고정 (재부팅 시 다시 실행)
```

## 1. 환경 확인

```bash
cat /etc/nv_tegra_release                 # JetPack(L4T) 버전 확인 → 기획서 9장 №2에 기록
python3 --version                         # 3.10이어야 함 (JetPack 6)
python3 -c "import tensorrt; print(tensorrt.__version__)"   # TensorRT는 JetPack에 내장
```

## 2. 저장소 클론 (Private이라 로그인 필요)

```bash
sudo apt update && sudo apt install -y git gh
gh auth login             # GitHub.com → HTTPS → 브라우저 로그인 선택
gh repo clone G0Sun9M0/GMtech_project
cd GMtech_project/gesture_kiosk
```

## 3. 가상환경 + 의존성 설치

```bash
# --system-site-packages 필수: JetPack 내장 tensorrt를 가상환경에서도 쓰기 위함
python3 -m venv venv --system-site-packages
source venv/bin/activate

# PyTorch는 일반 pip이 아니라 Jetson 전용 휠로 먼저 설치 (일반 휠은 GPU를 못 씀)
# JetPack 6.x / CUDA 12.6 기준:
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126

# 나머지 의존성 (torch가 이미 있으므로 덮어쓰지 않음)
pip install -r requirements.txt
```

GPU 인식 확인 — 반드시 True가 나와야 한다:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 기대 출력: True Orin
```

## 4. 모델 가중치 다운로드 (약 22MB, 학습 불필요)

```bash
python scripts/download_weights.py
```

## 5. 카메라 확인

```bash
ls /dev/video*                            # USB 웹캠 연결 확인
sudo apt install -y v4l-utils && v4l2-ctl --list-devices
```

`/dev/video0`이 아니면 `configs/config.yaml`의 `camera.device_id`를 해당 번호로 수정.

## 6. 1차 구동 (.pt 백엔드 — 동작 확인용)

```bash
python scripts/run_demo.py
```

같은 네트워크의 PC/폰 브라우저에서 `http://<젯슨IP>:5000` 접속 (IP는 `hostname -I`).
카메라 영상 + 제스처 인식 + 예시 키오스크 화면이 뜨면 성공.

## 7. TensorRT 엔진 빌드 → 실전 백엔드 전환

```bash
python scripts/build_engine.py            # 5~10분 소요, 1회만
```

완료 후 `configs/config.yaml`에서 `model.backend: torch` → `engine`으로 바꾸고 다시 실행.

## 8. 성능 측정 (기획서 6.1)

```bash
python scripts/benchmark.py               # 추론 단독 FPS (1,000프레임)
```

결과 수치를 `docs/`에 기록해 주간 보고에 사용.

## 문제 해결

| 증상 | 원인·해결 |
|---|---|
| `operator torchvision::nms does not exist` | torch와 torchvision 버전 불일치 — 둘 다 위의 jetson-ai-lab 인덱스에서 재설치 |
| `torch.cuda.is_available()` → False | 일반 pip torch가 설치됨 — `pip uninstall torch torchvision` 후 3단계 재실행 |
| 카메라를 열 수 없음 | `device_id` 불일치(5단계) 또는 다른 프로세스가 점유 중 |
| clone 시 403/404 | 저장소가 Private — `gh auth login` 먼저 |
| import tensorrt 실패 | 가상환경을 `--system-site-packages` 없이 만듦 — venv 재생성 |

## SD카드 128GB 운영 수칙 (재확인)

- 이 보드에서 **학습 금지** (`train.py`가 자동 차단함) — 학습은 개발 PC(맥)에서
- 데이터셋을 이 보드에 저장하지 않는다
- 남길 파일: 코드 + 가중치(.pt 22MB) + 엔진(.engine 수십 MB) 뿐
