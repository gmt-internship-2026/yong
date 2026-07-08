# gesture_kiosk — 제스처 인식 배리어프리 키오스크

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 손 제스처 5종을 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(고성모_기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 그대로 따른다.**

- 기반 코드: 캡스톤 프로젝트(Jetson Orin Nano + FastAPI + YOLO/TensorRT) 구조 이식
- 실행 보드: **Jetson Orin Nano** (SD카드 128GB — 학습 금지, 추론 전용)
- 모델: HaGRIDv2 사전학습 YOLOv10n 제스처 검출기 (**학습 0회로 즉시 동작**)

## 인식 제스처 (기획서 5.1)

| class_name | 동작 | 키오스크 명령 | 판정 방식 |
|---|---|---|---|
| point | 검지로 가리키기 | 항목 선택 | 정적 — N프레임 연속 검출 |
| palm_stop | 손바닥 정면 | 취소 / 뒤로 | 정적 — N프레임 연속 검출 |
| swipe_left | 손을 왼쪽으로 쓸기 | 이전 페이지 | 동적 — 손바닥 궤적 |
| swipe_right | 손을 오른쪽으로 쓸기 | 다음 페이지 | 동적 — 손바닥 궤적 |
| thumbs_up | 엄지 세우기 | 확인 / 결제 | 정적 — N프레임 연속 검출 |

정적 3종은 HaGRID 모델 클래스를 `config.yaml`의 `class_map`으로 매핑하고
(point←point/one, palm_stop←palm/stop, thumbs_up←like),
스와이프 2종은 손바닥 중심 좌표의 이동 궤적으로 후처리(`gesture_filter.py`)에서 판정한다.

## 폴더 구조 (기획서 2.3 + 예시 UI)

```
gesture_kiosk/
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ data/                    # raw / labeled / splits (4주차 데이터 수집 후 사용)
├─ models/weights|engines/  # .pt 가중치 / TensorRT .engine
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드
│   ├─ inference/preprocessor.py     # 전처리 (거울 반전)
│   ├─ inference/trt_engine.py       # 모델 로드·추론 (.pt / .engine)
│   ├─ postprocess/gesture_filter.py # 연속 프레임 판정·스와이프·쿨다운
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp 예시)
│   ├─ pipeline/demo_server.py       # ★ 예시 UI 서버 (회사 프로그램 대체 예정)
│   └─ utils/                        # logger / metrics / visualize / config_loader
├─ scripts/                 # run_demo · download_weights · export_onnx · build_engine · benchmark · train
├─ tests/                   # 단위 테스트 (카메라·모델 없이 실행 가능)
├─ demo_ui/index.html       # ★ 예시 키오스크 화면 (기획서 구조 외 추가 — 교체 예정)
└─ docs/TODO.md             # 작업 분해 및 미확정 항목
```

★ 표시는 **회사 키오스크 프로그램을 받으면 교체/제거되는 부분**이다 (기획서 1.2 제외 범위, 9장 №7·№8).

## Jetson Orin Nano 설치

> 상세 단계별 가이드: **[docs/JETSON_SETUP.md](docs/JETSON_SETUP.md)** (클론 인증·전용 PyTorch 휠·문제 해결 포함)
> 개발 PC(VS Code) 셋업: **[docs/CODE_SETUP.md](docs/CODE_SETUP.md)**

```bash
# 1) JetPack 6.x 기준. PyTorch는 NVIDIA 제공 Jetson용 휠 사용
#    https://developer.nvidia.com/embedded/downloads (README 하단 참고)
pip install -r requirements.txt

# 2) 사전학습 가중치 다운로드 (약 12MB, 학습 불필요)
python scripts/download_weights.py

# 3) 동작 확인 (.pt 그대로 — 첫 구동 확인용)
python scripts/run_demo.py
#    브라우저: http://<jetson-ip>:5000

# 4) TensorRT FP16 엔진 빌드 (약 5~10분, Jetson에서 1회만)
python scripts/build_engine.py
#    이후 configs/config.yaml 에서 model.backend: engine 으로 변경 → 30 FPS 목표

# 5) 추론 단독 FPS 측정 (기획서 6.1)
python scripts/benchmark.py
```

개발 PC(macOS/Windows)에서는 `model.backend: torch` 그대로 두면 CPU/GPU로 동작한다.

## 실행 모드

| 명령 | 용도 |
|---|---|
| `python scripts/run_demo.py` | 파이프라인 + 예시 UI (시연용) |
| `python scripts/run_demo.py --headless` | 파이프라인만 — 이벤트는 `event_output` 설정대로 전송 (회사 프로그램 연동 시) |
| `python -m unittest discover tests -v` | 판정 로직 단위 테스트 |

## 회사 프로그램(UI) 연동 방법

1. `configs/config.yaml`의 `event_output.mode`를 선택 (현재 console/udp 예시 구현)
2. UDP 모드: 이벤트가 JSON `{"class_name": "swipe_left", "conf": 0.87, "ts_sec": ...}`으로 전송됨
3. 회사 규격(소켓/시리얼/공유메모리)이 확정되면 `src/pipeline/event_sender.py`에
   Sender 클래스 1개를 추가하고 mode로 선택 — 파이프라인 코드는 수정 불필요
4. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

## SD카드 128GB 운영 원칙

- **학습 금지**: `scripts/train.py`는 Jetson(aarch64)에서 실행을 차단한다. 학습은 개발 PC에서.
- 데이터셋(`data/raw` 등)은 Jetson에 두지 않는다 — 수집·라벨링은 PC에서.
- Jetson에 필요한 파일은 코드 + `.engine`(약 10~30MB) 뿐이다.

## 라이선스 주의 (기획서 9장 №9)

HaGRID 데이터셋·모델은 **CC BY-SA 4.0 변형** 라이선스다.
상용 키오스크 탑재 전 반드시 회사 법무/담당자 검토를 거칠 것.

## 참고 링크

- HaGRID (모델·데이터셋): https://github.com/hukenovs/hagrid
- 기반 캡스톤 코드: https://github.com/kakabab12/My_project (2026 1학기 젯슨나노 객체인식 분류+르로봇)
- Jetson용 PyTorch 휠: https://forums.developer.nvidia.com/t/pytorch-for-jetson
