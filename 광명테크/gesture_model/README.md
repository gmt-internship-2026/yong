# 제스처 인식 모델 — 사용 설명서

키오스크 조작용 손 제스처 인식 파이프라인. 웹캠 → MediaPipe Hand Landmarker로 손
21개 랜드마크(+포즈로 손 든 높이) 추출 → 손모양/손 든 상태 판정 → 동작 판정 FSM →
동작 이벤트(`next_item` 등) 확정. **광명테크 공식 "시각장애인 키오스크 제스처
표준안"(8개 동작, 동작4 스크롤 제외 7개 구현) 기준으로 설계됨.**

## 지금 상태 (진행 중)

전체 구조는 두 층으로 나뉜다:
- **① 손모양/손 든 상태가 뭔지 판정하는 층** — 손모양(fist/palm/ok/one/like/none)은
  `detector_mediapipe.py`, 손 든 상태(어깨보다 위/머리보다 위)는 `person_lock.raised_hands()`.
  아래 표는 ①의 상태.
- **② 그걸 시간 흐름으로 묶어 next_item/select/go_home 등 8개 표준 동작으로 확정하는
  층** — `gesture_filter.py`. 유닛테스트 59건 통과 + 실제 웹캠으로 `cancel`/`next_item`/
  `sos_call` 발화 확인됨(재발화 간격까지 설계대로). `prev_item`/`select`/`pause_voice`/
  `go_home`은 같은 방식의 트래커라 테스트는 통과했지만 아직 실제 웹캠으로 직접 확인은 안 함.

①번 층(손모양 판정)은 현재 **교체 진행 중**이다:

| | 상태 |
|---|---|
| 기하 규칙 판정(`detector_mediapipe.classify_hand_landmarks`) | 완성, 현재 `run_demo.py`가 실제로 쓰는 중 (손가락 폄/굽힘 각도를 계산식으로 즉시 판정, 학습 불필요) |
| 학습된 분류기로 교체 (`hand_pose_classifier.py` + `collect_landmarks.py` + `train_classifier.py`) | **코드는 다 만들어서 합성 데이터로 정상 동작까지 확인함(학습→ONNX export→추론 전부 성공), 아직 실제 손 데이터로 학습은 안 함 — `models/weights/hand_pose_classifier.onnx`가 아직 없음** |

즉 지금 `python scripts/run_demo.py`를 돌리면 여전히 규칙 기반으로 동작한다.
"우리가 학습시킨 모델"로 바뀌려면: **① 실제 데이터 녹화(`collect_landmarks.py`) →
② 학습(`train_classifier.py`, `.onnx` 생성) → ③ `detector_mediapipe.py`가 규칙
대신 이 onnx를 쓰도록 배선** 이 세 단계가 남아있다. 아래 4장·5장 참고.

## 왜 이렇게 바뀌었는지

이전 버전은 MediaPipe Holistic으로 팔+양손 랜드마크를 뽑아 커스텀 GRU를 학습시키는
방식이었다. 팀의 실제 프로덕션 프로젝트(`GMtech_project/gesture_kiosk`, 광명테크
"제스처 인식 배리어프리 민원발급기")를 참고해 그 프로젝트가 실제로 쓰는 방식으로
전면 재작성했다:

- **학습 대신 기하 규칙.** 손가락이 펴졌는지(TIP이 PIP보다 손목에서 충분히 먼지)와
  엄지-검지 핀치 거리만으로 fist/palm/ok/one/like를 계산식으로 즉시 판정한다.
  `collect_data.py`로 사람별 데이터를 모으고 `train.py`로 학습하던 단계가 통째로 없다.
- **MediaPipe Hand Landmarker(Apache-2.0), 학습 없이 즉시 배포 가능.** 참고 프로젝트에는
  HaGRID 데이터셋으로 학습된 YOLOv10 ONNX 모델(`YOLOv10n_gestures.onnx`)도 있었지만,
  그 프로젝트 자체 문서에 "AGPL-3.0 라이선스 리스크로 상업 납품 금지"라고 명시되어 있고
  실제로 이미 폐기된 경로였다 — 그래서 이식하지 않았다.
- **사용자 잠금(person_lock) 추가.** rtmlib(RTMPose)로 사람 포즈를 추정해, 카메라
  오토포커스가 맞은(가장 선명하고 큰 얼굴) 사람에게 잠그고 그 사람 손만 인식한다.
  다른 사람이 옆에서 손을 흔들어도 반응하지 않는다.
- **동작 판정을 FSM으로 명확히 분리.** "주먹 쥐었다 펴기", "N프레임 정적 유지",
  "10초 유지" 같은 판정 로직이 `gesture_filter.py` 하나에 모여있고, 전부 config로
  튜닝 가능하며 카메라 없이 도는 단위 테스트로 검증되어 있다.

## 제스처 목록 (광명테크 공식 표준안 기준)

스마트폰 VoiceOver/TalkBack 조작 습관을 손 제스처로 옮긴 것. 동작4(화면 스크롤)는
표준안 자체가 "미정"이라 구현하지 않았다.

| No | 기능 | 판정 | 이벤트 | 확인 상태 |
|---|---|---|---|---|
| 1 | 다음 항목 이동 | 오른손 주먹쥐기 유지 | `next_item` | ✅ 실제 웹캠 확인 |
| 2 | 이전 항목 이동 | 왼손 주먹쥐기 유지 | `prev_item` | 테스트만 |
| 3 | 선택/실행 | OK 사인 유지 | `select` | 테스트만 |
| 4 | 화면 스크롤(상/하) | 표준안 미정 | — | 미구현 |
| 5 | 음성안내 일시정지 | 손바닥 카메라로 펴기 유지(한 손) | `pause_voice` | 테스트만 |
| 6 | 뒤로가기/취소 | 왼쪽 손'만' 들기(손모양 무관) | `cancel` | ✅ 실제 웹캠 확인 |
| 7 | 홈 화면 이동 | 양손 들기(어깨~머리 사이) | `go_home` | 테스트만 |
| 8 | 도움말/SOS 호출 | 양손을 머리보다 높이 들고 3초 이상 | `sos_call` | ✅ 실제 웹캠 확인 |

- **손모양 기반**(1·2·3·5): `detector_mediapipe.py`가 판정한 fist/ok/palm을
  `gesture_filter.py`가 N프레임 안정 유지로 확정.
- **손 든 높이 기반**(6·7·8): 손모양과 무관하게 `person_lock.raised_hands()`가
  포즈 추정(어깨·코 키포인트)으로 판정. 6·7은 "어깨보다 위", 8은 "머리보다 위"로
  구간을 나눠 서로 겹치지 않게 했다 — 겹치면 hold_sec이 짧은 7이 먼저 확정되며
  8의 타이머를 계속 리셋시켜 8이 영영 안 나오는 문제가 있었음(수정됨).
- 우선순위(동시에 여러 조건 충족 시): `sos_call` > `go_home` > `cancel` > 나머지.
  `cancel`은 "왼손만" 들렸을 때만 발동해 `go_home`(양손)과 자연히 구분된다.
- 좌/우 판정은 MediaPipe handedness(어느 손인지) 기준이라, **한쪽 팔이 없는 사용자도
  인식**된다.
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손만 인식하고 다른 사람 손은 무시한다.
- 레거시 제스처(`point`/`palm_stop`/`swipe_left`/`swipe_right`/`thumbs_up`, 이전
  자체 설계 버전)는 `configs/config.yaml`의 `gestures.legacy.enabled: true`로
  켜야 판정된다 (기본 꺼짐).
- **세부 임계값·우선순위는 잠정 설계다.** 표준안 문서에 정확히 명시 안 된 부분
  (예: 6·7·8 hold_sec 값, 동시 충족 시 우선순위)은 실측 후 팀 확인이 필요하다.

## 폴더 구조

```
gesture_model/
├── configs/config.yaml       # 모든 튜닝값의 단일 출처 — 수정은 여기서만
├── models/weights/
│   └── hand_landmarker.task   # MediaPipe 사전학습 모델 (Apache-2.0, 약 7.8MB)
├── scripts/
│   ├── run_demo.py             # [실행] 실시간 웹캠 데모 / 키오스크 연동 지점
│   ├── collect_landmarks.py    # [실행] 손모양 학습 데이터 녹화 (data/<label>/*.npy)
│   └── train_classifier.py     # [실행] 손모양 분류기 학습 -> ONNX export
├── src/
│   ├── capture/camera_stream.py       # 카메라 캡처 스레드
│   ├── inference/
│   │   ├── detector.py                # Detection 공통 구조 + 검출기 생성
│   │   ├── detector_mediapipe.py      # 손 랜드마크 -> 손모양 판정 (현재: 기하 규칙)
│   │   ├── hand_landmark_extractor.py # MediaPipe HandLandmarker 래퍼 (검출기·녹화 공유)
│   │   ├── hand_pose_classifier.py    # 정규화 함수 + 학습된 ONNX 분류기 추론 래퍼
│   │   ├── pose_estimator.py          # rtmlib RTMPose (person_lock용)
│   │   └── preprocessor.py            # 거울 반전
│   ├── postprocess/
│   │   ├── gesture_filter.py          # 동작 판정 FSM
│   │   └── person_lock.py             # 사용자 잠금 + 손 좌/우 귀속
│   ├── pipeline/
│   │   ├── event_sender.py            # 이벤트 출력 (console/udp)
│   │   └── realtime_loop.py           # 파이프라인 조립 (run_pipeline)
│   └── utils/                         # config_loader / logger / metrics / visualize
├── tests/                              # 카메라·모델 없이 도는 단위 테스트
├── requirements.txt
└── configs/config.yaml
```

**직접 실행하는 파일은 1개**: `scripts/run_demo.py`. 나머지는 이 스크립트가
가져다 쓰는 내부 모듈이거나 설정 파일이다.

## 0. 설치

`gesture_model/` 폴더에서:

```
pip install -r requirements.txt
```

`mediapipe`, `opencv-python`, `numpy`, `pyyaml`, `onnxruntime`, `rtmlib`이 설치된다.
`models/weights/hand_landmarker.task`는 이미 받아뒀으므로 추가로 할 일 없다.
`torch`/`onnxscript`는 `scripts/train_classifier.py`(4·5장) 전용이라 `run_demo.py`만
쓸 거면 없어도 되지만, requirements.txt에 같이 들어있어 `pip install`이면 전부 받아진다.

**중요 — 첫 실행 시 자동 다운로드**: `person_lock`이 켜져 있으면(기본값) `rtmlib`이
RTMPose 포즈 모델(약 40MB)을 처음 한 번 `~/.cache/rtmlib`에 자동으로 받는다.
인터넷 연결이 필요하고, 처음 실행할 때 몇 초~수십 초 더 걸린다. 이후에는 캐시를 쓴다.

---

## 1. `configs/config.yaml` — 설정

직접 실행하는 파일이 아니라, 다른 모든 모듈이 참조하는 설정 파일. 주요 항목:

| 키 | 기본값 | 의미 |
|---|---|---|
| `camera.device_id` | 0 | 웹캠 장치 번호 |
| `camera.windows_backend` | auto | 카메라가 안 열리거나 느리면 dshow/msmf로 변경 |
| `model.mediapipe.finger_extended_ratio` | 1.15 | 이 배율 이상 손목에서 멀면 "손가락 폄" — 손 크기가 다양한 사용자를 감안해 조절 가능 |
| `model.mediapipe.ok_pinch_ratio` | 0.35 | 엄지-검지 거리/손크기가 이 미만이면 OK로 판정 |
| `person_lock.enabled` | true | 끄면 화면 좌/우 절반 기준으로 단순 귀속 (rtmlib 없이도 동작 확인 가능) |
| `gestures.next_prev.stable_frame_count` | 5 | 주먹을 이 프레임 연속 유지해야 next_item/prev_item 확정 |
| `gestures.select.stable_frame_count` | 5 | OK 사인을 이 프레임 연속 유지해야 select 확정 |
| `gestures.pause_voice.stable_frame_count` | 5 | 손바닥을 이 프레임 연속 유지해야 pause_voice 확정 |
| `gestures.cancel.hold_sec` | 0.5 | 왼손만 든 상태를 이 시간 유지해야 cancel 확정 |
| `gestures.go_home.hold_sec` | 0.5 | 양손 든 상태를 이 시간 유지해야 go_home 확정 |
| `gestures.sos_call.hold_sec` | 3.0 | 양손을 머리보다 높이 든 상태를 이 시간 유지해야 sos_call 확정 |
| `person_lock.kpt_conf_threshold` | 0.3 | 어깨·코 키포인트 최소 신뢰도 — 손 든 판정(6·7·8)에도 쓰임 |
| `detect.cooldown_sec` | 1.0 | 이벤트 확정 직후 재발화 방지 시간 |

**접근성 관련**: 손가락/손 형태가 표준과 다른 사용자는 `finger_extended_ratio`나
`ok_pinch_ratio`를 조절해서 대응할 수 있다 — 재학습이 필요 없다는 게 기하 규칙
방식의 장점이다. 다만 MediaPipe의 손 검출 자체가 표준적인 손가락 5개 형태 위주로
학습되어 있어서, 손 형태가 많이 다르면 애초에 랜드마크 추출 단계에서 잘 안 잡힐 수
있다 — 이건 이 프로젝트가 못 건드리는 MediaPipe 자체의 한계다. 제스처 자체를
손가락 모양보다 팔의 이동(주먹→펴기, 좌우 위치)이나 손 전체 자세(펴짐/OK) 위주로
설계해 둔 것도 이 때문이다.

---

## 2. `scripts/run_demo.py` — 실시간 데모 / 키오스크 연동

```
python scripts/run_demo.py
```

웹캠 창에 검출 박스, 사용자 잠금 얼굴 박스, 손목 위치(L/R), 양손바닥 유지 진행 바,
FPS, 최근 확정 이벤트가 표시된다. 제스처가 확정되면 콘솔에
`>>> GESTURE: move_left (0.9x)` 형태로 출력된다. `q`로 종료.

**팀원 키오스크 프레임워크와 연동하는 지점**은 파일 상단의 이 함수 하나뿐:

```python
def on_gesture_detected(label: str, confidence: float):
    print(f">>> GESTURE: {label} ({confidence:.2f})")
```

이 함수 내용을 실제 키오스크 동작(화면 전환 함수 호출, 이벤트 큐에 넣기 등)으로
바꿔 끼우면 된다. `label`은 `move_left`/`move_right`/`select`/`go_home` 등
확정된 이벤트 이름 그대로 들어온다.

또는 `configs/config.yaml`의 `event_output.mode`를 `udp`로 바꾸면 같은 이벤트를
JSON으로 UDP 전송한다 (`class_name`/`conf`/`ts_sec`/`hand_side`).

---

## 3. 테스트

카메라·모델 없이 도는 순수 로직 테스트 (판정 규칙·잠금·FSM 검증):

```
python -m unittest discover tests -v
```

- `test_mediapipe_classify.py`: 손 랜드마크 -> 제스처 판정 규칙
- `test_person_lock.py`: 사용자 잠금·거울 좌우 보정·손 귀속
- `test_gesture_filter.py`: 이동/선택/양손유지/레거시/쿨다운 FSM

---

## 4. `scripts/collect_landmarks.py` — 손모양 학습 데이터 녹화

우리가 직접 학습시킨 분류기를 만드는 첫 단계. 규칙 기반 판정을 학습된 모델로
바꾸고 싶을 때만 필요하고, 지금 당장 `run_demo.py`를 쓰는 데는 필요 없다.

```
python scripts/collect_landmarks.py
```

1. 촬영 대상 이름 입력 (예: `kim`) — 여러 명이 각자 실행하면 데이터가 다양해져 좋음
2. 화면에 손 랜드마크(초록 점)가 보이면 정상
3. **SPACE**: 2초 카운트다운 후 1초간 녹화 (현재 선택된 라벨로 저장)
4. **n / p**: 라벨 전환 — `fist(주먹) → palm(손바닥) → ok(OK사인) → one(검지만 폄)
   → like(엄지만 폄) → none(그 외 모든 자연스러운 손모양)`
5. **q**: 종료, 저장은 `data/<label>/<이름>_<label>_<번호>_<타임스탬프>.npy`

**양손을 동시에 들고 녹화 가능** — `run_demo.py`(실전)와 똑같이 이 스크립트도 한
프레임에 최대 두 손을 인식한다. 양손으로 같은 모양을 만들면 한 번 SPACE로 샘플이
2개씩 쌓인다 (분류기는 손 하나하나를 독립적으로 판정하므로 왼손/오른손 구분 없이
그냥 각각 별도 샘플로 저장됨).

**라벨당 최소 15~20회** 이상, 매번 위치·각도·거리를 조금씩 바꿔가며 녹화 권장.
**`none`이 제일 중요함** — 5개 제스처 어디에도 안 속하는 손모양(반쯤 쥔 손, 브이
사인, 손 내리는 중 등)을 다양하게 담아야 실전에서 아무 손모양에나 반응하는
오탐을 줄일 수 있다 (다른 라벨보다 더 많이, 30회 이상 권장).

## 5. `scripts/train_classifier.py` — 학습 + ONNX export

`data/`에 녹화가 어느 정도 쌓인 뒤:

```
python scripts/train_classifier.py
python scripts/train_classifier.py --epochs 150 --batch_size 32 --lr 0.001 --val_ratio 0.2
```

손 21랜드마크(픽셀좌표) → 손목 원점·손크기로 정규화한 42차원 벡터 → 은닉 32짜리
소형 MLP(PyTorch) → 6클래스(fist/palm/ok/one/like/none) 분류. 학습 후 마지막에
softmax를 붙인 형태로 `models/weights/hand_pose_classifier.onnx`로 export하고,
`models/weights/hand_pose_label_map.json`에 라벨 목록을 같이 저장한다.

**출력 읽는 법**: 시작할 때 라벨별 프레임 수가 출력됨(30 미만이면 더 녹화 권장) →
`val_acc` 1.0에 가까울수록 좋음 → confusion matrix에서 recall 낮은 라벨은 더
녹화 후 재학습.

학습·export까지는 합성 데이터로 이미 정상 동작을 확인해 뒀다 (기계적으로는
문제 없음) — 실제 정확도는 진짜 손 데이터로 학습해봐야 알 수 있다.

**아직 안 된 것**: 이 스크립트가 만든 `.onnx`를 `detector_mediapipe.py`가 실제로
쓰도록 연결하는 배선. 지금은 학습해도 `run_demo.py` 동작에는 반영되지 않는다
(다음 작업으로 진행 예정).

---

## 트러블슈팅

- **`카메라(device_id=0)를 열 수 없습니다`**: 다른 프로그램이 웹캠을 점유 중인지
  확인. `configs/config.yaml`의 `camera.device_id`를 0→1로 바꿔서 다른 카메라도
  시도. 열리는 데 오래 걸리거나 FPS가 낮으면 `camera.windows_backend`를
  auto→dshow 또는 msmf로 바꿔볼 것.
- **`모델 파일이 없습니다` 류 에러**: `models/weights/hand_landmarker.task`가
  있는지 확인 (약 7.8MB). 없으면
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  에서 받아 같은 경로에 저장.
- **첫 실행이 유독 느림**: `person_lock.enabled: true`면 rtmlib이 포즈 모델을
  처음 한 번 인터넷에서 받는다 (`~/.cache/rtmlib`). 정상.
- **FPS가 낮음(CPU 전용)**: `person_lock.enabled: false`로 끄면 포즈 추정이 빠지고
  화면 좌/우 절반 기준으로 손을 귀속해 가벼워진다 — 단 **손 든 높이로 판정하는
  cancel/go_home/sos_call(6·7·8)은 포즈 추정이 있어야 동작**하므로 꺼두면 이 셋은
  아예 발화하지 않는다(1·2·3·5는 정상 동작). 또는 `model.pose_mode`를
  `lightweight`로 유지(기본값).
- **가만히 있어도 next_item/select가 자꾸 잡힘(오탐)**: `detect.conf_threshold`를
  올리거나 `gestures.select.stable_frame_count` / `gestures.next_prev.stable_frame_count`를
  올려서 재시도.
- **왼손/오른손 판정이 실제와 반대로 뜸**: `model.mediapipe.flip_handedness`를
  반전 (MediaPipe가 버전에 따라 handedness 라벨을 문서와 반대로 내는 경우가 보고돼 있음).
