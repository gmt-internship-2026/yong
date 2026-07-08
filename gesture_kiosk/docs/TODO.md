# TODO — 작업 분해 및 진행 상황

작성: 2026-07-08. 기획서(고성모_기획서.docx) 주차 계획·9장 체크리스트와 연동.

## ✅ 완료 (2026-07-08 — 1주차)

- [x] 기획서 2.3 디렉터리 구조 + 4장 코딩 컨벤션 적용한 저장소 골격
- [x] 캡스톤 코드(jetson_USB2.py) 구조 이식 — 단일 USB 카메라·제스처용으로 정리
      (듀얼 카메라·RealSense·아두이노 로봇팔 제어 제거)
- [x] 실시간 파이프라인: 캡처 스레드 → 전처리 → YOLO 추론 → 오인식 방지 필터 → 이벤트
- [x] 무학습 모델 경로 확보: HaGRIDv2 사전학습 YOLOv10n (다운로드 스크립트 포함)
- [x] 스와이프 2종 궤적 판정 + 정적 3종 연속 프레임 판정 + 쿨다운 (config로 튜닝)
- [x] 예시 UI (demo_ui) — 회사 프로그램 자리의 임시 시연 화면
- [x] 이벤트 전송 접점(event_sender) — console/udp 예시 구현
- [x] 판정 로직 단위 테스트 11건
- [x] 변환·벤치마크·학습 스크립트 골격 (export_onnx / build_engine / benchmark / train)

## 🔴 회사 확인 필요 — 멋대로 진행 금지 (기획서 9장 연동)

- [ ] **№7 회사 키오스크 프로그램(UI) 파일 수령** — 수령 후 demo_ui 교체 작업 시작
- [ ] **№7 이벤트 전달 규격 합의** — 소켓/시리얼/공유메모리 중 회사 기존 SW 규격 확인
      → 확정 시 `src/pipeline/event_sender.py`에 Sender 1개 추가로 대응
- [ ] **№8 통합 범위 확정** — 이번 8주가 UI 통합까지인지 이벤트 출력까지인지
- [ ] **№1 제스처 클래스 확정** — 현재 기획서 5.1 초안 5종으로 구현됨. 변경 시 class_map 수정
- [ ] **№5·№6 KPI 측정 기준 합의** — 정확도 85%(이벤트 단위? mAP?) / 30 FPS(엔드투엔드? 추론 단독?)
- [ ] **№9 HaGRID 라이선스(CC BY-SA 4.0 변형) 상용 사용 가능 여부** — 법무 검토
- [ ] **№2 Jetson JetPack 버전 확인** — TensorRT 버전에 따라 build_engine 재실행 필요
- [ ] **№10 코드 저장 위치·백업 정책** — 외부 Git 허용 여부 확인 후 원격 저장소 push

## 🟡 Jetson 반입 시 작업 (3주차 예정)

- [ ] JetPack·PyTorch(Jetson 휠)·ultralytics 설치, `pip install -r requirements.txt`
- [ ] `python scripts/download_weights.py` — 가중치 다운로드
- [ ] `camera.device_id` 확인 (`ls /dev/video*`) 후 config 수정
- [ ] `.pt` 백엔드로 1차 구동 확인 → `python scripts/build_engine.py` → backend: engine 전환
- [ ] `python scripts/benchmark.py` — 추론 단독 FPS 기록 (docs/에 측정 기록 남기기)
- [ ] 30 FPS 미달 시: input_size_px 축소(640→480→416) → INT8 검토 (기획서 8장 R4 순서)

## 🟢 이후 주차 (기획서 7.2 WBS)

- [ ] 4주차: 데이터 수집·라벨링 (PC에서 — Jetson에 데이터 저장 금지)
- [ ] 5주차: `scripts/train.py`로 PC 파인튜닝 → 사전학습 대비 정확도 비교
- [ ] 7주차: 임계값·N값 튜닝 (T4 오탐 시나리오 — config.yaml 값만 조정)
- [ ] 8주차: 6.2 시나리오 T1~T5 수행, `metrics.eval_accuracy`로 수치화 → 성능 검증 보고서

## 메모

- 로봇팔(르로봇/아두이노) 제어는 이번 프로젝트 범위에서 제외 (2026-07-08 결정)
- 카메라는 USB 웹캠 1대 구성 (2026-07-08 결정)
- 예시 UI의 메뉴(커피 등)는 데모용 더미 — 실제 화면은 회사 프로그램이 담당
