"""inference 모듈 — 사람 포즈(YOLO pose)를 추론해 얼굴·손목 키포인트를 얻는다.

사용자 잠금(person_lock)의 입력을 만든다: 프레임 안의 모든 사람에 대해
머리(코·눈·귀)와 손목(왼/오른) 키포인트를 돌려준다. 제스처 검출기(trt_engine)와
같은 backend 규칙(torch .pt / TensorRT .engine)을 따른다.

키포인트 번호는 COCO 17 규격이다 (0=코, 1·2=눈, 3·4=귀, 9=왼손목, 10=오른손목).
주의: 이 라벨은 "화면에 보이는 사람" 기준의 해부학적 좌/우다. 거울 반전된
프레임에서는 사용자의 실제 좌/우와 반대가 되며, 그 보정은 person_lock이 담당한다.
"""
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger("inference")

WARMUP_SIZE_PX = 640

# COCO 17 키포인트 인덱스 (ultralytics pose 출력 순서)
KPT_NOSE = 0
KPT_HEAD_INDICES = (0, 1, 2, 3, 4)  # 코·양눈·양귀 — 얼굴 영역 추정에 사용
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10


@dataclass
class PersonPose:
    """사람 1명의 포즈 추정 결과 (기획서 4.6 공통 데이터 구조 스타일)."""

    bbox: tuple                 # (x1, y1, x2, y2) 픽셀 좌표
    conf: float
    keypoints: np.ndarray       # shape (17, 3) — (x_px, y_px, conf)
    head_points: list = field(default_factory=list)  # 신뢰도 통과한 머리 키포인트 [(x, y)]

    def wrist(self, index, min_conf):
        """키포인트 신뢰도가 통과하면 (x_px, y_px), 아니면 None."""
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


def _resolve_device(backend):
    # .engine은 GPU 전용. torch 백엔드는 CUDA(윈도우/리눅스) → MPS(맥) → CPU 순 자동 선택
    if backend == "engine":
        return 0
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PoseEstimator:
    """YOLO pose 추정기. infer(frame) -> list[PersonPose]."""

    def __init__(self, config):
        backend = config["model"]["backend"]
        if backend == "engine":
            model_path = config["model"]["pose_engine_path"]
        else:
            model_path = config["model"]["pose_weights_path"]

        self._model = YOLO(model_path, task="pose")
        self._device = _resolve_device(backend)
        self._input_size_px = config["model"]["input_size_px"]
        self._kpt_conf_threshold = config["person_lock"]["kpt_conf_threshold"]

        dummy = np.zeros((WARMUP_SIZE_PX, WARMUP_SIZE_PX, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self._device)
        logger.info("포즈 모델 로딩 완료: %s (backend=%s, device=%s)", model_path, backend, self._device)

    def infer(self, frame):
        """프레임에서 사람 포즈를 추정한다."""
        results = self._model.predict(
            frame,
            imgsz=self._input_size_px,
            verbose=False,
            device=self._device,
        )
        persons = []
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return persons

        for box, kpts in zip(result.boxes, result.keypoints):
            keypoints = kpts.data[0].cpu().numpy()  # (17, 3)
            head_points = [
                (float(keypoints[i][0]), float(keypoints[i][1]))
                for i in KPT_HEAD_INDICES
                if keypoints[i][2] >= self._kpt_conf_threshold
            ]
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            persons.append(
                PersonPose(
                    bbox=(x1, y1, x2, y2),
                    conf=float(box.conf),
                    keypoints=keypoints,
                    head_points=head_points,
                )
            )
        return persons
