"""inference 모듈 — 모델을 로드하고 제스처를 검출한다 (기획서 2.2).

backend 설정에 따라 .pt(개발 PC) 또는 TensorRT .engine(Jetson)을 로드한다.
두 형식 모두 ultralytics YOLO 러너로 실행한다 — .engine 경로는 캡스톤
프로젝트(jetson_USB2.py)에서 Jetson Orin Nano 실기기 검증이 끝난 방식이다.
"""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger("inference")

WARMUP_SIZE_PX = 640


@dataclass
class Detection:
    """검출 결과 1건 (기획서 4.6 공통 데이터 구조)."""

    class_id: int
    class_name: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) 좌상단·우하단 픽셀 좌표


def _resolve_device(backend):
    # .engine은 GPU 전용. torch 백엔드는 CUDA가 없으면 CPU로 동작(개발 PC용)
    if backend == "engine":
        return 0
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


class GestureDetector:
    """YOLO 제스처 검출기. infer(input_tensor) -> list[Detection]."""

    def __init__(self, config):
        backend = config["model"]["backend"]
        if backend == "engine":
            model_path = config["model"]["engine_path"]
        else:
            model_path = config["model"]["weights_path"]

        self._model = YOLO(model_path, task="detect")
        self._device = _resolve_device(backend)
        self._conf_threshold = config["detect"]["conf_threshold"]
        self._input_size_px = config["model"]["input_size_px"]

        dummy = np.zeros((WARMUP_SIZE_PX, WARMUP_SIZE_PX, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self._device)
        logger.info("모델 로딩 완료: %s (backend=%s, device=%s)", model_path, backend, self._device)
        # class_map(설정)과 대조할 수 있게 모델의 실제 클래스명을 기록해 둔다
        logger.info("모델 클래스 목록: %s", list(self._model.names.values()))

    def infer(self, input_tensor):
        """전처리된 프레임에서 제스처를 검출한다."""
        results = self._model.predict(
            input_tensor,
            conf=self._conf_threshold,
            imgsz=self._input_size_px,
            verbose=False,
            device=self._device,
        )
        class_names = results[0].names
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_names[class_id],
                    conf=float(box.conf),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
