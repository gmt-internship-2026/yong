"""pipeline 모듈 — 캡처·추론·후처리를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

캡처는 CameraStream 스레드, 추론·판정은 별도 스레드에서 돈다.
PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.capture.camera_stream import CameraStream
from src.inference.preprocessor import Preprocessor
from src.inference.trt_engine import GestureDetector
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_bbox, draw_status

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


class PipelineState:
    """추론 결과·성능 수치를 스레드 안전하게 공유한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self.capture_fps = 0.0
        self.infer_fps = 0.0
        self.last_event = None
        self.event_log = []
        self.is_running = False

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def append_event(self, gesture_event):
        with self._lock:
            self.last_event = gesture_event
            self.event_log.append(gesture_event)
            if len(self.event_log) > EVENT_LOG_MAX_COUNT:
                self.event_log.pop(0)


def run_pipeline(config):
    """파이프라인 전체를 조립해 시작하고 PipelineState를 돌려준다 (기획서 4.6 계약)."""
    state = PipelineState()
    camera = CameraStream(config).start()
    preprocessor = Preprocessor(config)
    detector = GestureDetector(config)

    first_frame = camera.capture_frame()
    frame_width_px = first_frame.shape[1]
    gesture_filter = GestureFilter(config, frame_width_px)
    event_sender = create_event_sender(config)

    state.is_running = True

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        while state.is_running:
            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)
            detections = detector.infer(input_tensor)
            gesture_event = gesture_filter.filter_detections(detections)

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_bbox(input_tensor, detections)
            overlay_event = state.last_event
            if overlay_event is not None and (
                time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
            ):
                overlay_event = None
            annotated = draw_status(annotated, state.infer_fps, overlay_event)
            state.update_frame(annotated)

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
