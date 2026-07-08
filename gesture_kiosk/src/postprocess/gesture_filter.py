"""postprocess 모듈 — 연속 프레임 판정으로 오인식을 걸러 이벤트를 확정한다 (기획서 3.2).

판정 규칙(모든 수치는 config에서 읽는다):
- 정적 제스처(point·palm_stop·thumbs_up): 같은 제스처가 stable_frame_count 프레임
  연속 + 거의 제자리(max_static_move_ratio 이내)일 때 확정
- 스와이프(swipe_left/right): source_gesture(손바닥)의 중심 x좌표가 window_sec 안에
  프레임 폭의 min_dist_ratio 이상 이동하면 확정. 방향은 거울 모드 기준
  화면 이동 방향 = 사용자 손 이동 방향
- 이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지)
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"


@dataclass
class GestureEvent:
    """확정된 제스처 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float


class GestureFilter:
    def __init__(self, config, frame_width_px, clock=time.monotonic):
        detect = config["detect"]
        self._class_map = config["model"]["class_map"]
        self._stable_frame_count = detect["stable_frame_count"]
        self._cooldown_sec = detect["cooldown_sec"]
        self._max_static_move_ratio = detect["max_static_move_ratio"]
        self._swipe_source = detect["swipe"]["source_gesture"]
        self._swipe_window_sec = detect["swipe"]["window_sec"]
        self._swipe_min_dist_ratio = detect["swipe"]["min_dist_ratio"]

        self._frame_width_px = frame_width_px
        self._clock = clock

        self._stable_class_name = None
        self._stable_count = 0
        self._stable_start_cx_ratio = None
        self._track = deque()  # (ts_sec, cx_ratio) — 스와이프 궤적
        self._last_event_ts_sec = None

    def filter_detections(self, detections):
        """detections -> gesture_event | None (기획서 4.6 계약)."""
        now_sec = self._clock()

        if self._is_in_cooldown(now_sec):
            return None

        best = self._pick_best(detections)
        if best is None:
            self._reset_stable()
            return None

        detection, gesture = best
        cx_ratio = self._to_cx_ratio(detection.bbox)

        if gesture == self._swipe_source:
            swipe_event = self._check_swipe(gesture, detection, cx_ratio, now_sec)
            if swipe_event is not None:
                return swipe_event
        else:
            self._track.clear()

        return self._check_static(gesture, detection, cx_ratio, now_sec)

    # ----- 내부 판정 로직 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _pick_best(self, detections):
        """class_map에 등록된 검출 중 conf 최고 1건을 (Detection, 표준 제스처)로 돌려준다."""
        best_det = None
        best_gesture = None
        for det in detections:
            gesture = self._class_map.get(det.class_name)
            if gesture is None:
                continue
            if best_det is None or det.conf > best_det.conf:
                best_det = det
                best_gesture = gesture
        if best_det is None:
            return None
        return best_det, best_gesture

    def _to_cx_ratio(self, bbox):
        x1, _, x2, _ = bbox
        return ((x1 + x2) / 2.0) / self._frame_width_px

    def _check_swipe(self, gesture, detection, cx_ratio, now_sec):
        self._track.append((now_sec, cx_ratio))
        while self._track and now_sec - self._track[0][0] > self._swipe_window_sec:
            self._track.popleft()

        start_cx_ratio = self._track[0][1]
        move_ratio = cx_ratio - start_cx_ratio
        if abs(move_ratio) < self._swipe_min_dist_ratio:
            return None

        class_name = SWIPE_RIGHT if move_ratio > 0 else SWIPE_LEFT
        return self._confirm_event(class_name, detection.conf, now_sec)

    def _check_static(self, gesture, detection, cx_ratio, now_sec):
        is_same_run = gesture == self._stable_class_name and (
            abs(cx_ratio - self._stable_start_cx_ratio) <= self._max_static_move_ratio
        )
        if is_same_run:
            self._stable_count += 1
        else:
            self._stable_class_name = gesture
            self._stable_count = 1
            self._stable_start_cx_ratio = cx_ratio

        if self._stable_count < self._stable_frame_count:
            return None
        return self._confirm_event(gesture, detection.conf, now_sec)

    def _confirm_event(self, class_name, conf, now_sec):
        self._last_event_ts_sec = now_sec
        self._reset_stable()
        self._track.clear()
        event = GestureEvent(class_name=class_name, conf=conf, ts_sec=now_sec)
        logger.info("gesture_event: %s (conf=%.2f)", event.class_name, event.conf)
        return event

    def _reset_stable(self):
        self._stable_class_name = None
        self._stable_count = 0
        self._stable_start_cx_ratio = None
