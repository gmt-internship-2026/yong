"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter


class FakeDetection:
    """Detection과 같은 필드를 가진 테스트 대역 (ultralytics 임포트 회피)."""

    def __init__(self, class_name, conf=0.9, cx_px=640.0):
        self.class_id = 0
        self.class_name = class_name
        self.conf = conf
        half_w = 50.0
        self.bbox = (cx_px - half_w, 100.0, cx_px + half_w, 300.0)


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config():
    return {
        "model": {
            "class_map": {
                "point": "point",
                "one": "point",
                "palm": "palm_stop",
                "stop": "palm_stop",
                "like": "thumbs_up",
            }
        },
        "detect": {
            "conf_threshold": 0.5,
            "stable_frame_count": 5,
            "cooldown_sec": 1.0,
            "max_static_move_ratio": 0.08,
            "swipe": {
                "source_gesture": "palm_stop",
                "window_sec": 0.7,
                "min_dist_ratio": 0.35,
            },
        },
    }


FRAME_WIDTH_PX = 1280
FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), FRAME_WIDTH_PX, clock=self.clock)

    def _feed(self, detections, frame_count=1):
        event = None
        for _ in range(frame_count):
            event = self.filter.filter_detections(detections)
            self.clock.tick(FRAME_DT_SEC)
        return event

    def test_static_gesture_needs_stable_frames(self):
        det = [FakeDetection("point")]
        self.assertIsNone(self._feed(det, frame_count=4))
        event = self._feed(det)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "point")

    def test_class_map_translates_model_names(self):
        event = self._feed([FakeDetection("like")], frame_count=5)
        self.assertEqual(event.class_name, "thumbs_up")

    def test_unmapped_class_is_ignored(self):
        event = self._feed([FakeDetection("rock")], frame_count=10)
        self.assertIsNone(event)

    def test_cooldown_blocks_repeat_event(self):
        det = [FakeDetection("point")]
        self._feed(det, frame_count=5)  # 이벤트 확정
        event = self._feed(det, frame_count=5)  # 쿨다운(1초) 내 재시도
        self.assertIsNone(event)
        self.clock.tick(1.0)  # 쿨다운 경과
        event = self._feed(det, frame_count=5)
        self.assertIsNotNone(event)

    def test_detection_gap_resets_stable_run(self):
        det = [FakeDetection("point")]
        self._feed(det, frame_count=4)
        self._feed([])  # 검출 공백 -> 리셋
        self.assertIsNone(self._feed(det, frame_count=4))

    def test_moving_hand_does_not_fire_static(self):
        for frame_idx in range(10):
            cx_px = 200.0 + frame_idx * 150.0  # 프레임마다 크게 이동
            event = self._feed([FakeDetection("point", cx_px=min(cx_px, 1200.0))])
        self.assertIsNone(event)

    def test_swipe_right_from_palm_trajectory(self):
        event = None
        for frame_idx in range(12):
            cx_px = 200.0 + frame_idx * 80.0  # 0.4초 동안 880px 이동 (0.69 비율)
            event = self._feed([FakeDetection("palm", cx_px=cx_px)])
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "swipe_right")

    def test_swipe_left_direction(self):
        event = None
        for frame_idx in range(12):
            cx_px = 1100.0 - frame_idx * 80.0
            event = self._feed([FakeDetection("palm", cx_px=cx_px)])
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "swipe_left")

    def test_stationary_palm_fires_palm_stop_not_swipe(self):
        event = self._feed([FakeDetection("palm")], frame_count=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "palm_stop")

    def test_best_conf_detection_wins(self):
        detections = [
            FakeDetection("point", conf=0.6),
            FakeDetection("like", conf=0.95),
        ]
        event = self._feed(detections, frame_count=5)
        self.assertEqual(event.class_name, "thumbs_up")


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
