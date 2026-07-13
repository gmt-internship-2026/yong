"""gesture_filter 단위 테스트 — 카메라·모델 없이 판정 로직만 검증한다.

광명테크 공식 "시각장애인 키오스크 제스처 표준안"(8개 동작, 동작4 스크롤 제외) 기준.

실행 (프로젝트 루트에서):
    python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import HandObservation


class FakeClock:
    def __init__(self):
        self.now_sec = 1000.0

    def __call__(self):
        return self.now_sec

    def tick(self, dt_sec):
        self.now_sec += dt_sec


def make_config(legacy_enabled=True):
    return {
        "detect": {"cooldown_sec": 1.0},
        "gestures": {
            "next_prev": {"stable_frame_count": 5, "max_static_move_ratio": 0.08},
            "select": {"stable_frame_count": 5, "max_static_move_ratio": 0.08},
            "pause_voice": {"stable_frame_count": 5, "max_static_move_ratio": 0.08},
            "cancel": {"hold_sec": 0.5, "grace_sec": 0.3},
            "go_home": {"hold_sec": 0.5, "grace_sec": 0.3},
            "sos_call": {"hold_sec": 3.0, "grace_sec": 0.4},
            "legacy": {
                "enabled": legacy_enabled,
                "stable_frame_count": 5,
                "max_static_move_ratio": 0.08,
                "swipe": {
                    "source_gesture": "open_hand",
                    "window_sec": 0.7,
                    "min_dist_ratio": 0.35,
                },
            },
        },
    }


def obs(side, gesture, conf=0.9, cx_ratio=0.5):
    return HandObservation(side=side, gesture=gesture, conf=conf, cx_ratio=cx_ratio)


NONE_RAISED = {"left": False, "right": False}
FRAME_DT_SEC = 1.0 / 30.0  # 30 FPS 가정


class GestureFilterTestBase(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.filter = GestureFilter(make_config(), clock=self.clock)

    def _feed(self, observations, frame_count=1, dt_sec=FRAME_DT_SEC, raised=None, raised_high=None):
        """frame_count 프레임 공급 — 첫 확정 이벤트를 즉시 돌려준다 (없으면 None)."""
        for _ in range(frame_count):
            event = self.filter.filter_observations(observations, raised, raised_high)
            self.clock.tick(dt_sec)
            if event is not None:
                return event
        return None


class NextPrevItemTest(GestureFilterTestBase):
    """동작1·2 — 오른손/왼손 주먹쥐기 유지 = 다음/이전 항목."""

    def test_right_fist_fires_next_item(self):
        event = self._feed([obs("right", "fist")], frame_count=5)
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "next_item")
        self.assertEqual(event.hand_side, "right")

    def test_left_fist_fires_prev_item(self):
        event = self._feed([obs("left", "fist")], frame_count=5)
        self.assertEqual(event.class_name, "prev_item")
        self.assertEqual(event.hand_side, "left")

    def test_short_fist_does_not_fire(self):
        event = self._feed([obs("right", "fist")], frame_count=3)
        self.assertIsNone(event)

    def test_moving_fist_does_not_fire(self):
        event = None
        for i in range(10):
            event = self._feed([obs("right", "fist", cx_ratio=0.1 + i * 0.1)])
        self.assertIsNone(event)

    def test_other_gesture_between_resets_run(self):
        self._feed([obs("right", "fist")], frame_count=3)
        self._feed([obs("right", "ok")])
        event = self._feed([obs("right", "fist")], frame_count=3)
        self.assertIsNone(event)


class SelectGestureTest(GestureFilterTestBase):
    """동작3 — OK 사인 유지 = 선택/실행."""

    def test_ok_stable_frames_fires_select(self):
        self.assertIsNone(self._feed([obs("right", "ok")], frame_count=4))
        event = self._feed([obs("right", "ok")])
        self.assertEqual(event.class_name, "select")

    def test_moving_ok_does_not_fire(self):
        event = None
        for i in range(10):
            event = self._feed([obs("right", "ok", cx_ratio=0.1 + i * 0.1)])
        self.assertIsNone(event)


class PauseVoiceTest(GestureFilterTestBase):
    """동작5 — 손바닥을 카메라로 펴기 유지(한 손) = 음성안내 일시정지."""

    def test_palm_stable_fires_pause_voice(self):
        event = self._feed([obs("right", "open_hand")], frame_count=5)
        self.assertEqual(event.class_name, "pause_voice")
        self.assertEqual(event.hand_side, "right")

    def test_left_hand_palm_also_fires(self):
        event = self._feed([obs("left", "open_hand")], frame_count=5)
        self.assertEqual(event.class_name, "pause_voice")


class CancelGestureTest(GestureFilterTestBase):
    """동작6 — 왼쪽 손'만' 들기(손모양 무관) = 뒤로가기/취소."""

    def test_left_only_raised_fires_cancel(self):
        event = self._feed([], frame_count=20, raised={"left": True, "right": False})
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "cancel")
        self.assertEqual(event.hand_side, "left")

    def test_right_only_raised_does_not_fire_cancel(self):
        event = self._feed([], frame_count=20, raised={"left": False, "right": True})
        self.assertIsNone(event)

    def test_both_raised_does_not_fire_cancel_fires_home_instead(self):
        event = self._feed([], frame_count=20, raised={"left": True, "right": True})
        self.assertEqual(event.class_name, "go_home")


class GoHomeTest(GestureFilterTestBase):
    """동작7 — 양손 들기 = 홈 화면 이동."""

    def test_both_raised_fires_go_home(self):
        event = self._feed([], frame_count=1, raised={"left": True, "right": True})
        self.assertIsNone(event)  # hold_sec(0.5초) 미달
        event = self._feed([], frame_count=20, raised={"left": True, "right": True})
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")

    def test_single_hand_never_fires_home(self):
        event = self._feed([], frame_count=30, raised={"left": True, "right": False})
        self.assertNotEqual(getattr(event, "class_name", None), "go_home")


class SosCallTest(GestureFilterTestBase):
    """동작8 — 양손을 머리보다 높이 들고 3초 이상 유지 = 도움말/SOS 호출."""

    def test_both_high_held_3sec_fires_sos(self):
        event = self._feed([], frame_count=89, raised_high={"left": True, "right": True})
        self.assertIsNone(event)  # 89프레임 ≈ 2.97초 — 아직
        event = self._feed([], frame_count=5, raised_high={"left": True, "right": True})
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "sos_call")

    def test_sos_takes_priority_over_home(self):
        """양손이 raised_high면 raised(어깨 기준)도 자연히 True — sos가 home보다 우선해야 한다."""
        event = self._feed(
            [], frame_count=95,
            raised={"left": True, "right": True}, raised_high={"left": True, "right": True},
        )
        self.assertEqual(event.class_name, "sos_call")

    def test_hold_ratio_progresses(self):
        self._feed([], frame_count=45, raised_high={"left": True, "right": True})  # 1.5초
        self.assertGreater(self.filter.sos_hold_ratio, 0.4)
        self.assertLess(self.filter.sos_hold_ratio, 0.6)

    def test_break_beyond_grace_resets(self):
        self._feed([], frame_count=60, raised_high={"left": True, "right": True})
        self._feed([], frame_count=1, dt_sec=0.5, raised_high=NONE_RAISED)  # grace(0.4초) 초과 공백
        self._feed([], frame_count=1, raised_high={"left": True, "right": True})
        self.assertLess(self.filter.sos_hold_ratio, 0.1)


class LegacyGestureTest(GestureFilterTestBase):
    """레거시 — legacy.enabled 토글로 병행 유지."""

    def test_palm_swipe_right(self):
        event = None
        for frame_idx in range(12):
            cx_ratio = 0.15 + frame_idx * 0.0625
            event = self._feed([obs("right", "open_hand", cx_ratio=cx_ratio)])
            if event is not None:
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "swipe_right")

    def test_point_stable_fires_point(self):
        event = self._feed([obs("right", "point")], frame_count=5)
        self.assertEqual(event.class_name, "point")

    def test_legacy_disabled_silences_legacy_events(self):
        self.filter = GestureFilter(make_config(legacy_enabled=False), clock=self.clock)
        event = self._feed([obs("right", "point")], frame_count=10)
        self.assertIsNone(event)


class CooldownTest(GestureFilterTestBase):
    def test_cooldown_blocks_repeat_event(self):
        self._feed([obs("right", "ok")], frame_count=5)   # select 확정
        event = self._feed([obs("right", "ok")], frame_count=5)  # 쿨다운(1초) 내
        self.assertIsNone(event)
        self.clock.tick(1.0)
        event = self._feed([obs("right", "ok")], frame_count=5)
        self.assertIsNotNone(event)

    def test_cooldown_does_not_freeze_hold_timers(self):
        """쿨다운 중에도 손 들기 타이머는 계속 흘러야 한다(리셋되지 않아야 한다).

        cooldown_sec(1.0) + go_home.hold_sec(0.5)를 커버할 만큼 연속으로 공급하고,
        중간에 클록을 한번에 점프시키지 않는다 — 점프하면 grace_sec을 넘는 공백으로
        오인되어 타이머가 리셋되므로, 실제로는 "안 끊기고 계속 들고 있는" 상황이
        아니게 된다.
        """
        self._feed([obs("right", "ok")], frame_count=5)  # select 확정 -> 쿨다운 시작
        event = self._feed([], frame_count=50, raised={"left": True, "right": True})
        self.assertIsNotNone(event)
        self.assertEqual(event.class_name, "go_home")


class MetricsTest(unittest.TestCase):
    def test_measure_fps(self):
        from src.utils.metrics import measure_fps

        self.assertAlmostEqual(measure_fps(300, 10.0), 30.0)
        self.assertEqual(measure_fps(10, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
