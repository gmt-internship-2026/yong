"""postprocess 모듈 — 손 관측(HandObservation) + 손 들기 상태를 동작 이벤트로 확정한다.

광명테크 공식 "시각장애인 키오스크 제스처 표준안"(8개 동작) 기준.
스마트폰 VoiceOver/TalkBack 조작 습관을 손 제스처로 옮긴 것 — 동작4(화면 스크롤)는
표준안 자체가 "미정"이라 이 파일도 구현하지 않는다.

| No | 기능 | 판정 | 이벤트 |
|---|---|---|---|
| 1 | 다음 항목 이동 | 오른손 주먹쥐기 유지 | next_item |
| 2 | 이전 항목 이동 | 왼손 주먹쥐기 유지 | prev_item |
| 3 | 선택/실행 | OK 사인 유지 | select |
| 4 | 화면 스크롤(상/하) | 미정 — 미구현 | - |
| 5 | 음성안내 일시정지 | 손바닥 카메라로 펴기 유지(한 손) | pause_voice |
| 6 | 뒤로가기/취소 | 왼쪽 손만 들기 | cancel |
| 7 | 홈 화면 이동 | 양손 들기 | go_home |
| 8 | 도움말/SOS 호출 | 양손을 머리보다 높이 들고 3초+ 유지 | sos_call |

"손 들기"(6·7·8)는 손모양이 아니라 손목이 어깨/머리보다 높은지로 판정한다
(person_lock.raised_hands()) — HandObservation과는 별개 입력으로 filter_observations()에
전달된다.

우선순위(같은 프레임에 여러 조건이 겹칠 때): sos_call > go_home > cancel >
next_item/prev_item > select > pause_voice. cancel은 "왼손만" 들렸을 때만 발동해
go_home(양손)과 자연히 구분된다. 이 우선순위와 세부 임계값(hold_sec 등)은 표준안
문서에 정확히 명시되어 있지 않은 부분이 있어 잠정 설계 — 실측 후 팀 확인 필요.

이벤트 확정 직후 cooldown_sec 동안 모든 입력을 무시한다 (연타 방지).
모든 수치는 config에서 읽는다.
"""
import time
from collections import deque
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("postprocess")

FIST = "fist"
OK = "ok"
PALM = "open_hand"

SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"


@dataclass
class GestureEvent:
    """확정된 동작 이벤트 1건 — 회사 프로그램(키오스크 UI)으로 전달되는 단위."""

    class_name: str
    conf: float
    ts_sec: float
    hand_side: str = None   # next_item/prev_item/cancel처럼 특정 손에 매인 이벤트만 값이 있다
    data: dict = None       # 부가 정보


class _StableTracker:
    """정적 포즈 유지 판정 — 같은 제스처 N프레임 연속 + 거의 제자리."""

    def __init__(self, stable_frame_count, max_move_ratio):
        self._stable_frame_count = stable_frame_count
        self._max_move_ratio = max_move_ratio
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None

    def update(self, gesture, cx_ratio):
        """관측 1건을 반영하고, 유지 확정이면 True."""
        is_same_run = gesture == self._gesture and (
            abs(cx_ratio - self._start_cx_ratio) <= self._max_move_ratio
        )
        if is_same_run:
            self._count += 1
        else:
            self._gesture = gesture
            self._count = 1
            self._start_cx_ratio = cx_ratio
        if self._count >= self._stable_frame_count:
            self.reset()
            return True
        return False

    def reset(self):
        self._gesture = None
        self._count = 0
        self._start_cx_ratio = None


class _HoldTracker:
    """조건(bool)이 hold_sec 이상 지속되면 확정 — 손 들기(cancel/go_home/sos_call)용.

    grace_sec 이내의 짧은 끊김(포즈 검출 흔들림)은 유지로 봐준다.
    """

    def __init__(self, hold_sec, grace_sec):
        self._hold_sec = hold_sec
        self._grace_sec = grace_sec
        self._start_sec = None
        self._last_seen_sec = None
        self.hold_ratio = 0.0  # 시각화용 (0.0~1.0)

    def update(self, is_active, now_sec):
        if is_active:
            is_gap_too_long = self._start_sec is not None and (
                now_sec - self._last_seen_sec > self._grace_sec
            )
            if self._start_sec is None or is_gap_too_long:
                self._start_sec = now_sec
            self._last_seen_sec = now_sec
        elif self._start_sec is not None and now_sec - self._last_seen_sec > self._grace_sec:
            self.reset()

        if self._start_sec is None:
            self.hold_ratio = 0.0
            return False
        held_sec = now_sec - self._start_sec
        self.hold_ratio = min(held_sec / self._hold_sec, 1.0)
        return held_sec >= self._hold_sec

    def reset(self):
        self._start_sec = None
        self._last_seen_sec = None
        self.hold_ratio = 0.0


class GestureFilter:
    def __init__(self, config, clock=time.monotonic):
        gestures = config["gestures"]
        self._cooldown_sec = config["detect"]["cooldown_sec"]
        self._clock = clock

        next_prev = gestures["next_prev"]
        self._next_tracker = _StableTracker(next_prev["stable_frame_count"], next_prev["max_static_move_ratio"])
        self._prev_tracker = _StableTracker(next_prev["stable_frame_count"], next_prev["max_static_move_ratio"])

        select = gestures["select"]
        self._select_tracker = _StableTracker(select["stable_frame_count"], select["max_static_move_ratio"])

        pause_voice = gestures["pause_voice"]
        self._pause_tracker = _StableTracker(pause_voice["stable_frame_count"], pause_voice["max_static_move_ratio"])

        cancel = gestures["cancel"]
        self._cancel_tracker = _HoldTracker(cancel["hold_sec"], cancel["grace_sec"])

        go_home = gestures["go_home"]
        self._home_tracker = _HoldTracker(go_home["hold_sec"], go_home["grace_sec"])

        sos = gestures["sos_call"]
        self._sos_tracker = _HoldTracker(sos["hold_sec"], sos["grace_sec"])

        legacy = gestures["legacy"]
        self._legacy_enabled = legacy["enabled"]
        self._legacy_static_tracker = _StableTracker(
            legacy["stable_frame_count"], legacy["max_static_move_ratio"]
        )
        self._legacy_swipe_source = legacy["swipe"]["source_gesture"]
        self._legacy_swipe_window_sec = legacy["swipe"]["window_sec"]
        self._legacy_swipe_min_dist_ratio = legacy["swipe"]["min_dist_ratio"]
        self._legacy_track = deque()
        self._legacy_static_events = {"point": "point", PALM: "palm_stop", "thumbs_up": "thumbs_up"}

        self._last_event_ts_sec = None

    @property
    def go_home_hold_ratio(self):
        """시각화용 — 홈 판정 진행률 (0.0~1.0)."""
        return self._home_tracker.hold_ratio

    @property
    def sos_hold_ratio(self):
        """시각화용 — SOS 판정 진행률 (0.0~1.0)."""
        return self._sos_tracker.hold_ratio

    def filter_observations(self, observations, raised=None, raised_high=None):
        """observations(손모양) + raised/raised_high(손 들기, person_lock.raised_hands())
        -> gesture_event | None.

        raised/raised_high 형식: {"left": bool, "right": bool}. None이면 전부 False로 본다
        (person_lock이 꺼져 있거나 포즈 추정이 없는 환경 — 손모양 기반 동작만 동작).
        """
        raised = raised or {"left": False, "right": False}
        raised_high = raised_high or {"left": False, "right": False}
        now_sec = self._clock()

        if self._is_in_cooldown(now_sec):
            self._update_hold_trackers(raised, raised_high, now_sec)
            return None

        event = self._check_hold_gestures(raised, raised_high, now_sec)
        if event is not None:
            return event

        # 트래커는 손모양과 무관하게 항상 update()를 호출해야 한다 — 그래야 다른 손모양이
        # 끼어들었을 때 _StableTracker가 (gesture != self._gesture로 인식해) 카운트를 리셋한다.
        # 확정은 "표적 손모양으로 안정됐을 때"만 한다.
        for obs in observations:
            if obs.side == "right":
                fired = self._next_tracker.update(obs.gesture, obs.cx_ratio)
                if fired and obs.gesture == FIST:
                    return self._confirm("next_item", obs.conf, now_sec, hand_side="right")
            if obs.side == "left":
                fired = self._prev_tracker.update(obs.gesture, obs.cx_ratio)
                if fired and obs.gesture == FIST:
                    return self._confirm("prev_item", obs.conf, now_sec, hand_side="left")

        for obs in observations:
            fired = self._select_tracker.update(obs.gesture, obs.cx_ratio)
            if fired and obs.gesture == OK:
                return self._confirm("select", obs.conf, now_sec)

        for obs in observations:
            fired = self._pause_tracker.update(obs.gesture, obs.cx_ratio)
            if fired and obs.gesture == PALM:
                return self._confirm("pause_voice", obs.conf, now_sec, hand_side=obs.side)

        if self._legacy_enabled:
            return self._check_legacy(observations, now_sec)
        return None

    # ----- 손 들기 판정 (6·7·8) -----

    def _check_hold_gestures(self, raised, raised_high, now_sec):
        """우선순위: sos_call(양손 높이 3초+) > go_home(양손) > cancel(왼손만).

        go_home은 "머리보다 높이"(=raised_high) 상태는 명시적으로 제외한다 — 안 그러면
        양손을 계속 높이 들고 있을 때 hold_sec이 훨씬 짧은 go_home이 0.5초마다 먼저
        확정되면서 그때마다 sos_call 타이머를 리셋시켜, SOS가 영영 확정될 수 없게 된다.
        즉 "어깨~머리 사이"는 go_home, "머리 위"는 sos_call 전용 구간이다.
        """
        is_both_high = raised_high["left"] and raised_high["right"]
        is_sos = self._sos_tracker.update(is_both_high, now_sec)

        is_both_raised = raised["left"] and raised["right"] and not is_both_high
        is_home = self._home_tracker.update(is_both_raised, now_sec)

        is_left_only_raised = raised["left"] and not raised["right"]
        is_cancel = self._cancel_tracker.update(is_left_only_raised, now_sec)

        if is_sos:
            return self._confirm("sos_call", 1.0, now_sec)
        if is_home:
            return self._confirm("go_home", 1.0, now_sec)
        if is_cancel:
            return self._confirm("cancel", 1.0, now_sec, hand_side="left")
        return None

    def _update_hold_trackers(self, raised, raised_high, now_sec):
        """쿨다운 중에도 타이머는 그대로 흐르게(끊기지 않게) 갱신만 한다."""
        is_both_high = raised_high["left"] and raised_high["right"]
        self._sos_tracker.update(is_both_high, now_sec)
        self._home_tracker.update(raised["left"] and raised["right"] and not is_both_high, now_sec)
        self._cancel_tracker.update(raised["left"] and not raised["right"], now_sec)

    # ----- 레거시(기존 스와이프/정지 등) -----

    def _check_legacy(self, observations, now_sec):
        best = None
        for obs in observations:
            if obs.gesture not in self._legacy_static_events:
                continue
            if best is None or obs.conf > best.conf:
                best = obs
        if best is None:
            self._legacy_static_tracker.reset()
            return None

        if best.gesture == self._legacy_swipe_source:
            swipe_event = self._check_legacy_swipe(best, now_sec)
            if swipe_event is not None:
                return swipe_event
        else:
            self._legacy_track.clear()

        if self._legacy_static_tracker.update(best.gesture, best.cx_ratio):
            return self._confirm(self._legacy_static_events[best.gesture], best.conf, now_sec)
        return None

    def _check_legacy_swipe(self, obs, now_sec):
        self._legacy_track.append((now_sec, obs.cx_ratio))
        while self._legacy_track and (
            now_sec - self._legacy_track[0][0] > self._legacy_swipe_window_sec
        ):
            self._legacy_track.popleft()

        move_ratio = obs.cx_ratio - self._legacy_track[0][1]
        if abs(move_ratio) < self._legacy_swipe_min_dist_ratio:
            return None
        class_name = SWIPE_RIGHT if move_ratio > 0 else SWIPE_LEFT
        return self._confirm(class_name, obs.conf, now_sec)

    # ----- 공통 -----

    def _is_in_cooldown(self, now_sec):
        return (
            self._last_event_ts_sec is not None
            and now_sec - self._last_event_ts_sec < self._cooldown_sec
        )

    def _confirm(self, class_name, conf, now_sec, hand_side=None, data=None):
        self._last_event_ts_sec = now_sec
        self._next_tracker.reset()
        self._prev_tracker.reset()
        self._select_tracker.reset()
        self._pause_tracker.reset()
        self._cancel_tracker.reset()
        self._home_tracker.reset()
        self._sos_tracker.reset()
        self._legacy_static_tracker.reset()
        self._legacy_track.clear()

        event = GestureEvent(
            class_name=class_name, conf=conf, ts_sec=now_sec, hand_side=hand_side, data=data
        )
        logger.info("gesture_event: %s (conf=%.2f, side=%s)", class_name, conf, hand_side)
        return event
