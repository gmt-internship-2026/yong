"""디버그 시각화 — 검출 결과와 상태를 프레임 위에 그린다 (예시 UI 스트림에도 사용)."""
import cv2

BBOX_COLOR = (0, 220, 120)
EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)


def draw_bbox(frame, detections):
    """검출된 제스처 bbox와 이름·conf를 그린다."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)
        cv2.putText(
            frame,
            f"{det.class_name} {det.conf:.2f}",
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BBOX_COLOR,
            2,
        )
    return frame


def draw_status(frame, avg_fps, gesture_event=None):
    """FPS와 최근 확정 이벤트를 좌상단에 표시한다."""
    cv2.putText(
        frame, f"FPS {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2
    )
    if gesture_event is not None:
        cv2.putText(
            frame,
            f"EVENT {gesture_event.class_name}",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            EVENT_COLOR,
            2,
        )
    return frame
