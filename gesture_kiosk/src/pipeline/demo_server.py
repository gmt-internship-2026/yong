"""예시 UI 서버 — 회사 키오스크 프로그램이 들어올 자리의 임시 대체물.

TODO(기획서 9장 №7·№8): 회사 프로그램(UI) 파일을 받으면 이 서버와 demo_ui/는
제거하고, event_sender.py 규격으로 이벤트만 전달한다. 지금은 캡스톤 관제 UI
(jetson_USB2.py의 FastAPI + MJPEG 스트림 구조)를 단일 카메라·제스처용으로
옮겨 와 시연용으로 쓴다.
"""
import asyncio
import os

import cv2
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

try:
    import psutil
except ImportError:  # psutil 미설치 환경(개발 PC)에서도 서버는 떠야 한다
    psutil = None

DEMO_UI_HTML = "demo_ui/index.html"
RECENT_EVENT_COUNT = 20


def create_app(state, config):
    app = FastAPI(title="Gesture Kiosk Demo UI (예시 — 회사 프로그램 대체 예정)")
    index_path = os.path.join(config["root_dir"], DEMO_UI_HTML)
    stream_interval_sec = 1.0 / config["demo_ui"]["stream_fps"]
    jpeg_quality = config["demo_ui"]["jpeg_quality"]

    @app.get("/")
    async def serve_index():
        return FileResponse(index_path)

    async def _stream():
        while True:
            frame = state.get_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ret:
                await asyncio.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(stream_interval_sec)

    @app.get("/video_feed")
    async def video_feed():
        return StreamingResponse(
            _stream(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/data")
    async def get_data():
        events = [
            {"class_name": e.class_name, "conf": round(e.conf, 2), "ts_sec": e.ts_sec}
            for e in state.event_log[-RECENT_EVENT_COUNT:]
        ]
        return {
            "stats": {
                "cpu": psutil.cpu_percent(interval=None) if psutil else 0.0,
                "memory": psutil.virtual_memory().percent if psutil else 0.0,
                "capture_fps": round(state.capture_fps, 1),
                "infer_fps": round(state.infer_fps, 1),
            },
            "classes": config["classes"],
            "events": events,
        }

    return app
