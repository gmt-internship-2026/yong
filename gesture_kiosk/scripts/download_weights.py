"""모델 준비 — 제스처 ONNX 확인 + 포즈(RTMPose) 모델 캐시 프리페치.

라이선스 B안(2026-07-11) 이후 구성:
- 제스처: HaGRIDv2 YOLOv10n의 ONNX 변환본이 저장소(models/weights/)에 포함 — 다운로드 불필요
- 포즈: rtmlib(RTMPose)가 첫 실행 때 자동 다운로드하는 것을 여기서 미리 받아 둔다
  (캐시 위치: ~/.cache/rtmlib — 내부망 반입 시 make_offline_bundle.bat이 함께 담는다)

사용법:
    python scripts/download_weights.py

라이선스 고지(기획서 9장 №9): HaGRID 모델은 자체 라이선스(저작자 표시 필수) —
고지문은 models/weights/NOTICE_HaGRID.md 참고. rtmlib/RTMPose는 Apache-2.0.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    config = load_config(DEFAULT_CONFIG_PATH)

    onnx_path = config["model"]["gesture_onnx_path"]
    if os.path.exists(onnx_path):
        print(f"[OK] 제스처 ONNX 확인: {onnx_path}")
    else:
        print(f"[FAIL] 제스처 ONNX가 없습니다: {onnx_path}")
        print("       저장소를 다시 받거나, 개발 PC에서 scripts/export_onnx.py로 변환하세요")
        sys.exit(1)

    pose_mode = config["model"]["pose_mode"]
    print(f"[INFO] 포즈 모델(rtmlib {pose_mode}) 캐시 준비 — 없으면 지금 내려받습니다 (수십 MB)")
    from rtmlib import Body

    Body(mode=pose_mode, backend="onnxruntime", device="cpu")  # 다운로드만 목적 — CPU로 가볍게
    print("[DONE] 포즈 모델 캐시 완료 (~/.cache/rtmlib)")
    print("[고지] HaGRID 저작자 표시 의무 — models/weights/NOTICE_HaGRID.md 를 제품 문서에 포함할 것")


if __name__ == "__main__":
    main()
