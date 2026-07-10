"""모델 변환 2단계 — TensorRT FP16 엔진 빌드 (반드시 실행할 그 PC에서!).

.engine 파일은 GPU 기종·드라이버·TensorRT 버전에 묶여 있어 다른 PC로
복사·이식할 수 없다. 설치 대상 윈도우 PC에서 install.bat 마지막 단계로
실행하거나, 이 스크립트를 직접 실행한다 (약 5~15분, PC당 1회).

사용법:
    python scripts/build_engine.py            # 제스처 + 포즈 엔진 모두 빌드
    python scripts/build_engine.py --target gesture   # 하나만
빌드 후 config.yaml의 model.backend를 engine으로 바꾸면 파이프라인이 엔진을 사용한다.
"""
import argparse
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def _build(weights_path, engine_path, input_size_px):
    from ultralytics import YOLO

    print(f"[INFO] TensorRT 엔진 빌드 시작: {os.path.basename(weights_path)}")
    model = YOLO(weights_path)
    built_path = model.export(format="engine", half=True, imgsz=input_size_px, device=0)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    shutil.move(built_path, engine_path)
    print(f"[DONE] TensorRT 엔진 저장: {engine_path}")


def main():
    parser = argparse.ArgumentParser(description=".pt -> TensorRT .engine 빌드 (FP16)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target", choices=["all", "gesture", "pose"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    input_size_px = config["model"]["input_size_px"]

    if args.target in ("all", "gesture"):
        _build(config["model"]["weights_path"], config["model"]["engine_path"], input_size_px)
    if args.target in ("all", "pose"):
        _build(
            config["model"]["pose_weights_path"],
            config["model"]["pose_engine_path"],
            input_size_px,
        )
    print("[다음] config.yaml에서 model.backend: engine 으로 변경")


if __name__ == "__main__":
    main()
