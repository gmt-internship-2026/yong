"""모델 변환 2단계 — TensorRT FP16 엔진 빌드 (Jetson 실기기에서 실행할 것).

캡스톤의 turnonnx.py와 같은 원리(half=True FP16 가속)로,
결과물을 기획서 2.3 구조의 models/engines/ 경로에 배치한다.

사용법 (Jetson Orin Nano에서):
    python scripts/build_engine.py
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


def main():
    parser = argparse.ArgumentParser(description=".pt -> TensorRT .engine 빌드 (FP16)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights", default=None, help="미지정 시 config의 weights_path")
    args = parser.parse_args()

    config = load_config(args.config)
    weights_path = args.weights or config["model"]["weights_path"]
    engine_path = config["model"]["engine_path"]
    input_size_px = config["model"]["input_size_px"]

    from ultralytics import YOLO

    model = YOLO(weights_path)
    built_path = model.export(format="engine", half=True, imgsz=input_size_px, device=0)

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    shutil.move(built_path, engine_path)
    print(f"[DONE] TensorRT 엔진 저장: {engine_path}")
    print("[다음] config.yaml에서 model.backend: engine 으로 변경")


if __name__ == "__main__":
    main()
