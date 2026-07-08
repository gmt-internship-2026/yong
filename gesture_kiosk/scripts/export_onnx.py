"""모델 변환 1단계 — PyTorch 가중치(.pt) → ONNX (기획서 3.1 워크플로우).

사용법:
    python scripts/export_onnx.py                        # config의 weights_path 사용
    python scripts/export_onnx.py --weights models/weights/YOLOv10n_gestures.pt
"""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    parser = argparse.ArgumentParser(description=".pt -> .onnx 변환")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights", default=None, help="미지정 시 config의 weights_path")
    args = parser.parse_args()

    config = load_config(args.config)
    weights_path = args.weights or config["model"]["weights_path"]
    input_size_px = config["model"]["input_size_px"]

    from ultralytics import YOLO

    model = YOLO(weights_path)
    onnx_path = model.export(format="onnx", imgsz=input_size_px, simplify=True)
    print(f"[DONE] ONNX 저장: {onnx_path}")


if __name__ == "__main__":
    main()
