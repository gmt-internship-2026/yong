"""파인튜닝 골격 — 4주차 dataset_v1 확보 후 개발 PC에서 실행 (기획서 7.2).

주의: 학습은 반드시 개발 PC에서 한다. Jetson Orin Nano(SD카드 128GB)에서의
학습은 저장장치 수명·발열·시간 문제로 금지한다. Jetson은 추론 전용.

사용법 (PC에서):
    python scripts/train.py --data data/splits/dataset_v1.yaml --epochs 50

TODO(4주차): dataset_v1 라벨링 완료 후 아래 항목 확정
  - data/splits/dataset_v1.yaml 작성 (train/val/test 경로 + 기획서 5.1 클래스)
  - 인물 단위 분할 검증 (기획서 5.4 — 데이터 유출 방지)
  - 학습 결과 가중치 명명 규칙 적용: {모델}_{데이터셋}_{지표}_{날짜}.pt (기획서 4.8)
"""
import argparse
import os
import platform
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    parser = argparse.ArgumentParser(description="제스처 모델 파인튜닝 (PC 전용)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data", required=True, help="YOLO 데이터셋 yaml 경로")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    if platform.machine() == "aarch64":
        print("[중단] Jetson(aarch64)에서는 학습을 실행하지 않습니다 — SD카드·발열 보호.")
        print("       학습은 개발 PC에서 수행하고 가중치만 Jetson으로 옮기세요.")
        sys.exit(1)

    config = load_config(args.config)
    weights_path = config["model"]["weights_path"]
    input_size_px = config["model"]["input_size_px"]

    from ultralytics import YOLO

    model = YOLO(weights_path)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=input_size_px)
    print("[DONE] 학습 완료 — runs/ 폴더의 best.pt를 기획서 4.8 규칙으로 이름 바꿔 배포")


if __name__ == "__main__":
    main()
