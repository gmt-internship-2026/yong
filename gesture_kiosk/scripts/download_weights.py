"""사전학습 가중치(.pt) 다운로드 — 제스처(HaGRIDv2) + 사람 포즈(YOLO11n-pose).

사용법:
    python scripts/download_weights.py            # 제스처 YOLOv10n + 포즈 yolo11n-pose
    python scripts/download_weights.py --model x  # 제스처를 YOLOv10x(대형)로

라이선스 주의(기획서 9장 №9): HaGRID는 CC BY-SA 4.0 변형 라이선스다.
상용 제품 탑재 전 반드시 회사와 라이선스 검토를 거칠 것.
(yolo11n-pose는 ultralytics 배포 — AGPL-3.0. 상용 탑재 조건도 함께 검토 대상, TODO №9)
"""
import argparse
import os
import sys
import urllib.request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "models", "weights")

GESTURE_MODEL_URLS = {
    "n": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt",
    "x": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10x_gestures.pt",
}
POSE_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"


def _print_progress(block_count, block_size, total_size):
    done_mb = block_count * block_size / 1e6
    total_mb = total_size / 1e6
    sys.stdout.write(f"\r  다운로드 중... {done_mb:.1f} / {total_mb:.1f} MB")
    sys.stdout.flush()


def _download(url):
    weights_path = os.path.join(WEIGHTS_DIR, os.path.basename(url))
    if os.path.exists(weights_path):
        print(f"[SKIP] 이미 존재합니다: {weights_path}")
        return
    print(f"[INFO] {url}")
    urllib.request.urlretrieve(url, weights_path, reporthook=_print_progress)
    print(f"\n[DONE] 저장 완료: {weights_path}")


def main():
    parser = argparse.ArgumentParser(description="제스처·포즈 가중치 다운로드")
    parser.add_argument("--model", choices=["n", "x"], default="n", help="제스처 모델 크기")
    args = parser.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    _download(GESTURE_MODEL_URLS[args.model])
    _download(POSE_MODEL_URL)
    print("[주의] HaGRID(CC BY-SA 4.0 변형)·ultralytics(AGPL-3.0) 라이선스 — 상용 탑재 전 회사 검토 필요")


if __name__ == "__main__":
    main()
