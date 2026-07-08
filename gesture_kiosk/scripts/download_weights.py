"""HaGRIDv2 사전학습 제스처 검출 가중치(.pt) 다운로드 — 학습 없이 바로 사용.

사용법:
    python scripts/download_weights.py            # YOLOv10n (경량, Jetson 권장)
    python scripts/download_weights.py --model x  # YOLOv10x (대형, 정확도 우선)

라이선스 주의(기획서 9장 №9): HaGRID는 CC BY-SA 4.0 변형 라이선스다.
상용 제품 탑재 전 반드시 회사와 라이선스 검토를 거칠 것.
"""
import argparse
import os
import sys
import urllib.request

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(ROOT_DIR, "models", "weights")

MODEL_URLS = {
    "n": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_gestures.pt",
    "x": "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10x_gestures.pt",
}


def _print_progress(block_count, block_size, total_size):
    done_mb = block_count * block_size / 1e6
    total_mb = total_size / 1e6
    sys.stdout.write(f"\r  다운로드 중... {done_mb:.1f} / {total_mb:.1f} MB")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="HaGRIDv2 가중치 다운로드")
    parser.add_argument("--model", choices=["n", "x"], default="n")
    args = parser.parse_args()

    url = MODEL_URLS[args.model]
    weights_path = os.path.join(WEIGHTS_DIR, os.path.basename(url))

    if os.path.exists(weights_path):
        print(f"[SKIP] 이미 존재합니다: {weights_path}")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"[INFO] {url}")
    urllib.request.urlretrieve(url, weights_path, reporthook=_print_progress)
    print(f"\n[DONE] 저장 완료: {weights_path}")
    print("[주의] HaGRID 라이선스(CC BY-SA 4.0 변형) — 상용 탑재 전 회사 검토 필요 (기획서 9장 №9)")


if __name__ == "__main__":
    main()
