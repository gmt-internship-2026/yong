"""설치 검증(스모크 테스트) — 카메라 없이 설치가 제대로 됐는지 확인한다.

install.bat 마지막 단계에서 자동 실행된다. 확인 항목:
1. 파이썬 버전이 배포 기준(runtime.python_version)과 맞는가
2. 핵심 패키지 임포트 (torch·ultralytics·cv2·fastapi / 선택: easyocr·pyttsx3)
3. GPU(CUDA) 인식 여부
4. 모델 가중치 파일 존재 + 더미 프레임 1장 추론

사용법 (프로젝트 루트에서):
    python scripts/smoke_test.py
종료 코드 0 = 통과. 실패 항목은 [FAIL]로 표시된다.
"""
import os
import platform
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

is_all_passed = True


def check(label, is_ok, detail=""):
    global is_all_passed
    mark = "PASS" if is_ok else "FAIL"
    if not is_ok:
        is_all_passed = False
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def main():
    config = load_config(DEFAULT_CONFIG_PATH)

    expected_python = config["runtime"]["python_version"]
    actual_python = platform.python_version()
    check(
        f"파이썬 버전 {expected_python}",
        actual_python == expected_python,
        f"현재 {actual_python}" + ("" if actual_python == expected_python else " (버전 불일치 — 동작은 하나 배포 기준과 다름)"),
    )

    try:
        import torch

        check("torch 임포트", True, torch.__version__)
        check("CUDA(NVIDIA GPU) 인식", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU 폴백 동작")
    except ImportError as error:
        check("torch 임포트", False, str(error))

    for module_name in ("ultralytics", "cv2", "fastapi", "yaml", "numpy"):
        try:
            __import__(module_name)
            check(f"{module_name} 임포트", True)
        except ImportError as error:
            check(f"{module_name} 임포트", False, str(error))

    if config["ocr"]["enabled"]:
        try:
            __import__("easyocr")
            check("easyocr 임포트 (주민등록증 OCR)", True)
        except ImportError as error:
            check("easyocr 임포트 (주민등록증 OCR)", False, str(error))
    if config["announce"]["enabled"] and config["announce"]["backend"] == "tts":
        try:
            __import__("pyttsx3")
            check("pyttsx3 임포트 (음성 안내)", True)
        except ImportError as error:
            check("pyttsx3 임포트 (음성 안내)", False, str(error))

    for label, path_key in (("제스처 가중치", "weights_path"), ("포즈 가중치", "pose_weights_path")):
        path = config["model"][path_key]
        check(f"{label} 파일", os.path.exists(path), path)

    try:
        import numpy as np

        from src.inference.trt_engine import GestureDetector

        detector = GestureDetector(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.infer(dummy)
        check("더미 프레임 추론", True)
    except Exception as error:  # 모델 미다운로드·드라이버 문제 등 — 원인 그대로 보여준다
        check("더미 프레임 추론", False, repr(error))

    print()
    if is_all_passed:
        print("[OK] 스모크 테스트 통과 — run.bat으로 실행하세요")
        return 0
    print("[NG] 실패 항목이 있습니다 — 설치가이드.md 문제 해결 절 참고")
    return 1


if __name__ == "__main__":
    sys.exit(main())
