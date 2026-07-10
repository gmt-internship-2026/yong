@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d %~dp0

echo ============================================================
echo  gesture_kiosk 설치 (윈도우 + NVIDIA GPU + Python 3.11.5)
echo  실행기: ONNX Runtime — TensorRT 가속은 설치 후
echo          configs\config.yaml 의 use_tensorrt: true 로 켠다
echo ============================================================

REM ---- 1) Python 3.11 확인 -----------------------------------
set PY_CMD=
py -3.11 --version >nul 2>&1 && set PY_CMD=py -3.11
if not defined PY_CMD (
    python --version 2>nul | findstr /C:"3.11" >nul && set PY_CMD=python
)
if not defined PY_CMD (
    echo [FAIL] Python 3.11을 찾지 못했습니다.
    echo        https://www.python.org/downloads/release/python-3115/ 에서
    echo        3.11.5 설치 시 "Add python.exe to PATH"를 체크하세요.
    exit /b 1
)
for /f "tokens=2" %%v in ('%PY_CMD% --version') do set PY_VER=%%v
echo [INFO] Python !PY_VER! 사용
if not "!PY_VER!"=="3.11.5" echo [경고] 배포 기준은 3.11.5 입니다 — 현재 !PY_VER! (대체로 동작하나 기준과 다름)

REM ---- 2) 가상환경 -------------------------------------------
if not exist venv_win (
    echo [INFO] 가상환경 생성 중...
    %PY_CMD% -m venv venv_win || exit /b 1
)
call venv_win\Scripts\activate.bat
python -m pip install --upgrade pip >nul

REM ---- 3) 패키지 설치 (오프라인 wheelhouse 우선) --------------
if exist wheelhouse (
    echo [INFO] 오프라인 설치 — wheelhouse\ 사용 (내부망 모드)
    pip install --no-index --find-links wheelhouse torch torchvision || goto :pip_fail
    pip install --no-index --find-links wheelhouse -r requirements.txt || goto :pip_fail
) else (
    echo [INFO] 온라인 설치 — torch는 EasyOCR용 CUDA 12.8 휠 (RTX 50시리즈 포함 지원)
    pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128 || goto :pip_fail
    pip install -r requirements.txt || goto :pip_fail
)

REM ---- 4) 모델 준비 (제스처 ONNX는 저장소 포함, 포즈는 캐시) ----
if exist bundle_models\rtmlib (
    echo [INFO] 포즈 모델 캐시 복사 (오프라인)
    xcopy /y /q /e bundle_models\rtmlib "%USERPROFILE%\.cache\rtmlib\" >nul
)
if exist bundle_models\easyocr (
    echo [INFO] EasyOCR 한국어 모델 복사 (오프라인)
    xcopy /y /q /e bundle_models\easyocr "%USERPROFILE%\.EasyOCR\model\" >nul
)
python scripts\download_weights.py || goto :model_fail

REM ---- 5) 스모크 테스트 ---------------------------------------
echo.
echo [INFO] 설치 검증 실행...
python scripts\smoke_test.py
if errorlevel 1 (
    echo [경고] 검증 실패 항목이 있습니다 — 설치가이드.md의 "문제 해결" 참고
)

echo.
echo [DONE] 설치 완료 — run.bat 으로 실행하세요 (브라우저: http://localhost:5000)
echo [가속] 30 FPS 미달 시: configs\config.yaml 에서 use_tensorrt: true
echo        (첫 실행 때 이 PC 전용 엔진 캐시를 자동 생성 — 몇 분 걸림, 복사·이식 금지)
exit /b 0

:pip_fail
echo [FAIL] 패키지 설치 실패 — 인터넷 연결 또는 wheelhouse\ 내용을 확인하세요 (설치가이드.md)
exit /b 1
:model_fail
echo [FAIL] 모델 다운로드 실패 — 내부망이면 bundle_models\ 를 준비하세요 (설치가이드.md B절)
exit /b 1
