@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   SOP 实时动作检测系统 — 环境安装
echo ============================================================
echo.

cd /d "%~dp0"

:: ── Step 0: check Python ──
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.9+
    echo         下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [0/3] Python:
python --version

:: ── Step 1: install PyTorch ──
echo.
echo [1/3] 安装 PyTorch ...

:: Try cu118 first (covers all gaming GPUs: GTX 10xx ~ RTX 40xx)
:: If nvidia-smi is available, use GPU version. Otherwise try cu118 anyway
:: (torch CUDA wheels install fine even without GPU, just won't use CUDA at runtime)
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo   检测到 NVIDIA 显卡，安装 CUDA 11.8 版 (兼容所有游戏本)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --retries 3
) else (
    echo   未检测到 nvidia-smi，尝试安装 CUDA 版...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --retries 3
    if %errorlevel% neq 0 (
        echo   回退到 CPU 版...
        pip install torch torchvision --retries 3
    )
)
if %errorlevel% neq 0 (
    echo [WARN] PyTorch 安装失败，请检查网络后重试
)

:: ── Step 2: install other deps ──
echo.
echo [2/3] 安装其他依赖 ...
pip install ultralytics opencv-python "mediapipe>=0.10.30" Pillow scipy numpy pandas scikit-learn pyyaml tqdm matplotlib --retries 3
if %errorlevel% neq 0 (
    echo [WARN] 部分依赖安装失败，请检查网络后重试
)

:: ── Step 3: verify ──
echo.
echo [3/3] 验证安装 ...
python -c "import torch; print('  PyTorch:', torch.__version__, '| CUDA:', 'YES' if torch.cuda.is_available() else 'NO'); import ultralytics; print('  Ultralytics: OK'); import cv2; print('  OpenCV: OK'); import mediapipe; print('  MediaPipe: OK'); import numpy; print('  NumPy: OK')" 2>&1
if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   安装完成！所有依赖就绪。
    echo.
    echo   启动: python local_run.py --camera hik     (海康相机)
    echo         python local_run.py --camera webcam  (USB摄像头)
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo   安装完成，但验证发现问题。尝试重新运行本脚本。
    echo   如果持续失败，请检查 Python 环境和网络连接。
    echo ============================================================
)
pause
