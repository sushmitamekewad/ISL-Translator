@echo off
cd /d "%~dp0"
echo ========================================================
echo    Training All 3 Indian Sign Language (ISL) Models
echo    (MobileNetV3-Small, EfficientNet-Lite0, YOLO11n)
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
".venv\Scripts\python.exe" train_all.py %*
pause
