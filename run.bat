@echo off
cd /d "%~dp0"
echo ========================================================
echo    Starting Indian Sign Language (ISL) Translator
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please create the virtual environment first.
    pause
    exit /b 1
)

IF NOT EXIST ".env" (
    echo [INFO] Creating .env from .env.example...
    copy .env.example .env
)

echo Activating virtual environment and starting server...
echo Access the live application at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.
call .venv\Scripts\activate.bat
".venv\Scripts\python.exe" main.py
pause
