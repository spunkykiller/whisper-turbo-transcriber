@echo off
cd /d "%~dp0"
title Whisper Transcription Dashboard

echo === Whisper Transcription Dashboard ===
echo.

if not exist "Code\venv" (
    echo [1/3] Creating virtual environment...
    py -m venv Code\venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Make sure Python 3.8+ is installed.
        pause
        exit /b %errorlevel%
    )
) else (
    echo [1/3] Virtual environment found.
)

echo [2/3] Installing dependencies...
call Code\venv\Scripts\activate.bat
pip install -r Code\requirements.txt -q
if %errorlevel% neq 0 (
    echo Dependency installation failed.
    pause
    exit /b %errorlevel%
)

echo [3/3] Starting server...
echo.
echo The dashboard will open in your browser shortly.
echo If it doesn't, navigate to http://127.0.0.1:5000
echo.
python Code\server.py

pause
