@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo             Whisper Transcriber Setup
echo ===================================================
echo.

:: Check Python
where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Please install Python 3.10+ and make sure it is added to your PATH.
  pause
  exit /b 1
)
echo [OK] Python is installed.

:: Setup Python virtual environment
cd /d "%~dp0"
if not exist "Code\venv" (
  echo Creating Python virtual environment in 'Code\venv'...
  python -m venv Code\venv
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
  )
)
echo [OK] Python virtual environment is set up.

echo Upgrading pip and installing Python dependencies (faster-whisper, cuda support, etc.)...
call Code\venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo WARNING: Failed to upgrade pip. Proceeding anyway...
)

call Code\venv\Scripts\pip.exe install -r Code\requirements.txt
if errorlevel 1 (
  echo ERROR: Failed to install Python dependencies.
  pause
  exit /b 1
)
echo [OK] Python dependencies installed successfully.

echo.
echo ===================================================
echo Setup complete! You can now start the dashboard.
echo Click 'run.bat' to launch the app.
echo ===================================================
echo.
pause
