@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Python 3.12 is required. See README.txt for installation instructions.
    pause
    exit /b 1
  )
)
if not exist ".venv\installed.txt" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
  echo installed>".venv\installed.txt"
)
".venv\Scripts\python.exe" app.py
pause
