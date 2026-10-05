@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Follow the setup instructions in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m titanic.server --open-browser
if errorlevel 1 pause
