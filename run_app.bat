@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=D:\dev\anacondalenvs\26xv01q\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Python environment not found: %PYTHON_EXE%
  echo Run setup_env.ps1 first, or create .venv in this project.
  pause
  exit /b 1
)

"%PYTHON_EXE%" main.py
endlocal
