@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "RESOLVED_PYCHARM=%PYCHARM_EXE%"

if defined RESOLVED_PYCHARM if not exist "%RESOLVED_PYCHARM%" set "RESOLVED_PYCHARM="
if not defined RESOLVED_PYCHARM if exist "D:\dev\pycharm\PyCharm 2025.3.3\bin\pycharm64.exe" set "RESOLVED_PYCHARM=D:\dev\pycharm\PyCharm 2025.3.3\bin\pycharm64.exe"

if not defined RESOLVED_PYCHARM (
  for /f "delims=" %%I in ('where pycharm64.exe 2^>nul') do if not defined RESOLVED_PYCHARM set "RESOLVED_PYCHARM=%%I"
)

if not defined RESOLVED_PYCHARM (
  echo PyCharm was not found.
  echo Set the PYCHARM_EXE environment variable to pycharm64.exe and try again.
  pause
  exit /b 1
)

if /I "%~1"=="--dry-run" (
  echo PyCharm: %RESOLVED_PYCHARM%
  echo Project: %PROJECT_ROOT%
  exit /b 0
)

start "" "%RESOLVED_PYCHARM%" "%PROJECT_ROOT%"
endlocal
