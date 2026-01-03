@echo off
setlocal enabledelayedexpansion

REM Build a standalone Windows .exe that ships Python and all dependencies
REM Usage: run from Developer Command Prompt or PowerShell: build_windows_exe.bat

if not exist dist (
  mkdir dist
)

python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python -m pip install -r requirements_v22.txt pyinstaller
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name WorkloggerProcessor ^
  ssv_zip_processor_gui_v22.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

if not exist dist\WorkloggerProcessor.exe (
  echo dist\WorkloggerProcessor.exe was not created
  exit /b 1
)

echo Standalone executable created at dist\WorkloggerProcessor.exe
