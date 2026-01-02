@echo off
setlocal enabledelayedexpansion

REM Build a standalone Windows .exe that ships Python and all dependencies
REM Usage: run from Developer Command Prompt or PowerShell: build_windows_exe.bat

python -m pip install --upgrade pip
python -m pip install -r requirements_v22.txt pyinstaller

pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name WorkloggerProcessor ^
  ssv_zip_processor_gui_v22.py

echo Standalone executable created at dist\WorkloggerProcessor.exe
