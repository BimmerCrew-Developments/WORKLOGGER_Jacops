# Worklogger Processor

This repository contains the Worklogger Processor tool (`ssv_zip_processor_gui_v22.py`). It converts exported SafetyAuditor ZIP files into compressed images and a formatted PDF report.

## Ship a standalone build (no Python required for end users)

You can package the tool with PyInstaller so macOS and Windows users can run it without installing Python or dependencies. Both scripts bundle the Python runtime, ReportLab, Pillow, and the GUI into the output.

### macOS (.app)

1. On a Mac with Python 3.10+ installed for **build time only**, run:
   ```sh
   chmod +x build_macos_app.sh
   ./build_macos_app.sh
   ```
2. Distribute `dist/WorkloggerProcessor.app` to your users. They can double-click the app without installing Python.
3. If macOS Gatekeeper blocks the app ("kan niet worden geopend"):
   - Right-click the app, choose **Open**, then confirm.
   - Or clear quarantine via Terminal: `xattr -d com.apple.quarantine WorkloggerProcessor.app`.

### Windows (.exe)

1. On a Windows machine with Python 3.10+ available for **build time only**, run:
   ```bat
   build_windows_exe.bat
   ```
2. Distribute `dist/WorkloggerProcessor.exe` to your users. They can run the executable without installing Python.

### CI builds on GitHub Actions

Pushes, pull requests, and manual `workflow_dispatch` runs trigger `.github/workflows/build-binaries.yml`, which produces ready-to-distribute bundles:

- `worklogger-windows`: a zipped `WorkloggerProcessor.exe` built with `build_windows_exe.bat` on `windows-latest`.
- `worklogger-macos`: a zipped `WorkloggerProcessor.app` built with `build_macos_app.sh` on `macos-latest`.

Download the artifacts from the workflow run to share with testers or end users.

### Optional: run from source with Python

If you prefer to run the script directly with your local Python (instead of a bundled build):

1. Install Python 3.10+ with Tkinter (python.org installers include Tk).
2. Install dependencies:
   ```sh
   pip install -r requirements_v22.txt
   ```
3. Start the GUI:
   ```sh
   python3 ssv_zip_processor_gui_v22.py
   ```

You can still use `run_worklogger.command` on macOS if you already have Python installed; it will install dependencies and launch the GUI.
