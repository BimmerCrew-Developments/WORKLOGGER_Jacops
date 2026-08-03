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

To process an export from the command line, pass the input ZIP and output
directory. You can select a saved export template by its stable ID:

```sh
python3 ssv_zip_processor_gui_v22.py --list-export-templates
python3 ssv_zip_processor_gui_v22.py --zip audit.zip --out reports \
  --export-template werklogger-report-v1
```

The listing shows both the stable ID accepted by `--export-template` and the
editable display name. Always use the ID in scripts.

You can still use `run_worklogger.command` on macOS if you already have Python installed; it will install dependencies and launch the GUI.

## Create a CSV-based export template

Open **Manage templates...** and choose **New**. Select an example CSV export when
prompted. A table preview lets you choose the row containing the column names and
how many leading columns belong to the export table, so introductory CSV rows and
unrelated trailing columns can be excluded. Worklogger stores only the selected
column names and range, not the example data. In the **CSV column mapping** tab
you can then:

- map renamed CSV columns to the standard import purposes (ID, type, label,
  answer, media, and so on); and
- optionally map a CSV column directly to each PDF field, such as Building ID,
  project name, address, or report date.

The other tabs continue to control the shared report layout, section order,
labels, text, colors, photo grid, and output files. This allows templates to use
the same visual layout while supporting exports with different column names and
purposes. The selected template's mapping is automatically applied when its ZIP
is processed.

## Run the test suite

The tests create isolated temporary directories and generate their own CSV, JPEG,
ZIP, and PDF fixtures, so no sample customer exports are required:

```sh
python3 -m pip install -r requirements-test.txt
python3 -m pytest
```
