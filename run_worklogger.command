#!/bin/bash
# Simple launcher for macOS users to run the Worklogger Processor GUI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 is not available. Install Python 3.10+ from https://www.python.org/downloads/."
  exit 1
fi

# Check Tkinter availability so the GUI can start.
if ! "$PYTHON_BIN" - <<'PY'
try:
    import tkinter  # noqa: F401
except Exception as e:
    raise SystemExit(f"Tkinter is required for the GUI. Install the Python.org build of Python 3 with Tk support. Details: {e}")
PY
then
  exit 1
fi

# Install dependencies locally for the user account (no sudo required).
"$PYTHON_BIN" -m pip install --user -r requirements_v22.txt

# Launch the GUI.
exec "$PYTHON_BIN" ssv_zip_processor_gui_v22.py
