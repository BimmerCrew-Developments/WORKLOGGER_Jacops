#!/usr/bin/env bash
set -euo pipefail

# Build a standalone macOS .app bundle that ships Python and all dependencies
# Usage: ./build_macos_app.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements_v22.txt pyinstaller

pyinstaller \
  --noconfirm \
  --clean \
  --windowed \
  --name WorkloggerProcessor \
  ssv_zip_processor_gui_v22.py

echo "Standalone app created at dist/WorkloggerProcessor.app"
