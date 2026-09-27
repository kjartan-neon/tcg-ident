#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$REPO_ROOT/app"
DIST_DIR="$REPO_ROOT/dist"

cd "$REPO_ROOT"

# Read version
VERSION=$(python3 -c "import sys; sys.path.insert(0,'$APP_DIR/source'); from version import __version__; print(__version__)")
echo "Building Scramble Switch Sorter v$VERSION …"

# Install PyInstaller if missing
pip install pyinstaller --quiet

# Clean previous build
rm -rf "$DIST_DIR" build

pyinstaller \
  --name "Scramble Switch Sorter" \
  --onedir \
  --windowed \
  --icon "$APP_DIR/assets/logo.svg" \
  --add-data "$APP_DIR/assets:assets" \
  --add-data "$APP_DIR/source:source" \
  --hidden-import cv2 \
  --hidden-import PIL \
  --hidden-import PIL._tkinter_finder \
  --hidden-import cairosvg \
  --hidden-import serial \
  --hidden-import serial.tools.list_ports \
  --hidden-import doctr \
  --hidden-import paddleocr \
  --noconfirm \
  "$APP_DIR/main.py"

APP_BUNDLE="$DIST_DIR/Scramble Switch Sorter.app"
ZIP_NAME="$DIST_DIR/ScrambleSwitchSorter-v${VERSION}-macOS.zip"

if [ -d "$APP_BUNDLE" ]; then
  echo "Zipping app bundle…"
  cd "$DIST_DIR"
  zip -r "$ZIP_NAME" "Scramble Switch Sorter.app"
  echo "Done: $ZIP_NAME"
else
  # onedir mode — zip the folder
  ZIP_NAME="$DIST_DIR/ScrambleSwitchSorter-v${VERSION}-macOS.zip"
  cd "$DIST_DIR"
  zip -r "$ZIP_NAME" "Scramble Switch Sorter"
  echo "Done: $ZIP_NAME"
fi
