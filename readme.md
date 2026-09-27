# Scramble Switch Sorter

A desktop GUI for automatically identifying and physically sorting Pokémon TCG cards. The app controls a serial-connected Arduino sorter, a webcam, and two OCR engines from a single interface.

Supports all sets that use a 3-letter set ID (Scarlet & Violet and onwards), which is read from the bottom of each card alongside the card number. Cards are matched against a local JSON database and routed to one of three physical sort carts.

---

## Features

- **Sort & Scan automation** — feed, scan, identify, and sort cards continuously with configurable timing and retries
- **Three-cart sorting** — assign sets, types, or categories to each cart; unmatched cards fall back to a configurable default cart
- **Live camera feed** with drag-to-set OCR crop area
- **Dual OCR engines** — DocTR (primary, fast CPU inference) with PaddleOCR fallback
- **Connection status panel** — one-click "Connect All" with per-item pulsing indicator during setup, and detail readout (port, resolution, model names, card count) once connected
- **Settings persistence** — all configuration auto-saves and restores between sessions
- **macOS app bundle** via PyInstaller; releases built automatically by GitHub Actions on version tags

---

## Tabs

### Sort & Scan
Main control tab. Shows connection indicators for serial, camera, OCR, and database. "Connect All" attempts all four connections simultaneously and pulses blue while each is in progress. Start and stop the automated sort-and-scan loop here, and track cards sorted.

### Sort Settings
Assign sorting rules per cart. For each of the three carts, pick a field (types, category, set) and select the values that should route to that cart. Multiple selections use Ctrl+click.

### Connect
Manual connection controls for each subsystem independently — serial port, camera index, OCR model loading, and database file path. Includes port and camera discovery buttons.

### Sorter
Low-level hardware controls: gate servo angles, stepper step size and delay, manual cart movement, feed/stop/release, and multi-cycle test runs.

### Scanner
Crop area controls and on-demand scan testing. Drag on the camera feed to set the OCR region, or clear it to use the full frame. Scan Card Now and Scan & Identify Cart let you test recognition without running the full loop.

---

## Hardware

| Component | Details |
|-----------|---------|
| Arduino (or compatible) | Connected via USB serial; runs a custom firmware that accepts simple serial commands |
| Stepper motor | Moves the three-cart tray; step size and delay are configurable |
| Servo motor | Opens and closes the card drop gate |
| Webcam | Any USB or built-in camera; index selectable in the Connect tab |

The app communicates over serial at a configurable baud rate (default 115200). Use the Connect tab to scan for ports and select the correct one.

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd tcg-ident
```

### 2. Create a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Python dependencies

```bash
pip install opencv-python pillow pyserial cairosvg
pip install python-doctr[torch]
pip install paddlepaddle paddleocr
```

PaddleOCR is optional but recommended — the app loads it as a fallback. DocTR and Paddle models download automatically on first use.

### 4. Prepare the card database

The app looks up identified cards in a local JSON file (`card_data_lookup.json`).

1. Clone or download the [tcgdex/cards-database](https://github.com/tcgdex/cards-database) repository.
2. Create a `tcgdex/data/` folder in this project and copy the set data folders into it.
   See [`tcgdex/readme.md`](tcgdex/readme.md) for details.
3. Generate the lookup file:

   ```bash
   python3 get-card-data.py
   ```

This produces `card_data_lookup.json` in the project root. Point the app to it via the Connect tab's database path field (it auto-detects if the file is in the project root).

### 5. Run the app

```bash
python3 app/main.py
```

---

## Building a macOS app bundle

```bash
bash scripts/build_macos.sh
```

Requires PyInstaller and the Python dependencies above. Produces a `.zip` containing the `.app` bundle in `dist/`.

GitHub Actions builds and attaches the zip to a release automatically when a `v*` tag is pushed:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

---

## Test and legacy scripts

The [`test-scripts/`](test-scripts/) directory contains the original experimental scripts from before the GUI app existed — webcam scanners, picture-directory scanners, servo/stepper testers, and OCR experiments. See [`test-scripts/readme.md`](test-scripts/readme.md) for details on each script.

---

## Project structure

```
app/
  main.py              Entry point
  gui.py               Full GUI — tabs, camera, indicators, sort loop
  source/
    scanner.py         Camera capture, OCR pipeline, database lookup
    serial_controller.py  Arduino serial protocol
    ocr_processing.py  Text extraction and regex helpers
    version.py         App version string
  assets/
    logo.svg           Header logo

scripts/
  build_macos.sh       PyInstaller macOS build script

test-scripts/          Legacy and diagnostic scripts (see test-scripts/readme.md)

tcgdex/                Card database source data
  readme.md            Instructions for obtaining set data
```
