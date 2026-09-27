# Scramble Switch Sorter

A desktop GUI for automatically identifying and physically sorting Pokémon TCG cards. The app controls a serial-connected Arduino sorter, a webcam, and two OCR engines from a single interface.

Supports all sets that use a 3-letter set ID (Scarlet & Violet and onwards), which is read from the bottom of each card alongside the card number. Cards are matched against a local JSON database and routed to one of three physical sort carts.

---

## Features

- **Sort & Scan automation** — feed, scan, identify, and sort cards continuously with configurable timing and retries
- **Three-cart sorting** — assign sets, types, or categories to each cart; unmatched cards fall back to a configurable default cart
- **Live camera feed** with drag-to-set OCR crop area
- **Dual OCR engines** — DocTR (primary, fast CPU inference) with PaddleOCR fallback
- **In-app database builder** — point the app at a downloaded tcgdex repo and build the card database with one click; no scripts needed
- **Active Set Filters** — enable only the sets you own to reduce false-positive matches; populated automatically from the loaded database
- **Connection status panel** — one-click "Connect All" with per-item pulsing indicator during setup, and detail readout (port, resolution, model names, card count) once connected
- **Settings persistence** — all configuration auto-saves and restores between sessions
- **macOS app bundle** via PyInstaller; releases built automatically by GitHub Actions on version tags

---

## Tabs

### Sort & Scan
Main control tab. Shows connection indicators for serial, camera, OCR, and database. "Connect All" attempts all four connections simultaneously and pulses blue while each is in progress. Start and stop the automated sort-and-scan loop here, and track cards sorted.

### Sort Settings
Three sections:

**Cart rules** — For each of the three carts, pick a sort field (types, category, or set) and select the values that should route cards to that cart. Multiple values use Ctrl+click. Unmatched cards go to the fallback cart set at the bottom.

**Build Database** — Download the [tcgdex/cards-database](https://github.com/tcgdex/cards-database) repository (see setup instructions below), then point the folder picker at its `data/` subfolder and click **Build Database**. The app walks every set folder, extracts card data from the TypeScript source files, and writes `card_data_lookup.json` to the path configured in the Connect tab. Progress is shown inline. The database is loaded automatically when the build completes.

**Active Set Filters** — A list of all set abbreviations found in the loaded database. Select the sets you own; the OCR will only attempt to match cards from selected sets, which reduces false positives for sets you do not have. Use **Select All** / **Deselect All** for convenience. Your selection is saved automatically.

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

The app looks up identified cards in a local JSON file (`card_data_lookup.json`). You can build this file directly from the GUI — no command-line steps needed.

#### Option A: Build from the GUI (recommended)

1. Clone or download the tcgdex card database:

   ```bash
   git clone https://github.com/tcgdex/cards-database.git
   ```

2. Open the app (`python3 app/main.py`) and go to the **Sort Settings** tab.

3. Under **Build Database**, click **…** next to the folder picker and navigate to the `data/` subfolder inside the cloned `cards-database` repo (e.g. `cards-database/data`).

4. The **Output** path shown below the folder picker is the database file path configured in the Connect tab. If it is empty, go to Connect → Card Database and set an output path first (e.g. `card_data_lookup.json`).

5. Click **Build Database**. The app walks every set folder, extracts card data from the TypeScript source files, and writes the JSON file. A progress message updates inline; a green "✓ Built: N cards, M sets" message confirms success.

6. The database is loaded automatically when the build completes. The **Active Set Filters** list at the bottom of the tab is populated with every set abbreviation found in the database.

#### Option B: Provide an existing JSON file

If you already have a `card_data_lookup.json`, set its path in the Connect tab's database path field and click **Load Database**.

### 5. Configure Active Set Filters

After building or loading a database, open **Sort Settings → Active Set Filters**. Deselect any sets you do not own. The OCR will only attempt to match cards from selected sets, which significantly reduces false positives when sets share abbreviation characters.

Your selection is saved automatically and restored the next time the app starts.

### 6. Run the app

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
    card_db_builder.py Database builder — reads tcgdex .ts files, writes JSON
    version.py         App version string
  assets/
    logo.svg           Header logo

scripts/
  build_macos.sh       PyInstaller macOS build script

test-scripts/          Legacy and diagnostic scripts (see test-scripts/readme.md)
```
