# Test & Legacy Scripts

These are the original experimental and diagnostic scripts written before the main GUI app existed. They are kept here for reference and low-level hardware testing.

---

## Scripts

### `camScan.py`

Live webcam scanner with Arduino control via pyfirmata. Supports two modes:

- **Manual mode** — press `n` to trigger a scan, `q` to quit.
- **Autonomous mode** — Arduino feeds cards automatically, servo sorts into two piles.

Config variables are at the top of the file (serial port, motor pin, servo pin, timings).

**Dependencies:** `opencv-python`, `python-doctr[torch]`, `paddleocr`, `paddlepaddle`, `pyfirmata2`

```bash
python3 camScan.py
```

---

### `pictureScan.py`

Scans a directory of card image files using DocTR as the primary OCR engine with PaddleOCR as fallback.

```bash
python3 pictureScan.py
```

Prompts for an image directory (default: `photos/`) and cropping mode.

---

### `pictureScan-paddleocr.py`

Same as `pictureScan.py` but uses PaddleOCR only — no DocTR.

```bash
python3 pictureScan-paddleocr.py
```

---

### `pictureScan-surya.py`

Uses Surya OCR instead of DocTR or Paddle. Slower and heavier but can be more accurate on difficult text.

```bash
python3 pictureScan-surya.py
```

**Install:** `pip install surya-ocr==0.16.0` (v0.17.1+ has compatibility issues)

---

### `servo-test.py`

Interactive servo angle tester. Connects to Arduino via pyfirmata and lets you enter degree values to move the servo in real time. Useful for calibrating sort positions.

```bash
python3 servo-test.py
```

---

### `sorter-test.py`

Tests the full sorter movement sequence (center → pile A → center → pile B) in a loop.

```bash
python3 sorter-test.py
```

---

### `step-test.py`

Sends raw step commands to the stepper motor controller to verify direction, speed, and step count.

```bash
python3 step-test.py
```

---

### `test.py`

General scratch test script used during early development.

---

### `ocr_processing.py`

Shared OCR utility module used by the scan scripts. Handles text extraction, regex matching against set IDs, and card number parsing.

---

## Directories

| Directory | Contents |
|-----------|----------|
| `photos/` | Sample card images used as input for the picture-scan scripts |
| `webcam/` | Frame captures from early webcam testing sessions |
| `debug_output/` | Intermediate images saved during scan debugging |
| `debug_output_webcam/` | Intermediate images from webcam scan debugging |

---

## Common Dependencies

```bash
pip install opencv-python numpy pyfirmata2
pip install python-doctr[torch]
pip install paddlepaddle paddleocr
```

Arduino must have the StandardFirmata sketch uploaded before connecting.
