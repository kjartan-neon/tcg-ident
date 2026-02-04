# 🃏 TCG Identifier

**TCG Identifier** is a Python-based tool for automatically identifying Pokémon Trading Card Game (TCG) cards from images or a live webcam feed. It uses computer vision to detect and isolate the card from its background and Optical Character Recognition (OCR) to extract key information like the set ID and card number. It supports all sets that have a *3 letter identification*, and uses OCR to find the card's name after looking up in a comprehensive card database.

## 🤖 Automated Card Feeder & Sorter (Arduino Integration)

The `camScan.py` script includes **innovative hardware automation** for high-speed card scanning and sorting using an Arduino-controlled system:

### 🔄 Automated Card Feeding
*   **Motor Control**: Arduino-controlled motor feeds cards one at a time into the scanning area
*   **Precise Timing**: Configurable motor run time and settle time for consistent positioning
*   **Autonomous Mode**: Continuous scanning with automatic card feeding after each identification

### 🎯 Intelligent Card Sorting
*   **Dual-Pile Sorting**: Servo motor sorts cards into two separate piles (A and B) based on identification results
*   **Intermediate Positioning**: Uses intermediate servo positions for smooth, reliable card sorting
*   **Customizable Sort Logic**: Easily configure sorting rules based on card type, set, or any other attribute
*   **Failure Handling**: Failed scans automatically sorted to Pile B

### 🔧 Smart Retry with Wiggle Feature
*   **Wiggle Mechanism**: If OCR fails, the servo wiggles the card to improve positioning
*   **Multi-Attempt Scanning**: 
    - 5 initial scan attempts with fresh frames
    - 3 additional wiggle-and-retry attempts if needed
    - Dual OCR engine fallback (DocTR → PaddleOCR)
*   **Adaptive Strategy**: Combines hardware movement with software retries for maximum success rate

### 📊 Stop Card Detection
*   **Auto-Stop Feature**: Configure a "stop card" (e.g., specific Pokémon name)
*   **Consecutive Detection**: Requires detecting the stop card twice in a row to prevent false positives
*   **Batch Processing**: Perfect for scanning entire decks or collections with a known end marker

### 🎮 Dual Operating Modes
1. **Autonomous Mode**: Fully automated feeding, scanning, and sorting
2. **Manual Mode**: Press 'n' to trigger scans on-demand (no Arduino required)

**Hardware Requirements**: Arduino board, servo motor (pin 9), DC motor/relay (pin 8), webcam

## ✨ Features

*   **Robust Card Detection:** Locates cards in images, even against complex backgrounds.
*   **Perspective Correction:** Applies a four-point perspective transform to "flatten" the card for accurate analysis.
*   **Targeted OCR:** Crops the card image to the bottom section where the set ID and number are typically located, improving OCR accuracy.
*   **Card Verification:** Looks up the extracted set ID and card number in a generated JSON database to find the card's name.
*   **Dual OCR Engine System:** DocTR (fast) with automatic PaddleOCR fallback for maximum accuracy.
*   **Multiple Scan Modes:**
    *   `pictureScan.py`: Scans a directory of card images using DocTR (fast) with PaddleOCR fallback (recommended).
    *   `pictureScan-paddleocr.py`: Scans a directory of card images using PaddleOCR only.
    *   `pictureScan-surya.py`: Scans a directory of card images using Surya OCR only.
    *   `camScan.py`: Scans for cards using a live webcam feed with DocTR (fast) and PaddleOCR fallback, with optional Arduino automation.

## ⚙️ Setup and Installation

Follow these steps to set up the project and its dependencies.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <repository-url>
cd tcg-ident
```

### 2. Install Python Dependencies

It is highly recommended to use a virtual environment.

```bash
python3 -m venv env
source env/bin/activate
```

This project requires several Python libraries. While a `requirements.txt` is not provided, you can install the necessary packages using pip:

```bash
pip install opencv-python numpy
```

### 3. Install OCR Engine

This project supports three OCR engines:

#### Option A: PaddleOCR

This is used by `pictureScan-paddleocr.py` and `camScan.py` (fallback). Install PaddlePaddle and PaddleOCR:

```bash
# Install PaddlePaddle (CPU version recommended for simplicity)
pip install paddlepaddle

# Install PaddleOCR
pip install paddleocr
```
*Note: PaddleOCR will automatically download the necessary detection and recognition models the first time it runs.*

#### Option B: DocTR with MobileNet (Recommended - Fast & Accurate)

This is used by `pictureScan.py` and `camScan.py` (primary). DocTR with MobileNet models provides the best balance of speed and accuracy for CPU-based scanning:

```bash
# Install DocTR with PyTorch backend
pip install python-doctr[torch]
```
*Note: DocTR uses lightweight MobileNet architectures that are optimized for CPU inference. Models download automatically on first run. This is the recommended option for fast card identification.*

**Dual-Engine Mode**: Both `pictureScan.py` and `camScan.py` use DocTR as the primary (fast) engine and automatically fall back to PaddleOCR if DocTR fails to identify a card. For best results, install both engines.

#### Option C: Surya OCR (Alternative - Slower but Accurate)

This is used by `pictureScan-surya.py`. Install the specific version that works correctly:

```bash
# Install Surya OCR version 0.16.0 (version 0.17.1+ has compatibility issues)
pip install surya-ocr==0.16.0
```
*Note: Surya OCR will automatically download models on first run. It requires more disk space and memory than other engines and is slower on CPU, but may provide better accuracy for challenging text recognition tasks.*

### 4. Prepare the Card Database

The script can verify the identified card against a comprehensive card database from **tcgdex**. To make it faster and usable for python, a script reads the data and extract the needed content, and then saves as a .json that can be used by python.

1.  **Download the Database:** Download the latest card data from the [tcgdex/cards-database](https://github.com/tcgdex/cards-database) repository. You can either clone it or download it as a ZIP file.

2.  **Organize Data:**
    *   Create a folder named `tcgdex` in the root of this project if it doesn't exist.
    *   Inside `tcgdex`, create a folder named `data`.
    *   Copy the downloaded card data folders for Scarlet & Violet and Mega Evolution into the `tcgdex/data/` directory.
    *   Only these and future sets with 3 letter card identification is usable for this method of identification.

3.  **Generate the Lookup File:** Run the `get-card-data.py` script to process the raw data into a single, optimized JSON file (`card_data_lookup.json`) that the main scanning scripts use.

    ```bash
    python3 get-card-data.py
    ```

## 🚀 Usage

### Scanning from Image Files

1.  Place your card images (e.g., `.jpg`, `.png`) into the `photos` folder.
2.  Run the scanning script depending on which OCR engine you installed:

    **Using DocTR + PaddleOCR (recommended - fast with fallback):**
    ```bash
    python3 pictureScan.py
    ```
    
    **Using PaddleOCR only:**
    ```bash
    python3 pictureScan-paddleocr.py
    ```
    
    **Using Surya OCR (slower, alternative):**
    ```bash
    python3 pictureScan-surya.py
    ```

3.  The script will prompt you to:
    *   Use the default `photos` directory or specify a different one
    *   Choose a cropping mode (corner crop, detect crop, or full image)

### Scanning from Webcam

#### Basic Usage (Manual Mode - No Arduino Required)

1.  Ensure you have a webcam connected.
2.  Run the `camScan.py` script:

    ```bash
    python3 camScan.py
    ```

3.  Select **Manual Mode** when prompted.
4.  Position a card in front of the webcam and press **'n'** to scan.
5.  The script will display the identified card information.

#### Advanced Usage (Autonomous Mode with Arduino)

**Hardware Setup:**
*   Arduino board connected via USB (update `PORT` variable in script)
*   DC motor or relay connected to pin 8 (card feeder)
*   Servo motor connected to pin 9 (card sorter)
*   Webcam positioned to capture card area

**Configuration Variables** (in `camScan.py`):
```python
PORT = '/dev/cu.usbserial-1130'        # Arduino serial port
MOTOR_PIN = 8                          # Card feeder motor pin
MOTOR_RUN_TIME = 0.2                   # Motor activation duration (seconds)
SETTLE_TIME = 0.5                      # Wait time for card to settle
SERVO_PIN = 9                          # Card sorter servo pin
SORTER_CENTER_POS = 23                 # Servo center position (degrees)
SORTER_PILE_A_POS = 1                  # Pile A final position
SORTER_PILE_B_POS = 46                 # Pile B final position
SORTER_PILE_A_INTERMEDIATE_POS = 33    # Pile A intermediate position
SORTER_PILE_B_INTERMEDIATE_POS = 11    # Pile B intermediate position
```

**Running Autonomous Mode:**

1.  Connect Arduino and verify port settings
2.  Run the script:
    ```bash
    python3 camScan.py
    ```

3.  Select **Autonomous Mode** when prompted
4.  (Optional) Enter a stop card name to auto-stop after detecting it twice
5.  Choose whether to use the card sorter
6.  The system will:
    - Automatically feed cards one at a time
    - Scan each card with 5 attempts
    - Use wiggle feature if initial scans fail
    - Sort cards into piles based on identification
    - Continue until stopped or stop card is detected

**Sorting Logic:**
*   Customize the `sort_card()` function to define sorting rules
*   Default: sorts by card type (easily modifiable)
*   Failed scans automatically go to Pile B

**Wiggle Feature:**
*   Activates after 5 failed scan attempts
*   Performs 3 additional scan attempts with card repositioning
*   Servo wiggles ±10° from center to improve OCR angle

The script will display the live feed with scan results. DocTR processes cards quickly, with PaddleOCR providing a backup for difficult cases. Press **'q'** to quit at any time.