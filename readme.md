# 🃏 TCG Identifier

**TCG Identifier** is a Python-based tool for automatically identifying Pokémon Trading Card Game (TCG) cards from images or a live webcam feed. It uses computer vision to detect and isolate the card from its background and Optical Character Recognition (OCR) to extract key information like the set ID and card number. It supports all sets that have a *3 letter identification*, and uses OCR to find the card's name after looking up in a comprehensive card database.


## ✨ Features

*   **Robust Card Detection:** Locates cards in images, even against complex backgrounds.
*   **Perspective Correction:** Applies a four-point perspective transform to "flatten" the card for accurate analysis.
*   **Targeted OCR:** Crops the card image to the bottom section where the set ID and number are typically located, improving OCR accuracy.
*   **Card Verification:** Looks up the extracted set ID and card number in a generated JSON database to find the card's name.
*   **Multiple Scan Modes:**
    *   `pictureScan.py`: Scans a directory of card images.
    *   `camScan.py`: Scans for cards using a live webcam feed.

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

### 3. Install PaddleOCR

This project uses PaddleOCR for its text recognition capabilities. You must install the PaddlePaddle deep learning framework and the PaddleOCR library.

```bash
# Install PaddlePaddle (CPU version recommended for simplicity)
pip install paddlepaddle

# Install PaddleOCR
pip install paddleocr
```
*Note: PaddleOCR will automatically download the necessary detection and recognition models the first time it runs.*

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
2.  Run the `pictureScan.py` script. It will prompt you to use the default `photos` directory or specify a different one.

    ```bash
    python3 pictureScan.py
    ```

### Scanning from Webcam

1.  Ensure you have a webcam connected.
2.  Run the `camScan.py` script.

    ```bash
    python3 camScan.py
    ```

The script will display the live feed, and when it identifies a card, it will attempt to extract its information and display the result.