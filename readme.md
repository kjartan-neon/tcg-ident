# 🃏 tcg-ident: Trading Card Identifier

**tcg-ident** is a Python script designed to automatically detect, crop, flatten, and identify the Set ID and Card Number from physical Trading Card Game (TCG) cards using computer vision and Optical Character Recognition (OCR). It is specifically optimized to handle common issues like perspective distortion and low-contrast borders (e.g., trainer cards) by leveraging a forced 7:5 aspect ratio and aggressive contour detection.

## ✨ Features

* **Robust Card Detection:** Uses HSV saturation and aggressive dilation to find the card, even against complex backgrounds or low-contrast borders.
* **Perspective Correction:** Applies a four-point perspective transform to flatten the card, regardless of the angle in the input photo.
* **Forced Aspect Ratio:** Guarantees a clean, rectangular crop by enforcing the standard 7:5 TCG aspect ratio.
* **Targeted OCR:** Crops to the specific bottom-left area of the card (where set IDs and numbers are located) for faster and more accurate OCR using PaddleOCR.
* **Detailed Debugging Output:** Generates intermediate image files to help visualize every step of the detection and cropping pipeline.

## ⚙️ Prerequisites

You must have Python 3 installed. This script relies on several external libraries:

* **OpenCV (`cv2`):** For all computer vision tasks (contour detection, warping, cropping).
* **NumPy:** For efficient array manipulation.
* **PaddlePaddle/PaddleOCR:** For the OCR engine.

### Installation

1.  **Install Python Libraries:** It is highly recommended to use a virtual environment.

    ```bash
	python3 -m venv env
	source env/bin/activate
    	pip install opencv-python numpy paddlepaddle paddleocr
    ```
    *(Note: PaddleOCR will automatically download necessary models on first run.)*

## 🚀 Usage

### 1. Setup

1.  Save the provided Python code as `get-set-full.py`.
2.  Create a folder named `photos` in the same directory as the script.
3.  Place all your TCG card images (JPG, JPEG, PNG) into the `photos` folder.

### 2. Running the Script

Execute the script from your terminal:

```bash
python3 get-set-full.py

### 3. Raspberry pi 5

* python3 -m venv env
* source env/bin/activate
* pip install paddlepaddle
* python3 -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
* pip install "paddleocr>=2.8.1"

The error 
--------------------------------------
C++ Traceback (most recent call last):
--------------------------------------
No stack trace in paddle, may be caused by external reasons.

----------------------
Error Message Summary:
----------------------
FatalError: `Segmentation fault` is detected by the operating system.
  [TimeInfo: *** Aborted at 1764325698 (unix time) try "date -d @1764325698" if you are using GNU date ***]
  [SignalInfo: *** SIGSEGV (@0x0) received by PID 21710 (TID 0x7fff359ff160) from PID 0 ***]

Segmentation fault

