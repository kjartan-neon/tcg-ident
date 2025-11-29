import paddle
from paddleocr import PaddleOCR
import os

# --- 1. CONFIGURATION ---
# Set the language to English ('en'). This will automatically download
# the necessary English-optimized models (detection, recognition) the first time.
# The 'use_angle_cls=True' is generally recommended for better accuracy.
ocr = PaddleOCR(lang='en', show_log=False) 

# Replace this with the path to your image
IMAGE_PATH = 'photos/20250423_062747.jpg' 

# --- 2. EXECUTION ---
print(f"--- Starting OCR on {IMAGE_PATH} ---")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}. Please provide a valid image path.")
else:
    try:
        # Run the OCR pipeline (detection + recognition)
        result = ocr.ocr(IMAGE_PATH, cls=True)
        
        # --- 3. RESULT PARSING & OUTPUT ---
        # The result is a nested list. We flatten it to get the text.
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                # 'line' format is: [bounding_box, (text, confidence)]
                bounding_box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                print(f"Detected Text: {text}")
                # Optional: Print the location and confidence
                # print(f"  - BBox: {bounding_box}") 
                # print(f"  - Conf: {confidence:.4f}")

        print("--- OCR Test Complete ---")

    except Exception as e:
        print(f"An error occurred during OCR: {e}")
