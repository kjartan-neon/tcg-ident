import easyocr
import os

# --- Configuration ---
# Set the language (e.g., English)
# The reader will automatically download models (if not cached)
reader = easyocr.Reader(['en']) 

# Replace this with the path to your image
IMAGE_PATH = 'photos/20250423_062747.jpg' 

# --- Execution ---
print(f"--- Starting EasyOCR on {IMAGE_PATH} ---")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}.")
else:
    try:
        # Run the OCR pipeline
        results = reader.readtext(IMAGE_PATH)
        
        # --- Result Parsing & Output ---
        for (bbox, text, conf) in results:
            print(f"Detected Text: {text}")
            print(f"  - Confidence: {conf:.4f}")

        print("--- EasyOCR Test Complete ---")

    except Exception as e:
        print(f"An error occurred during OCR: {e}")
