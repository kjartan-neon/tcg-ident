import cv2
import numpy as np
import os
import sys
import time
from paddleocr import PaddleOCR

# --- Configuration ---
DEBUG_OUTPUT_FOLDER = "debug_output_webcam"
WEBCAM_INDEX = 0      # Usually 0 for the default camera
WAIT_TIME_NO_CARD_DETECTED = 3.0 # Wait 3 seconds if no text is found

# --- SIMPLE PREPROCESSING FUNCTION (UNCHANGED) ---

def preprocess_card_image_simple(img, debug_folder=None, filename_prefix="capture"):
    """
    Performs a fixed crop (bottom 50% height, left 30% width) directly on the frame.
    No contour detection or perspective correction is performed.
    Returns the final cropped image for OCR.
    """
    if img is None:
        return None
    
    (h_orig, w_orig) = img.shape[:2]
    
    # --- 1. Define Fixed Crop Area (Bottom-Left) ---
    crop_height_perc = 0.50 
    crop_width_perc = 0.30  
    
    y_start_crop = int(h_orig * (1 - crop_height_perc))
    x_start_crop = 0 
    x_end_crop = int(w_orig * crop_width_perc)
    
    final_crop = img[y_start_crop:h_orig, x_start_crop:x_end_crop]

    # --- Debugging: Save the final processed image (OCR Input) ---
    if debug_folder:
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        
        debug_final_crop_filename = os.path.join(debug_folder, f"{filename_prefix}_processed_final_crop.jpg")
        cv2.imwrite(debug_final_crop_filename, final_crop)
        
    return final_crop


# --- OCR AND HEURISTIC FUNCTIONS (FIXED) ---

def identify_card_info(img_for_ocr, ocr_engine):
    """
    Runs OCR and heuristic extraction on the cropped image.
    FIXED: Robustly handles "SET EN" and "NUMBER/TOTAL" formats.
    """
    if img_for_ocr is None:
        return "--- FAILED: No Crop Input ---"

    result = ocr_engine.predict(img_for_ocr)

    set_id = None
    card_number = None
    detected_texts = []

    if result and isinstance(result, list) and len(result) > 0:
        ocr_result_object = result[0]
        if hasattr(ocr_result_object, 'rec_texts'):
            detected_texts = ocr_result_object.rec_texts
        elif isinstance(ocr_result_object, dict) and 'rec_texts' in ocr_result_object:
            detected_texts = ocr_result_object['rec_texts']
    
    # --- DEBUG: Output all found text ---
    print("\n--- DEBUG: Raw OCR Text Output ---")
    if detected_texts:
        for text in detected_texts:
            print(f"[RAW OCR]: {text}")
    else:
        print("[RAW OCR]: No text detected.")
    print("---------------------------------")
    # --- END DEBUG ---
    
    for text in detected_texts:
        text = str(text).upper().strip()
        
        # --- Heuristic 1: Look for potential Set IDs (Robust to "XXX EN") ---
        # Look for text with letters and optionally space/language code
        if text.isalpha() or ' ' in text:
            parts = text.split()
            potential_set_id = ""

            # Case: TWM EN
            if len(parts) >= 2 and parts[-1] in ['EN', 'FR', 'JP', 'DE', 'IT', 'POL']: 
                potential_set_id = parts[0]
            # Case: TWM (only letters)
            elif len(parts) == 1 and 2 <= len(parts[0]) <= 5 and parts[0].isalpha(): 
                potential_set_id = parts[0]

            # Final check and assignment
            if potential_set_id and 2 <= len(potential_set_id) <= 5 and potential_set_id not in ['NO', 'TM']: 
                set_id = potential_set_id
        
        # --- Heuristic 2: Look for potential Card Numbers (Robust to "123/456") ---
        number_part = None
        
        if '/' in text:
            # Handles "153/167"
            parts = text.split('/')
            if parts and parts[0].isdigit():
                 number_part = parts[0]
        elif text.isdigit() and len(text) <= 4:
            # Handles standalone numbers like "153"
            number_part = text
        
        if number_part is not None and number_part.isdigit():
            card_number = number_part
            # Exit the loop early if both pieces are found
            if card_number and set_id: 
                break 

    if set_id and card_number:
        return f"{set_id}-{card_number}"
    else:
        # Include found parts in the failure message for better debugging
        return f"--- FAILED: OCR Heuristic (Set ID: {set_id}, Number: {card_number}) ---"

# --- MAIN WEBCAM EXECUTION LOOP (UNCHANGED) ---

if __name__ == "__main__":
    
    print("Initializing PaddleOCR...")
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu') 
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Check your PaddleOCR installation and dependencies.")
        sys.exit(1)

    WEBCAM_INDEX = 0 
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam index {WEBCAM_INDEX}.")
        sys.exit(1)
        
    cv2.namedWindow('TCG-ident Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TCG-ident Live', 800, 600)

    print("\n--- TCG-IDENT LIVE MODE (FIXED CROP) ---")
    print("Camera should be focused on the bottom-left area of the card.")
    print(f"Processing ONLY the bottom 50% height and left 30% width of the frame.")
    print("Press 'n' to scan the current frame.")
    print("Press 'q' to quit.")
    
    scan_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Display live feed
        cv2.putText(frame, "PRESS 'n' TO SCAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('TCG-ident Live', frame)

        key = cv2.waitKey(1) & 0xFF
        
        # 1. Quit check
        if key == ord('q'):
            break

        # 2. Manual trigger check ('n')
        if key == ord('n'):
            print("\nSCAN TRIGGERED (Key 'n' pressed)...")
            
            filename_prefix = f"scan_{scan_count:03d}_{int(time.time())}" 
            
            # --- RUN SIMPLE CROP ---
            img_for_ocr = preprocess_card_image_simple(frame, debug_folder=DEBUG_OUTPUT_FOLDER, filename_prefix=filename_prefix)
            
            # Since we always get a crop, we assume the input is good and run OCR
            result = identify_card_info(img_for_ocr, ocr)
            
            if "FAILED" not in result:
                # --- SUCCESS ---
                print(f"✅ IDENTIFIED: {result}")
                
                # Show result on the live frame
                cv2.putText(frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('TCG-ident Live', frame)

                # Show the cropped area to confirm the OCR input
                cv2.imshow('OCR Input Crop', img_for_ocr)
                
                # Wait for user to review before continuing
                cv2.waitKey(0) 
                cv2.destroyWindow('OCR Input Crop')

                scan_count += 1
                
            else:
                # --- FAILURE ---
                print(f"❌ SCAN FAILED: {result}. Waiting {WAIT_TIME_NO_CARD_DETECTED}s and retrying...")
                
                start_time = time.time()
                while (time.time() - start_time) < WAIT_TIME_NO_CARD_DETECTED:
                    
                    ret_auto, frame_auto = cap.read()
                    if not ret_auto:
                        break

                    countdown = int(WAIT_TIME_NO_CARD_DETECTED - (time.time() - start_time)) + 1
                    status_text = f"OCR FAILED! Retrying in {countdown}s..."
                    
                    frame_display = frame_auto.copy()
                    cv2.putText(frame_display, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('TCG-ident Live', frame_display)
                    
                    key_inner = cv2.waitKey(50) & 0xFF
                    if key_inner == ord('q'):
                        key = ord('q') 
                        break
                
                if key == ord('q'):
                    break
                
    cap.release()
    cv2.destroyAllWindows()