import cv2
import numpy as np
import os
import sys
import time
from paddleocr import PaddleOCR
import pyfirmata
import serial

# --- Configuration ---
DEBUG_OUTPUT_FOLDER = "debug_output_webcam"
WEBCAM_INDEX = 0          # Usually 0 for the default camera
WAIT_TIME_NO_TEXT = 3.0   # Wait time if OCR fails

# --- Arduino/PyFirmata Configuration ---
# IMPORTANT: Change this to your Arduino's serial port (e.g., 'COM3', '/dev/ttyACM0')
PORT = '/dev/ttyACM0' 
MOTOR_PIN = 8 
MOTOR_RUN_TIME = 0.5      # Seconds the motor runs to move the card
SETTLE_TIME = 0.5         # Time to wait after motor stops for card vibration to settle

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
    
    # --- 1. Define Fixed Crop Area (Bottom-Left: 50% H, 30% W) ---
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


# --- OCR AND HEURISTIC FUNCTIONS (UNCHANGED) ---

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

# --- PYFIRMATA CONTROL FUNCTION (UNCHANGED) ---

def run_card_feeder(board):
    """Activates the motor pin for a set duration to feed the next card."""
    print("Commanding Arduino: Feeding next card...")
    
    try:
        # 1. Activate the motor (set pin HIGH)
        board.digital[MOTOR_PIN].write(1)
        
        # 2. Let the motor run
        time.sleep(MOTOR_RUN_TIME)
        
        # 3. Stop the motor (set pin LOW)
        board.digital[MOTOR_PIN].write(0)
        
        # 4. Wait for the card vibration to stop before scanning
        time.sleep(SETTLE_TIME) 
        print("Card positioned. Ready for scan.")
        return True
    except Exception as e:
        print(f"ERROR: Could not communicate with Arduino motor pin {MOTOR_PIN}: {e}")
        return False

# --- MAIN EXECUTION START ---

if __name__ == "__main__":
    
    print("Initializing PaddleOCR...")
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu') 
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Check your PaddleOCR installation and dependencies.")
        sys.exit(1)

    # --- 1. Webcam Setup ---
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open webcam index {WEBCAM_INDEX}.")
        sys.exit(1)
        
    cv2.namedWindow('TCG-ident Live', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('TCG-ident Live', 800, 600)
    
    # --- 2. Mode Selection ---
    print("\n--- TCG-IDENT MODE SELECTION ---")
    
    valid_choice = False
    autonomous_mode = False
    
    while not valid_choice:
        choice = input("Select Mode: (A)utonomous via Arduino / (M)anual 'n' trigger: ").upper()
        if choice == 'A':
            autonomous_mode = True
            valid_choice = True
        elif choice == 'M':
            autonomous_mode = False
            valid_choice = True
        else:
            print("Invalid choice. Please enter 'A' or 'M'.")

    # --- 3. PyFirmata/Arduino Connection Attempt (only if needed) ---
    board = None
    if autonomous_mode:
        try:
            print(f"Attempting connection to Arduino on {PORT}...")
            board = pyfirmata.Arduino(PORT)
            board.digital[MOTOR_PIN].mode = pyfirmata.OUTPUT
            print("✅ Arduino connection successful.")
        except (serial.SerialException, pyfirmata.BoardNotFound, AttributeError) as e:
            print(f"❌ Could not connect to Arduino on port {PORT}. Exiting.")
            cap.release()
            sys.exit(1)

    print(f"\nMode Selected: {'AUTONOMOUS' if autonomous_mode else 'MANUAL'}.")
    
    scan_count = 0
    should_feed_next_card = autonomous_mode # Start by feeding if in autonomous mode

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # --- Display Live Feed Status ---
        mode_text = "AUTONOMOUS (Press 'q' to quit)" if autonomous_mode else "MANUAL (Press 'n' to scan, 'q' to quit)"
        cv2.putText(frame, mode_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('TCG-ident Live', frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        trigger_scan = False
        
        # --- Manual Trigger Check ---
        if not autonomous_mode and key == ord('n'):
            trigger_scan = True

        # --- Autonomous Trigger Check ---
        if autonomous_mode:
            
            # 1. Feed the card if needed
            if should_feed_next_card:
                if not run_card_feeder(board):
                    print("Exiting due to feeder error.")
                    break
                should_feed_next_card = False # Card is now in position
                time.sleep(0.1) 
                
                # Grab a fresh frame after the card has settled
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame after feed.")
                    break
            
            # If we just fed the card, trigger the scan on the new frame
            trigger_scan = True 

        # --- SCAN EXECUTION ---
        if trigger_scan:
            print(f"\nSCAN TRIGGERED ({'Auto' if autonomous_mode else 'Manual'})...")
            
            filename_prefix = f"scan_{scan_count:03d}_{int(time.time())}" 
            img_for_ocr = preprocess_card_image_simple(frame, debug_folder=DEBUG_OUTPUT_FOLDER, filename_prefix=filename_prefix)
            
            # CRITICAL FIX: Create a deep copy to prevent Segmentation Fault
            if img_for_ocr is not None:
                img_for_ocr_safe = img_for_ocr.copy() 
            else:
                img_for_ocr_safe = None
                
            result = identify_card_info(img_for_ocr_safe, ocr)
            
            if "FAILED" not in result:
                # --- SUCCESS ---
                print(f"✅ IDENTIFIED: {result}")
                
                # Show result briefly on the live frame
                cv2.putText(frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('TCG-ident Live', frame)
                cv2.waitKey(100) # Show result briefly
                
                scan_count += 1
                
                if autonomous_mode:
                    should_feed_next_card = True # Ready to feed the next one
                
            else:
                # --- FAILURE ---
                print(f"❌ SCAN FAILED: {result}.")
                
                if autonomous_mode:
                    print("Retrying scan on the SAME card after 1 second.")
                    time.sleep(1)
                else:
                    # In manual mode, wait for the user to press 'n' again
                    pass
                
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if board:
        board.exit()
