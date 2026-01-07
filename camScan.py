import cv2
import numpy as np
import os
import sys
import time
from paddleocr import PaddleOCR
from ocr_processing import extract_card_info_from_text
import pyfirmata2 as pyfirmata
import serial
import json

# --- Configuration ---
DEBUG_OUTPUT_FOLDER = "debug_output_webcam"
WEBCAM_INDEX = 0          # Usually 0 for the default camera
WAIT_TIME_NO_TEXT = 3.0   # Wait time if OCR fails

# --- Arduino/PyFirmata Configuration ---
# IMPORTANT: Change this to your Arduino's serial port (e.g., 'COM3', '/dev/ttyACM0')
# MUST USE pip install pyfirmata2 on newer pythons
PORT = '/dev/cu.usbserial-120' 
MOTOR_PIN = 8 
MOTOR_RUN_TIME = 0.2      # Seconds the motor runs to move the card
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

def identify_card_info(img_for_ocr, ocr_engine, card_database=None):
    """
    Runs OCR and heuristic extraction on the cropped image.
    """
    if img_for_ocr is None:
        return "--- FAILED: No Crop Input ---"

    result = ocr_engine.predict(img_for_ocr)

    detected_texts = []
    if result and isinstance(result, list) and len(result) > 0:
        ocr_result_object = result[0]
        if hasattr(ocr_result_object, 'rec_texts'):
            detected_texts = ocr_result_object.rec_texts
        elif isinstance(ocr_result_object, dict) and 'rec_texts' in ocr_result_object:
            detected_texts = ocr_result_object['rec_texts']
    
    # Use the shared function for text processing
    return extract_card_info_from_text(detected_texts, card_database)

# --- PYFIRMATA CONTROL FUNCTION (UNCHANGED) ---

def run_card_feeder(board):
    """Activates the motor pin for a set duration to feed the next card."""
    print("Commanding Arduino: Feeding next card...")
    
    try:
        # 1. Activate the motor (set pin high)
        board.digital[MOTOR_PIN].write(1)
        
        # 2. Let the motor run
        time.sleep(MOTOR_RUN_TIME)
        
        # 3. Stop the motor (set pin low)
        board.digital[MOTOR_PIN].write(0)
        
        # 4. Wait for the card vibration to stop before scanning
        time.sleep(SETTLE_TIME) 
        print("Card positioned. Ready for scan.")
        return True
    except Exception as e:
        print(f"ERROR: Could not communicate with Arduino motor pin {MOTOR_PIN}: {e}")
        return False

# --- OFFLINE PROCESSING FUNCTION ---

def process_images_from_folder(folder_path, ocr_engine, card_database=None):
    """
    Processes all images from a specified folder instead of using the webcam.
    """
    print(f"\n--- STARTING OFFLINE PROCESSING ---")
    print(f"Source Folder: {folder_path}")

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"❌ ERROR: Folder not found or is not a directory: {folder_path}")
        return

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("❌ No images found in the folder to process.")
        return

    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    for i, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        print(f"\n--- Processing image {i+1}/{total_images}: {filename} ---")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"⚠️  Could not read image: {filename}")
            continue

        # Use a prefix based on the original filename for debug outputs
        filename_prefix = os.path.splitext(filename)[0]
        
        img_for_ocr = preprocess_card_image_simple(frame, debug_folder=DEBUG_OUTPUT_FOLDER, filename_prefix=f"{filename_prefix}_offline")
        
        result = identify_card_info(img_for_ocr, ocr_engine, card_database)
        print(f"➡️  Result for {filename}: {result}")

    print("\n--- OFFLINE PROCESSING COMPLETE ---")
# --- MAIN EXECUTION START ---

if __name__ == "__main__":
    
    print("Initializing PaddleOCR...")
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en') 
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Check your PaddleOCR installation and dependencies.")
        sys.exit(1)

    # --- Load Card Database ---
    card_database = None
    try:
        with open('card_data_lookup.json', 'r') as f:
            card_database = json.load(f)
        print("Successfully loaded card database.")
    except FileNotFoundError:
        print("Warning: 'card_data_lookup.json' not found. Card names will not be looked up.")
    except json.JSONDecodeError:
        print("Error: Could not decode 'card_data_lookup.json'. File might be corrupted.")
        card_database = None

    # --- 0. Ensure debug folder exists ---
    if not os.path.exists(DEBUG_OUTPUT_FOLDER):
        os.makedirs(DEBUG_OUTPUT_FOLDER)
        print(f"Created debug output folder: {DEBUG_OUTPUT_FOLDER}")

    # --- NEW: Ask for offline processing mode ---
    process_offline = False
    while True:
        offline_choice = input("Process images from folder instead of webcam? (y/n): ").lower()
        if offline_choice == 'y':
            process_offline = True
            break
        elif offline_choice == 'n':
            break
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

    if process_offline:
        process_images_from_folder(DEBUG_OUTPUT_FOLDER, ocr, card_database)
        sys.exit(0)

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
            board = pyfirmata.Arduino(PORT) # This will now use pyfirmata2
            board.digital[MOTOR_PIN].mode = pyfirmata.OUTPUT # This will also use pyfirmata2
            board.digital[MOTOR_PIN].write(0) # set board low
            print("✅ Arduino connection successful.")
        except serial.SerialException as e:
            print(f"❌ Could not connect to Arduino on port {PORT}. Exiting.")
            cap.release()
            sys.exit(1)

    print(f"\nMode Selected: {'AUTONOMOUS' if autonomous_mode else 'MANUAL'}.")
    
    scan_count = 0
    should_feed_next_card = autonomous_mode # Start by feeding if in autonomous mode
    frame_save_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # --- FEATURE: Save every full frame for debugging ---
        frame_save_path = os.path.join(DEBUG_OUTPUT_FOLDER, f"webcam_frame_{frame_save_counter:05d}.jpg")
        cv2.imwrite(frame_save_path, frame)
        frame_save_counter += 1
        # ---
        
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
                
            result = identify_card_info(img_for_ocr_safe, ocr, card_database)
            
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
