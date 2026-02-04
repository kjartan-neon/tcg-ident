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
PORT = '/dev/cu.usbserial-1130'
MOTOR_PIN = 8 
MOTOR_RUN_TIME = 0.2      # Seconds the motor runs to move the card
SETTLE_TIME = 0.5         # Time to wait after motor stops for card vibration to settle
SERVO_PIN = 9
SORTER_CENTER_POS = 23
SORTER_PILE_A_POS = 1
SORTER_PILE_B_POS = 46
SORTER_SETTLE_TIME = 0.5
SORTER_PILE_A_INTERMEDIATE_POS = 33
SORTER_PILE_B_INTERMEDIATE_POS = 11
SORTER_INTERMEDIATE_SETTLE_TIME = 0.2

# --- SIMPLE PREPROCESSING FUNCTION  ---

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


# --- OCR AND HEURISTIC FUNCTIONS  ---

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

# --- PYFIRMATA CONTROL FUNCTION  ---

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

def wiggle_card(servo):
    """Wiggles the servo slightly to try and improve OCR results."""
    print("Wiggling card...")
    try:
        # Assumes the servo is at the center position.
        # Wiggle back and forth by 10 degrees from center.
        center_pos = SORTER_CENTER_POS
        pos1 = center_pos - 10
        pos2 = center_pos + 10
        
        # Clamp values to be within 0-180 servo range
        pos1 = max(0, min(180, pos1))
        pos2 = max(0, min(180, pos2))

        # Perform the wiggle
        servo.write(pos1)
        time.sleep(0.3)
        servo.write(pos2)
        time.sleep(0.3)
        
        # Return to center
        servo.write(center_pos)
        time.sleep(SORTER_SETTLE_TIME)
        print("Wiggle complete. Returned to center.")
    except Exception as e:
        print(f"ERROR: Could not wiggle servo: {e}")

def sort_card(servo, result_string):
    """Moves the servo to sort the card based on its type with an intermediate step."""
    print(f"Sorting card based on result: {result_string}")
    
    # Default to Pile B unless specific conditions for Pile A are met.
    is_pile_a = False

    # Explicit check for forced ejection to Pile B
    if 'force_eject_pile_b' in result_string:
        print("Forcing eject to Pile B due to scan failure.")
        is_pile_a = False
    # Check for 'fighting' type for Pile A sorting
    elif 'stadium' in result_string.lower():
        print(f"Card is a 'fighting' type. Sorting to Pile A.")
        is_pile_a = True
    else:
        print(f"Card is not a 'fighting' type. Defaulting to Pile B.")
        is_pile_a = False

    if is_pile_a:
        print(f"Card is a 'fighting' type. Sorting to Pile A.")
        print(f"  - Moving to intermediate position: {SORTER_PILE_A_INTERMEDIATE_POS} degrees.")
        servo.write(SORTER_PILE_A_INTERMEDIATE_POS)
        time.sleep(SORTER_INTERMEDIATE_SETTLE_TIME)
        print(f"  - Moving to final position: {SORTER_PILE_A_POS} degrees.")
        servo.write(SORTER_PILE_A_POS)
    else: # Sort to Pile B
        print(f"Card is not a 'fighting' type or is a failure. Sorting to Pile B.")
        print(f"  - Moving to intermediate position: {SORTER_PILE_B_INTERMEDIATE_POS} degrees.")
        servo.write(SORTER_PILE_B_INTERMEDIATE_POS)
        time.sleep(SORTER_INTERMEDIATE_SETTLE_TIME)
        print(f"  - Moving to final position: {SORTER_PILE_B_POS} degrees.")
        servo.write(SORTER_PILE_B_POS)
    time.sleep(SORTER_SETTLE_TIME)


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

    # --- 3. Stop Card Configuration (only if autonomous mode) ---
    stop_card_name = None
    consecutive_stop_card_count = 0
    if autonomous_mode:
        stop_card_input = input("Enter stop card name (Pokemon name, leave empty to skip): ").strip()
        if stop_card_input:
            stop_card_name = stop_card_input
            print(f"Stop card set to: '{stop_card_name}'. Script will exit after detecting it twice in a row.")
    
    # --- 4. PyFirmata/Arduino Connection Attempt (only if needed) ---
    board = None
    servo = None
    use_sorter = False
    if autonomous_mode:
        while True:
            sorter_choice = input("Use card sorter? (y/n): ").lower()
            if sorter_choice == 'y':
                use_sorter = True
                break
            elif sorter_choice == 'n':
                break
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")
        try:
            print(f"Attempting connection to Arduino on {PORT}...")
            board = pyfirmata.Arduino(PORT) # This will now use pyfirmata2
            board.digital[MOTOR_PIN].mode = pyfirmata.OUTPUT # This will also use pyfirmata2
            board.digital[MOTOR_PIN].write(0) # set board low
            if use_sorter:
                servo = board.get_pin(f'd:{SERVO_PIN}:s')
                servo.write(SORTER_CENTER_POS)
                time.sleep(SORTER_SETTLE_TIME)
            print("✅ Arduino connection successful.")
        except serial.SerialException as e:
            print(f"❌ Could not connect to Arduino on port {PORT}. Exiting. Error: {e}")
            cap.release()
            if board:
                board.exit()
            sys.exit(1)
        except Exception as e:
            print(f"❌ An error occurred during Arduino setup: {e}")
            cap.release()
            if board:
                board.exit()
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
                if use_sorter:
                    print(f"Setting servo to center position ({SORTER_CENTER_POS} degrees)...")
                    servo.write(SORTER_CENTER_POS)
                    time.sleep(SORTER_SETTLE_TIME)
                
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
            
            result = "FAILED" # Default to failed

            # --- Initial Scan Loop (5 retries) ---
            for attempt in range(5):
                scan_retries = attempt + 1
                print(f"Scanning card... (Attempt {scan_retries}/5)")

                # We need a fresh frame for each attempt
                ret, current_frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame for scan attempt.")
                    time.sleep(1)
                    continue # Try again

                # Update the live view with the frame being processed
                cv2.imshow('TCG-ident Live', current_frame)
                cv2.waitKey(1)

                filename_prefix = f"scan_{scan_count:03d}_{int(time.time())}_attempt_{scan_retries}" 
                img_for_ocr = preprocess_card_image_simple(current_frame, debug_folder=DEBUG_OUTPUT_FOLDER, filename_prefix=filename_prefix)
                
                if img_for_ocr is not None:
                    img_for_ocr_safe = img_for_ocr.copy() 
                else:
                    img_for_ocr_safe = None
                    
                result = identify_card_info(img_for_ocr_safe, ocr, card_database)
                
                if "FAILED" not in result:
                    break # Success, exit the retry loop
                else:
                    print(f"Scan attempt {scan_retries} failed. Result: {result}")
                    if attempt < 4: # Don't sleep on the last attempt
                        time.sleep(1)

            # --- Wiggle and Retry Loop (if initial scan failed) ---
            if "FAILED" in result and autonomous_mode and use_sorter:
                for attempt in range(3):
                    wiggle_retries = attempt + 1
                    print(f"\nInitial scans failed. Wiggling card and retrying... (Wiggle Attempt {wiggle_retries}/3)")
                    
                    # Wiggle the card
                    wiggle_card(servo)
                    time.sleep(0.5) # Settle time

                    # Grab fresh frame and scan again
                    ret, current_frame = cap.read()
                    if not ret:
                        print("Error: Failed to grab frame for wiggle-scan attempt.")
                        time.sleep(1)
                        continue

                    # Update the live view
                    cv2.imshow('TCG-ident Live', current_frame)
                    cv2.waitKey(1)

                    filename_prefix = f"scan_{scan_count:03d}_{int(time.time())}_wiggle_{wiggle_retries}" 
                    img_for_ocr = preprocess_card_image_simple(current_frame, debug_folder=DEBUG_OUTPUT_FOLDER, filename_prefix=filename_prefix)
                    
                    if img_for_ocr is not None:
                        img_for_ocr_safe = img_for_ocr.copy()
                    else:
                        img_for_ocr_safe = None
                    
                    result = identify_card_info(img_for_ocr_safe, ocr, card_database)

                    if "FAILED" not in result:
                        break # Success, exit the wiggle loop
                    else:
                        print(f"Wiggle-scan attempt {wiggle_retries} failed. Result: {result}")
                        if attempt < 2: # Don't sleep on last attempt
                            time.sleep(1)

            # --- Final Result Processing ---
            if "FAILED" not in result:
                # --- SUCCESS ---
                print(f"✅ IDENTIFIED: {result}")
                
                # --- Check for Stop Card ---
                if autonomous_mode and stop_card_name:
                    # Extract the card name from the result string to check for exact match
                    if stop_card_name.lower() in result.lower():
                        consecutive_stop_card_count += 1
                        print(f"⚠️  Stop card detected! Count: {consecutive_stop_card_count}/2")
                        
                        if consecutive_stop_card_count >= 2:
                            print(f"🛑 Stop card '{stop_card_name}' detected twice in a row. Exiting script.")
                            # Sort the current card first if using sorter
                            if use_sorter:
                                sort_card(servo, result)
                            break  # Exit the main loop
                    else:
                        # Reset counter if a different card is detected
                        consecutive_stop_card_count = 0
                
                if autonomous_mode and use_sorter:
                    sort_card(servo, result)
                
                # Show result briefly on the live frame
                success_frame = frame.copy()
                cv2.putText(success_frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow('TCG-ident Live', success_frame)
                cv2.waitKey(2000) # Show result for 2 seconds
                
                scan_count += 1
                
                if autonomous_mode:
                    should_feed_next_card = True # Ready to feed the next one
                
            else:
                # --- FINAL FAILURE ---
                print(f"❌ ALL SCAN ATTEMPTS FAILED. Last result: {result}.")
                
                # Reset stop card counter on failed scans
                if autonomous_mode and stop_card_name:
                    consecutive_stop_card_count = 0
                
                if autonomous_mode:
                    print("Ejecting card to Pile B.")
                    if use_sorter:
                        sort_card(servo, "force_eject_pile_b") # Force eject to B
                    else:
                        # If no sorter, we still need to move on
                        print("Sorter not in use, cannot eject. Moving to next card anyway.")
                    
                    should_feed_next_card = True # Give up and move on.
                else:
                    # In manual mode, we just report failure and wait for 'n' again.
                    print("Scan failed. Press 'n' to try again.")
                    pass
                
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if board:
        board.exit()
