import cv2
import numpy as np
import os
import glob
import sys
from paddleocr import PaddleOCR

# --- Configuration ---
CARD_FOLDER = "photos" 
SUPPORTED_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']
CARD_ASPECT_RATIO = 1.4 # Height/Width ratio for a standard Pokemon card (7/5 = 1.4)

# --- Computer Vision Preprocessing Functions ---

def order_points(pts):
    """
    Orders the four points of a quadrilateral: top-left, top-right, 
    bottom-right, and bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts, target_width, target_height):
    """
    Applies the perspective transform to get a top-down, flat view of the card.
    """
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped

def preprocess_card_image(image_path, target_width=800, target_height=1120, debug_output_folder=None):
    """
    Locates, crops, flattens, and reduces resolution of a card from a full image.
    Uses HSV Saturation isolation, aggressive dilation, and FORCES the 7:5 card aspect ratio for warping.
    Includes robust debug image saving at each stage.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    original_img = img.copy() 
    (h_orig, w_orig) = img.shape[:2]
    max_input_dim = 1000 
    
    # Base name for debug files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Ensure debug output folder exists
    if debug_output_folder and not os.path.exists(debug_output_folder):
        os.makedirs(debug_output_folder)

    # Initial resize
    if max(h_orig, w_orig) > max_input_dim:
        scale_factor = max_input_dim / max(h_orig, w_orig)
        img = cv2.resize(img, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)
    else:
        scale_factor = 1.0
        
    # 1. HSV Saturation Isolation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(saturation, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # --- Aggressive Dilation and Erosion ---
    dilate_kernel_large = np.ones((25, 25), np.uint8) 
    thresh = cv2.dilate(thresh, dilate_kernel_large, iterations=2) 
    
    erode_kernel_small = np.ones((7, 7), np.uint8)
    thresh = cv2.erode(thresh, erode_kernel_small, iterations=1)
    
    # Clean up remaining noise
    kernel_close_open = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close_open)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_close_open) 

    # --- Debugging: Save the thresholded image ---
    if debug_output_folder:
        try:
            debug_thresh_filename = os.path.join(debug_output_folder, f"{base_name}_threshold.jpg")
            cv2.imwrite(debug_thresh_filename, thresh)
            print(f"Debug: Saved thresholded image to '{debug_thresh_filename}'")
        except Exception as e:
            print(f"Error saving debug threshold image: {e}")

    # 2. Contour Detection
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Find the Largest, Card-like Contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    min_card_area = (img.shape[0] * img.shape[1]) * 0.05 
    
    largest_card_contour = None
    max_contour_area = -1

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_card_area: # Skip very small contours
            continue
            
        # Get minimum enclosing rectangle (rotated)
        rect = cv2.minAreaRect(c)
        (center, (width, height), angle) = rect

        # Normalize width and height to always have height >= width
        if width > height:
            width, height = height, width
        
        if width == 0 or height == 0:
            continue

        current_aspect_ratio = height / width
        
        # Check if aspect ratio is close to card's (with a generous tolerance)
        if (CARD_ASPECT_RATIO * 0.8 < current_aspect_ratio < CARD_ASPECT_RATIO * 1.2) and (area > max_contour_area):
            max_contour_area = area
            largest_card_contour = c

    if largest_card_contour is None:
        print("Warning: Could not locate a significant, card-like contour with a reasonable aspect ratio.")
        return None

    # Use minAreaRect on the *chosen* largest_card_contour
    rect = cv2.minAreaRect(largest_card_contour)
    (center, (width, height), angle) = rect

    # Ensure height is the longer dimension
    if width > height:
        width, height = height, width
        angle += 90
        
    # Calculate the ideal width and height based on the detected size and the 7:5 ratio
    ideal_height = height
    ideal_width = ideal_height / CARD_ASPECT_RATIO

    # Add padding
    padding_factor = 1.04 
    ideal_height *= padding_factor
    ideal_width *= padding_factor
    
    # Construct the 4 points (box) based on forced aspect ratio
    rect_forced = (center, (ideal_width, ideal_height), angle)
    card_contour_for_warp = cv2.boxPoints(rect_forced)
    
    # FIX 1: Replace np.int0 with np.int32
    card_contour_for_warp = np.int32(card_contour_for_warp) 

    # --- Debugging: Draw the FORCED 7:5 contour (blue) ---
    if debug_output_folder:
        debug_forced_contour_img = img.copy()
        cv2.drawContours(debug_forced_contour_img, [card_contour_for_warp], 0, (255, 0, 0), 4) 
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_contours_forced_blue.jpg")
            cv2.imwrite(debug_filename, debug_forced_contour_img)
            print(f"Debug: Saved forced 7:5 contour detection to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug forced contours: {e}")
    # --- End Debugging ---

    # 4. Perspective Transformation (Flattening)
    if scale_factor != 1.0:
        # FIX 2: Replace np.int0 with np.int32
        card_contour_original_scale = (card_contour_for_warp / scale_factor).astype(np.int32)
        warped = four_point_transform(original_img, card_contour_original_scale.reshape(4, 2), target_width, target_height)
    else:
        warped = four_point_transform(img, card_contour_for_warp.reshape(4, 2), target_width, target_height)
    
    # 5. Crop the SET ID/Number area (bottom-LEFT quadrant)
    crop_height_perc = 0.20 
    crop_width_perc = 0.35  
    
    y_start = int(target_height * (1 - crop_height_perc))
    x_start = 0 
    x_end = int(target_width * crop_width_perc)
    
    final_crop = warped[y_start:target_height, x_start:x_end]

    # --- DEBUGGING STEP: Save the final processed image ---
    if debug_output_folder:
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_processed_final_crop.jpg")
            cv2.imwrite(debug_filename, final_crop)
            print(f"Debug: Saved final processed crop to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug final crop to '{debug_filename}': {e}")

    return final_crop

# --- OCR and Heuristic Functions (unchanged) ---

def identify_card_info(image_path, ocr_engine, debug_output_folder=None):
    """
    Full pipeline: Preprocess -> OCR -> Heuristic Extraction.
    """
    print(f"\n--- Processing Image: {os.path.basename(image_path)} ---")
    
    img_for_ocr = preprocess_card_image(image_path, debug_output_folder=debug_output_folder) 
    
    if img_for_ocr is None:
        print("Pre-processing failed. Skipping OCR.")
        return None

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

    print("--- DEBUG: All Detected Text Strings ---")
    if detected_texts:
        for text in detected_texts:
            print(f"[RAW]: {text}")
    else:
        print("[RAW]: No text detected by the OCR model.")
    print("-----------------------------------------")
    
    for text in detected_texts:
        text = str(text).upper().strip()
        
        # Heuristic 1: Look for potential set IDs
        if text.isalpha() and len(text) >= 2 and len(text) <= 5:
            if text.endswith('EN'):
                text = text[:-2] 
            
            if text.isalpha() and len(text) >= 2 and text not in ['EN', 'FR', 'JP', 'DE', 'IT', 'POL']: 
                set_id = text
        
        # Heuristic 2: Look for potential card numbers
        number_part = None
        if '/' in text:
            parts = text.split('/')
            number_part = parts[0]
        elif text.isdigit() and len(text) <= 4:
            number_part = text
        
        if number_part is not None and number_part.isdigit():
            card_number = number_part

    if set_id and card_number:
        formatted_info = f"{set_id}-{card_number}"
        return formatted_info
    else:
        print(f"Final Extraction Failed: Set ID found: {set_id}, Card Number found: {card_number}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    
    DEBUG_OUTPUT_FOLDER = "debug_output" 
    
    if not os.path.isdir(CARD_FOLDER):
        print(f"Error: The folder '{CARD_FOLDER}' was not found.")
        print("Please create a folder named 'photos' in the same directory as this script.")
        sys.exit(1)

    print("Initializing PaddleOCR (This may take a moment and download models the first time)...")
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', device='cpu') 
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Please check your PaddleOCR installation and its dependencies.")
        sys.exit(1)

    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(CARD_FOLDER, ext)))
        
    if not image_paths:
        print(f"No supported image files found in the '{CARD_FOLDER}' folder. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(image_paths)} image(s) to process. Starting batch analysis...")
    
    results = {}
    
    for path in image_paths:
        info = identify_card_info(path, ocr, debug_output_folder=DEBUG_OUTPUT_FOLDER)
        if info:
            results[os.path.basename(path)] = info
        else:
            results[os.path.basename(path)] = "--- FAILED ---"

    print("\n" + "="*40)
    print("      BATCH PROCESSING SUMMARY")
    print("="*40)
    for filename, result in results.items():
        print(f"[{filename:<20}]: **{result}**")
    print("="*40)