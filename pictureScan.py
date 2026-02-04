import cv2
import numpy as np
import os
import glob
import sys
import json
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from paddleocr import PaddleOCR
from PIL import Image
from ocr_processing import extract_card_info_from_text

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

def corner_crop_image(image_path, height_percent=0.50, width_percent=0.50):
    """
    Crops the lower-left corner of the image.
    Default: 50% height from bottom, 50% width from left.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Calculate crop dimensions
    crop_height = int(height * height_percent)
    crop_width = int(width * width_percent)
    
    # Crop lower-left corner
    y_start = height - crop_height
    x_start = 0
    
    cropped = img[y_start:height, x_start:crop_width]
    
    return cropped

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
    s_channel = hsv[:, :, 1]
    
    # --- Debugging: Save S-channel ---
    if debug_output_folder:
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_1_s_channel.jpg")
            cv2.imwrite(debug_filename, s_channel)
            print(f"Debug: Saved S-channel to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug S-channel: {e}")
    # --- End Debugging ---

    # 2. Thresholding and Morphology
    _, thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- Debugging: Save threshold ---
    if debug_output_folder:
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_2_threshold.jpg")
            cv2.imwrite(debug_filename, thresh)
            print(f"Debug: Saved threshold to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug threshold: {e}")
    # --- End Debugging ---

    kernel_dilation = np.ones((11, 11), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilation, iterations=3)
    
    # --- Debugging: Save dilated threshold ---
    if debug_output_folder:
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_3_dilated.jpg")
            cv2.imwrite(debug_filename, thresh)
            print(f"Debug: Saved dilated threshold to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug dilated: {e}")
    # --- End Debugging ---

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
        
        # Calculate aspect ratio
        aspect_ratio = height / width
        
        # Check if aspect ratio is close to 1.4 (card-like)
        aspect_tolerance = 0.3
        if abs(aspect_ratio - CARD_ASPECT_RATIO) < aspect_tolerance:
            if area > max_contour_area:
                largest_card_contour = c
                max_contour_area = area

    if largest_card_contour is None:
        print("Warning: No suitable card-like contour found. Returning None.")
        return None

    # --- Debugging: Draw all contours and highlight the chosen one ---
    if debug_output_folder:
        try:
            debug_contour_img = img.copy()
            cv2.drawContours(debug_contour_img, contours[:5], -1, (0, 255, 0), 2)  # Top 5 in green
            cv2.drawContours(debug_contour_img, [largest_card_contour], -1, (0, 0, 255), 3)  # Chosen in red
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_4_contours.jpg")
            cv2.imwrite(debug_filename, debug_contour_img)
            print(f"Debug: Saved contours to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug contours: {e}")
    # --- End Debugging ---

    # Get the minimum area rectangle and force it to 7:5 aspect ratio
    rect = cv2.minAreaRect(largest_card_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Calculate the dimensions and force 7:5 aspect ratio
    (center, (width, height), angle) = rect
    if width > height:
        width, height = height, width
        angle = angle + 90
    
    # Force aspect ratio to exactly 7:5
    target_aspect = CARD_ASPECT_RATIO
    if height / width > target_aspect:
        # Height is too large, reduce it
        height = width * target_aspect
    else:
        # Width is too large, reduce it
        width = height / target_aspect
    
    # Create new rotated rectangle with forced dimensions
    forced_rect = (center, (width, height), angle)
    card_contour_for_warp = cv2.boxPoints(forced_rect)
    card_contour_for_warp = np.intp(card_contour_for_warp)
    
    # --- Debugging: Draw forced 7:5 contour in blue ---
    if debug_output_folder:
        try:
            debug_forced_contour_img = img.copy()
            cv2.drawContours(debug_forced_contour_img, [card_contour_for_warp], -1, (255, 0, 0), 3)  # Blue
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_5_contours_forced_blue.jpg")
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

    # --- Debugging: Save warped card ---
    if debug_output_folder:
        try:
            debug_filename = os.path.join(debug_output_folder, f"{base_name}_6_warped.jpg")
            cv2.imwrite(debug_filename, warped)
            print(f"Debug: Saved warped card to '{debug_filename}'")
        except Exception as e:
            print(f"Error saving debug warped: {e}")
    # --- End Debugging ---

    return warped

# --- OCR and Heuristic Functions ---

def identify_card_info_doctr(img_for_ocr, ocr_model, card_database=None):
    """
    Runs DocTR OCR and heuristic extraction on a given image.
    """
    if img_for_ocr is None:
        print("Cannot process a null image. Skipping OCR.")
        return None

    # Convert OpenCV image (BGR) to RGB numpy array
    img_rgb = cv2.cvtColor(img_for_ocr, cv2.COLOR_BGR2RGB)
    
    # DocTR expects numpy array or PIL Image
    # Run OCR
    result = ocr_model([img_rgb])
    
    # Extract text from DocTR results
    detected_texts = []
    if result:
        # DocTR returns a Document object
        # Navigate: result.pages[0].blocks -> lines -> words
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    # Concatenate all words in the line
                    line_text = ' '.join([word.value for word in line.words])
                    if line_text.strip():
                        detected_texts.append(line_text.strip())
    
    # Use the shared function for text processing
    info = extract_card_info_from_text(detected_texts, card_database)

    if "FAILED" in info:
        return None
    else:
        return info

def identify_card_info_paddle(img_for_ocr, paddle_ocr, card_database=None):
    """
    Runs PaddleOCR and heuristic extraction on a given image (fallback method).
    """
    if img_for_ocr is None:
        print("Cannot process a null image. Skipping OCR.")
        return None

    result = paddle_ocr.predict(img_for_ocr)

    detected_texts = []
    if result and isinstance(result, list) and len(result) > 0:
        ocr_result_object = result[0]
        
        if hasattr(ocr_result_object, 'rec_texts'):
            detected_texts = ocr_result_object.rec_texts
        elif isinstance(ocr_result_object, dict) and 'rec_texts' in ocr_result_object:
            detected_texts = ocr_result_object['rec_texts']
            
    # Use the shared function for text processing
    info = extract_card_info_from_text(detected_texts, card_database)

    if "FAILED" in info:
        return None
    else:
        return info

# --- Main execution ---
if __name__ == "__main__":
    
    # --- Get Card Folder ---
    use_default = input(f"Do you want to use the default folder '{CARD_FOLDER}'? (Y/n): ").lower().strip()
    if use_default == 'n':
        custom_folder = input("Please enter the path to the folder with your card images: ").strip()
        if os.path.isdir(custom_folder):
            CARD_FOLDER = custom_folder
        else:
            print(f"Error: The folder '{custom_folder}' was not found. Using default folder '{CARD_FOLDER}'.")

    DEBUG_OUTPUT_FOLDER = "debug_output" 
    
    if not os.path.isdir(CARD_FOLDER):
        print(f"Error: The folder '{CARD_FOLDER}' was not found.")
        print("Please create a folder named 'photos' in the same directory as this script, or provide a valid path.")
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
        card_database = None # Ensure it's None if loading fails

    # --- Ask for cropping mode ---
    cropping_mode = ''
    while cropping_mode not in ['c', 'd', 'f']:
        print("Cropping options:")
        print("  c = Corner crop (lower-left 50% height, 50% width - lightweight)")
        print("  d = Detect crop (automatic card detection - heavier processing)")
        print("  f = Full image (no cropping)")
        cropping_mode = input("Select cropping mode (c/d/f): ").lower()

    print("Initializing DocTR OCR with MobileNet (This may take a moment and download models the first time)...")
    try:
        # Initialize DocTR with lightweight MobileNet models
        # det_arch: detection architecture (db_mobilenet_v3_large is fast and accurate)
        # reco_arch: recognition architecture (crnn_mobilenet_v3_small is lightweight)
        ocr_model = ocr_predictor(
            det_arch='db_mobilenet_v3_large',
            reco_arch='crnn_mobilenet_v3_small',
            pretrained=True
        )
        print("✅ DocTR OCR models loaded successfully.")
    except Exception as e:
        print(f"Error initializing DocTR OCR: {e}")
        print("Please check your DocTR installation and its dependencies.")
        print("Install with: pip install python-doctr[torch]")
        sys.exit(1)

    print("Initializing PaddleOCR (fallback engine)...")
    try:
        paddle_ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            enable_mkldnn=False
        )
        print("✅ PaddleOCR loaded successfully (fallback ready).")
    except Exception as e:
        print(f"Warning: Could not initialize PaddleOCR fallback: {e}")
        print("Will continue with DocTR only.")
        paddle_ocr = None

    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(CARD_FOLDER, ext)))
        
    if not image_paths:
        print(f"No supported image files found in the '{CARD_FOLDER}' folder. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(image_paths)} image(s) to process. Starting batch analysis...")
    
    results = {}
    
    for path in image_paths:
        print(f"\n--- Processing Image: {os.path.basename(path)} ---")
        
        img_to_process = None
        if cropping_mode == 'c':
            img_to_process = corner_crop_image(path)
        elif cropping_mode == 'd':
            img_to_process = preprocess_card_image(path, debug_output_folder=DEBUG_OUTPUT_FOLDER)
        else: # mode == 'f'
            img_to_process = cv2.imread(path)

        if img_to_process is None:
            print(f"❌ FAILED to read or process image.")
            results[os.path.basename(path)] = "--- FAILED (Preprocessing/Read) ---"
            continue
            
        # Try DocTR first (faster)
        print("Attempting OCR with DocTR (fast)...")
        info = identify_card_info_doctr(img_to_process, ocr_model, card_database)

        if info:
            print(f"✅ IDENTIFIED with DocTR: {info}")
            results[os.path.basename(path)] = info
        elif paddle_ocr is not None:
            # Fallback to PaddleOCR if DocTR failed
            print("⚠️  DocTR failed to identify card. Trying PaddleOCR fallback...")
            info = identify_card_info_paddle(img_to_process, paddle_ocr, card_database)
            
            if info:
                print(f"✅ IDENTIFIED with PaddleOCR (fallback): {info}")
                results[os.path.basename(path)] = f"{info} [PaddleOCR]"
            else:
                print(f"❌ IDENTIFICATION FAILED with both engines.")
                results[os.path.basename(path)] = "--- FAILED ---"
        else:
            print(f"❌ IDENTIFICATION FAILED (no fallback available).")
            results[os.path.basename(path)] = "--- FAILED ---"

    print("\n" + "="*40)
    print("      BATCH PROCESSING SUMMARY")
    print("="*40)
    for filename, result in results.items():
        print(f"[{filename:<20}]: **{result}**")
    print("="*40)
