import json

# --- User Configuration ---
# To avoid false positives, only look for set abbreviations that are known to exist in your collection.
# This helps prevent misinterpreting random text as a set ID (e.g., "THE" -> "TEF").
# Add or remove set abbreviations from this list based on the sets you are scanning.
ALLOWED_SET_ABBREVIATIONS = [
    "SVI", "DRI", "PAL", "MEW", "SCR", "TEF", "JTG", "BLK", "SVP", "OBF",
    "PAF", "PRE", "TWM", "PAR", "SSP", "SFA", "WHT", "MEG", "PFL", "MEP"
]

# Configure the language suffixes to look for. This helps identify set IDs more reliably.
# The script is optimized for European languages. Add or remove suffixes as needed.
SUPPORTED_LANGUAGES = ['EN', 'ES', 'FR', 'DE', 'IT']


def find_card_in_database(set_id, card_number, card_database):
    """
    Searches the card database for a match.
    The database is expected to be a dictionary where keys are set groups
    and values are lists of card objects.
    """
    if not card_database or not isinstance(card_database, dict):
        return None
        
    try:
        card_num_int = int(card_number)
    except (ValueError, TypeError):
        return None # OCR'd card_number is not a valid integer

    # Flatten the dictionary of lists into a single list of cards.
    all_cards = []
    for card_list in card_database.values():
        if isinstance(card_list, list):
            all_cards.extend(card_list)

    # Now we have a single list `all_cards` to search through.
    for card in all_cards:
        if not isinstance(card, dict):
            continue 

        if card.get('set_abbreviation') == set_id:
            db_card_num = card.get('card_number')
            try:
                # Robustly compare card numbers as integers
                if db_card_num is not None and int(db_card_num) == card_num_int:
                    return card
            except (ValueError, TypeError):
                # Ignore if the card_number in the DB isn't a clean integer string
                continue
            
    return None

def extract_card_info_from_text(detected_texts, card_database=None):
    """
    Extracts Set ID and Card Number from a list of OCR-detected text strings,
    and looks up the card in the provided database.
    """
    set_id = None
    card_number = None

    if not detected_texts:
        print("[Text Proc]: No text provided to process.")
        return "--- FAILED: No text detected ---"

    print("\n--- DEBUG: Raw Text for Processing ---")
    for text in detected_texts:
        print(f"[RAW]: {text}")
    print("------------------------------------")
    
    # --- Stage 1: Find all candidates with a priority score ---
    set_candidates = [] # (preference, value)
    num_candidates = [] # (preference, value)

    for text in detected_texts:
        cleaned_text = str(text).upper().strip()
        
        # --- Card Number Logic ---
        # Pref 1: "XXX/YYY" format (highest priority)
        if '/' in cleaned_text:
            parts = cleaned_text.split('/')
            # Extract just the number part, handling cases like "G SVIEN 196/198"
            num_part = parts[0].strip()
            # Get the last token before the slash
            if ' ' in num_part:
                num_part = num_part.split()[-1]
            num_part = num_part.replace('O', '0')
            if num_part.isdigit():
                num_candidates.append((1, num_part))

        # Pref 2: Standalone number (lower priority)
        elif cleaned_text.replace('O', '0').isdigit() and len(cleaned_text) <= 4:
            num_candidates.append((2, cleaned_text.replace('O', '0')))

        # --- Set ID Logic ---
        # First, try to extract a pattern like "H SFAEN" or "G SVIEN"
        # Remove leading single letters and spaces
        tokens = cleaned_text.split()
        potential_set_id = cleaned_text
        
        # If we have multiple tokens, look for the set code pattern
        if len(tokens) > 1:
            for token in tokens:
                # Skip single character tokens (H, G, etc.) and numbers
                if len(token) <= 1 or token.isdigit() or '/' in token:
                    continue
                # This could be our set code
                potential_set_id = token
                break
        
        is_suffixed = False
        
        # Check for language suffixes after extracting potential set ID
        # First check if it ends with a language code (no space)
        for suffix in SUPPORTED_LANGUAGES:
            if potential_set_id.endswith(suffix) and len(potential_set_id) > len(suffix):
                potential_set_id = potential_set_id[:-len(suffix)]
                is_suffixed = True
                break
        
        # Additional check: Look for set abbreviation followed by language code
        # e.g., "SVIEN" -> "SVI" + "EN", "TWMEN" -> "TWM" + "EN"
        if not is_suffixed:
            for allowed_abbr in ALLOWED_SET_ABBREVIATIONS:
                for suffix in SUPPORTED_LANGUAGES:
                    combined = allowed_abbr + suffix
                    if potential_set_id == combined or potential_set_id.startswith(combined):
                        potential_set_id = allowed_abbr
                        is_suffixed = True
                        break
                if is_suffixed:
                    break
        
        # Validation and scoring
        if potential_set_id in ALLOWED_SET_ABBREVIATIONS:
            if is_suffixed:
                set_candidates.append((1, potential_set_id)) # Pref 1 (high)
            else:
                set_candidates.append((2, potential_set_id)) # Pref 2 (low)

    # --- Stage 1b: Fuzzy Fallback for Set ID ---
    if not set_candidates:
        for text in detected_texts:
            cleaned_text = str(text).upper().strip()
            
            # Extract potential set ID from tokens
            tokens = cleaned_text.split()
            potential_set_id = cleaned_text
            
            if len(tokens) > 1:
                for token in tokens:
                    if len(token) <= 1 or token.isdigit() or '/' in token:
                        continue
                    potential_set_id = token
                    break
            
            is_suffixed = False
            
            # Check for language suffixes
            for suffix in SUPPORTED_LANGUAGES:
                if potential_set_id.endswith(suffix) and len(potential_set_id) > len(suffix):
                    potential_set_id = potential_set_id[:-len(suffix)]
                    is_suffixed = True
                    break
            
            # Check for concatenated set+language
            if not is_suffixed:
                for allowed_abbr in ALLOWED_SET_ABBREVIATIONS:
                    for suffix in SUPPORTED_LANGUAGES:
                        combined = allowed_abbr + suffix
                        if potential_set_id == combined or potential_set_id.startswith(combined):
                            potential_set_id = allowed_abbr
                            is_suffixed = True
                            break
                    if is_suffixed:
                        break
            
            # Fuzzy comparison (1-character difference)
            for allowed_abbr in ALLOWED_SET_ABBREVIATIONS:
                if len(potential_set_id) == len(allowed_abbr):
                    diff = sum(c1 != c2 for c1, c2 in zip(potential_set_id, allowed_abbr))
                    if diff == 1:
                        # Use a lower preference for fuzzy matches
                        if is_suffixed:
                            set_candidates.append((3, allowed_abbr)) # Pref 3 (fuzzy with suffix)
                        else:
                            set_candidates.append((4, allowed_abbr)) # Pref 4 (fuzzy without suffix)

    # --- Stage 2: Select the best candidates ---
    if num_candidates:
        num_candidates.sort(key=lambda x: x[0]) # Sort by preference
        card_number = num_candidates[0][1]

    if set_candidates:
        set_candidates.sort(key=lambda x: x[0])
        set_id = set_candidates[0][1]

    # --- Stage 3: Lookup and Return ---
    if set_id and card_number:
        # We found a set and number, now look it up.
        found_card = find_card_in_database(set_id, card_number, card_database)
        
        if found_card:
            card_name = found_card.get("name_en", "Unknown Name")
            trainer_type = found_card.get("trainer_type")
            card_types = found_card.get("types")

            extra_info = ""
            if trainer_type:
                extra_info = f" ({trainer_type})"
            elif card_types and isinstance(card_types, list) and card_types:
                extra_info = f" ({card_types[0]})"
            
            return f"{set_id}-{card_number}: {card_name}{extra_info}"
        else:
            # Still return the found ID if lookup fails
            return f"{set_id}-{card_number} (Unverified)"
    else:
        # Include found parts in the failure message for better debugging
        return f"--- FAILED: OCR Heuristic (Set ID: {set_id}, Number: {card_number}) ---"