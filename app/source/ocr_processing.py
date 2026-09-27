import json

# --- User Configuration ---
# To avoid false positives, only look for set abbreviations that are known to
# exist in your collection. This prevents random OCR noise (e.g. "THE") from
# being misread as a set ID (e.g. "TEF").
# Add or remove set abbreviations based on the sets you are scanning.
ALLOWED_SET_ABBREVIATIONS = [
    "SVI", "DRI", "PAL", "MEW", "SCR", "TEF", "JTG", "BLK", "SVP", "OBF",
    "PAF", "PRE", "TWM", "PAR", "SSP", "SFA", "WHT", "MEG", "PFL", "MEP"
]

# The card's set line often includes a language suffix (e.g. "SVIEN" = SVI + EN).
# Knowing these lets us strip the suffix and recover the pure set abbreviation.
SUPPORTED_LANGUAGES = ['EN', 'ES', 'FR', 'DE', 'IT']


def find_card_in_database(set_id, card_number, card_database):
    """Search the card database for a card matching set_id + card_number.

    The database is a dict of lists, e.g.:
        { "Scarlet & Violet": [ {set_abbreviation, card_number, name_en, ...}, ... ] }

    Returns the matching card dict, or None if not found.
    """
    if not card_database or not isinstance(card_database, dict):
        return None

    try:
        card_num_int = int(card_number)
    except (ValueError, TypeError):
        return None  # OCR'd card_number is not a valid integer

    # Flatten the dict of lists into a single list for simple iteration
    all_cards = []
    for card_list in card_database.values():
        if isinstance(card_list, list):
            all_cards.extend(card_list)

    for card in all_cards:
        if not isinstance(card, dict):
            continue

        if card.get('set_abbreviation') == set_id:
            db_card_num = card.get('card_number')
            try:
                # Compare as integers so "007" == "7"
                if db_card_num is not None and int(db_card_num) == card_num_int:
                    return card
            except (ValueError, TypeError):
                continue

    return None


def extract_card_info_from_text(detected_texts, card_database=None):
    """Extract set ID and card number from a list of OCR text strings.

    OCR output is noisy — words may be split, merged, or garbled. This
    function uses a priority-scoring system: each candidate gets a numeric
    preference score (lower = better), and we take the best-scoring one.

    Stage 1 — find all candidates and assign priority scores.
    Stage 1b — fuzzy fallback if no exact set ID was found.
    Stage 2 — pick the highest-priority candidate from each group.
    Stage 3 — look up the card in the database and format the result.
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
    set_candidates = []  # list of (preference_score, set_abbreviation)
    num_candidates = []  # list of (preference_score, card_number_string)

    for text in detected_texts:
        cleaned_text = str(text).upper().strip()

        # ── Card number extraction ──────────────────────────────────────────
        # Priority 1 (best): "NNN/TTT" format — e.g. "196/198"
        # The slash form is the most reliable because it appears on every card.
        if '/' in cleaned_text:
            parts = cleaned_text.split('/')
            num_part = parts[0].strip()
            # The token before the slash may have a prefix (e.g. "G SVIEN 196/198")
            # so we take only the last space-separated token.
            if ' ' in num_part:
                num_part = num_part.split()[-1]
            # OCR often confuses 'O' and '0'
            num_part = num_part.replace('O', '0')
            if num_part.isdigit():
                num_candidates.append((1, num_part))

        # Priority 2 (fallback): a standalone number (no slash context)
        elif cleaned_text.replace('O', '0').isdigit() and len(cleaned_text) <= 4:
            num_candidates.append((2, cleaned_text.replace('O', '0')))

        # ── Set ID extraction from slash lines ─────────────────────────────
        # On cards, the set line looks like: "SVIEN 196/198"
        # We scan tokens from that line for a known set+language pattern.
        if '/' in cleaned_text:
            tokens = cleaned_text.split()
            for i, token in enumerate(tokens):
                # Skip single-char noise tokens
                if len(token) <= 1 or token.isdigit() or token in ['-', 'G', 'H', 'SE', 'O']:
                    continue
                potential_set_id = token
                # Check if the token is exactly "ABBR+LANG" (e.g. "SVIEN")
                for allowed_abbr in ALLOWED_SET_ABBREVIATIONS:
                    for suffix in SUPPORTED_LANGUAGES:
                        combined = allowed_abbr + suffix
                        if potential_set_id == combined or potential_set_id.startswith(combined):
                            set_candidates.append((1, allowed_abbr))
                            break
                    else:
                        continue
                    break

        # ── Set ID extraction from any line (no slash required) ─────────────
        tokens = cleaned_text.split()
        potential_set_id = cleaned_text

        # If there are multiple tokens, pick the first substantive one
        if len(tokens) > 1:
            for token in tokens:
                if len(token) <= 2 or token.isdigit() or '/' in token or token in ['-', 'SE', 'G', 'H', 'O']:
                    continue
                potential_set_id = token
                break

        is_suffixed = False

        # Strip a trailing language suffix (e.g. "SVIEN" → "SVI")
        for suffix in SUPPORTED_LANGUAGES:
            if potential_set_id.endswith(suffix) and len(potential_set_id) > len(suffix):
                potential_set_id = potential_set_id[:-len(suffix)]
                is_suffixed = True
                break

        # Handle the concatenated form (e.g. "TWMEN" → "TWM")
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

        # Only accept candidates that exactly match a known abbreviation
        if potential_set_id in ALLOWED_SET_ABBREVIATIONS:
            # A language suffix is evidence the OCR read the whole token correctly
            if is_suffixed:
                set_candidates.append((1, potential_set_id))  # high confidence
            else:
                set_candidates.append((2, potential_set_id))  # lower confidence

    # --- Stage 1b: Fuzzy fallback for set ID ---
    # If no exact match was found, try again allowing one character difference.
    # This recovers from a single misread character (e.g. "TVV" → "TWM").
    if not set_candidates:
        for text in detected_texts:
            cleaned_text = str(text).upper().strip()

            tokens = cleaned_text.split()
            potential_set_id = cleaned_text

            if len(tokens) > 1:
                for token in tokens:
                    if len(token) <= 1 or token.isdigit() or '/' in token:
                        continue
                    potential_set_id = token
                    break

            is_suffixed = False

            for suffix in SUPPORTED_LANGUAGES:
                if potential_set_id.endswith(suffix) and len(potential_set_id) > len(suffix):
                    potential_set_id = potential_set_id[:-len(suffix)]
                    is_suffixed = True
                    break

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

            # Count character-position differences between the candidate and
            # each known abbreviation. Accept if exactly 1 position differs.
            for allowed_abbr in ALLOWED_SET_ABBREVIATIONS:
                if len(potential_set_id) == len(allowed_abbr):
                    diff = sum(c1 != c2 for c1, c2 in zip(potential_set_id, allowed_abbr))
                    if diff == 1:
                        # Lower priority scores for fuzzy matches
                        if is_suffixed:
                            set_candidates.append((3, allowed_abbr))  # fuzzy + suffix
                        else:
                            set_candidates.append((4, allowed_abbr))  # fuzzy, no suffix

    # --- Stage 2: Select the best candidate from each group ---
    # sort() on tuples compares the first element first, so the lowest
    # preference score (= most confident match) bubbles to index 0.
    if num_candidates:
        num_candidates.sort(key=lambda x: x[0])
        card_number = num_candidates[0][1]

    if set_candidates:
        set_candidates.sort(key=lambda x: x[0])
        set_id = set_candidates[0][1]

    # --- Stage 3: Lookup and return ---
    if set_id and card_number:
        found_card = find_card_in_database(set_id, card_number, card_database)

        if found_card:
            card_name    = found_card.get("name_en", "Unknown Name")
            trainer_type = found_card.get("trainer_type")
            card_types   = found_card.get("types")

            # Add the card type in parentheses so the result is human-readable
            extra_info = ""
            if trainer_type:
                extra_info = f" ({trainer_type})"
            elif card_types and isinstance(card_types, list) and card_types:
                extra_info = f" ({card_types[0]})"

            return f"{set_id}-{card_number}: {card_name}{extra_info}"
        else:
            # We identified the card position but it wasn't in the database
            return f"{set_id}-{card_number} (Unverified)"
    else:
        # Include whatever partial info we found to aid debugging
        return f"--- FAILED: OCR Heuristic (Set ID: {set_id}, Number: {card_number}) ---"
