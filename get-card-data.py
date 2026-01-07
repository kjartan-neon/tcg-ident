import os
import re
import json
import glob

# --- Configuration ---
INPUT_FOLDER = "tcgdex/data" # Start traversal from the 'data' folder
OUTPUT_FILENAME = "card_data_lookup.json"

# --- Regex Patterns ---

# Pattern 1: Find the English Card Name within the 'name' object.
CARD_NAME_REGEX = re.compile(
    r'name:\s*\{\s*.*?\ben:\s*"(.*?)"',
    re.DOTALL | re.IGNORECASE
)

# Pattern 2: Find the English Name of the FIRST Attack.
ATTACK_NAME_REGEX = re.compile(
    r'attacks:\s*\[.*?name:\s*\{\s*.*?\ben:\s*"(.*?)"',
    re.DOTALL | re.IGNORECASE
)

# Pattern 3: Find Set ID (e.g., "me02")
SET_ID_REGEX = re.compile(r'id:\s*"(.*?)"', re.DOTALL)

# Pattern 4: Find Official Set Abbreviation (e.g., "PFL")
SET_ABBREVIATION_REGEX = re.compile(
    r'abbreviations:\s*\{\s*.*?\bofficial:\s*"(.*?)"',
    re.DOTALL | re.IGNORECASE
)

# Pattern 5: Find Card Category (e.g., "Pokemon", "Trainer")
CARD_CATEGORY_REGEX = re.compile(r'category:\s*"(.*?)"', re.DOTALL)

# Pattern 6: Find Card HP (e.g., 70, 180)
CARD_HP_REGEX = re.compile(r'hp:\s*(\d+)', re.DOTALL)

# Pattern 7: Find Trainer Type (conditional, e.g., "Supporter", "Item")
TRAINER_TYPE_REGEX = re.compile(r'trainerType:\s*"(.*?)"', re.DOTALL)

# Pattern 8: Find Card Types (e.g., ["Grass"], ["Fire", "Water"])
CARD_TYPES_REGEX = re.compile(r'types:\s*\[(.*?)\]', re.DOTALL)


# --- Helper Functions ---

def read_ts_file(filepath):
    """Safely reads the content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_set_metadata(filepath):
    """Extracts ID and official abbreviation from a Set TS file."""
    content = read_ts_file(filepath)
    if not content:
        return None

    set_id_match = SET_ID_REGEX.search(content)
    set_abbr_match = SET_ABBREVIATION_REGEX.search(content)

    # Changed default from "N/A" to None
    set_id = set_id_match.group(1).strip() if set_id_match else None
    set_abbr = set_abbr_match.group(1).strip() if set_abbr_match else None

    return {
        "set_id": set_id,
        "set_abbreviation": set_abbr
    }

def extract_card_data(filepath, set_abbreviation):
    """
    Reads a single Card TS file and extracts details, including new fields.
    Note: set_id is NOT included in the return data as it will be the key in the final JSON.
    """
    
    # 1. Get Card Number from filename
    filename = os.path.basename(filepath)
    card_number_full = filename.replace(".ts", "")
    card_number_int = card_number_full.lstrip('0')
    if not card_number_int:
        card_number_int = "0"

    content = read_ts_file(filepath)
    if not content:
        return None

    # 2. Extract Card Name (English)
    card_name_match = CARD_NAME_REGEX.search(content)
    card_name_en = card_name_match.group(1).strip() if card_name_match else None

    # 3. Extract First Attack Name (English)
    attack_name_match = ATTACK_NAME_REGEX.search(content)
    attack_name_en = attack_name_match.group(1).strip() if attack_name_match else None
    
    # 4. Extract Category
    category_match = CARD_CATEGORY_REGEX.search(content)
    category = category_match.group(1).strip() if category_match else None

    # 5. Extract HP
    hp_match = CARD_HP_REGEX.search(content)
    # Convert HP to integer if found, otherwise None
    hp = int(hp_match.group(1).strip()) if hp_match else None

    # 6. Extract Trainer Type (Conditional)
    trainer_type_match = TRAINER_TYPE_REGEX.search(content)
    trainer_type = trainer_type_match.group(1).strip() if trainer_type_match else None

    # 7. Extract Card Types
    types_match = CARD_TYPES_REGEX.search(content)
    types = None
    if types_match:
        types_str = types_match.group(1)
        # Split by comma, strip whitespace and quotes, and filter out empty strings
        types = [t.strip().replace('"', '').replace("'", "") for t in types_str.split(',') if t.strip()]


    return {
        "id": int(card_number_int),
        "card_number": card_number_full,
        "name_en": card_name_en,
        "first_attack_name_en": attack_name_en,
        "category": category,
        "hp": hp,
        "trainer_type": trainer_type,
        "types": types,
        "set_abbreviation": set_abbreviation # Keep abbreviation for card context
    }

def main():
    """Main function to traverse files, extract data, and generate the final JSON grouped by set_id."""
    # This will now be a dictionary: {"set_id_1": [card_data_1, card_data_2], "set_id_2": [...]}
    grouped_card_data = {} 
    
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found. Please create it and structure your files inside.")
        return

    print(f"Starting traversal in: {os.path.abspath(INPUT_FOLDER)}")
    
    # os.walk iterates through directory tree, returning (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(INPUT_FOLDER):
        
        # Check if the current directory is a Card Folder (contains card files like '001.ts')
        card_files = [f for f in filenames if re.match(r'^\d{3}\.ts$', f)]
        
        if card_files:
            set_folder_name = os.path.basename(dirpath)
            
            # The Set TS file is a sibling of the current card folder (e.g., 'Phantasmal Flames.ts')
            series_folder_path = os.path.dirname(dirpath)
            set_ts_filepath = os.path.join(series_folder_path, f"{set_folder_name}.ts")
            
            if not os.path.exists(set_ts_filepath):
                print(f"Warning: Set metadata file not found at {set_ts_filepath}. Skipping cards in {dirpath}.")
                continue
            
            # 1. Extract Set Metadata
            set_metadata = get_set_metadata(set_ts_filepath)
            set_id = set_metadata["set_id"]
            set_abbreviation = set_metadata["set_abbreviation"]

            if set_abbreviation == "SV":
                set_abbreviation = "SVI"

            # If set_id extraction failed (now None), we skip the cards in this folder
            if set_id is None:
                 print(f"Warning: Could not extract valid set_id from {set_ts_filepath}. Skipping cards.")
                 continue
            
            print(f"  -> Processing Set: {set_folder_name} (ID: {set_id}, Abbr: {set_abbreviation})")
            
            # Initialize the list for this set_id if it doesn't exist
            if set_id not in grouped_card_data:
                grouped_card_data[set_id] = []
            
            # 2. Process Card Files inside this folder
            for filename in sorted(card_files):
                card_filepath = os.path.join(dirpath, filename)
                # Pass only set_abbreviation, as set_id is used for grouping key
                data = extract_card_data(card_filepath, set_abbreviation) 
                
                if data:
                    grouped_card_data[set_id].append(data)

    if not grouped_card_data:
        print("\nNo card data processed. Check file paths and structure.")
        return

    # Write the results to the final JSON file
    try:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as outfile:
            # Output the grouped dictionary
            json.dump(grouped_card_data, outfile, indent=4, ensure_ascii=False)
        
        print("\n--- Success ---")
        print(f"Successfully processed {sum(len(v) for v in grouped_card_data.values())} cards across {len(grouped_card_data)} sets.")
        print(f"Output saved to: {os.path.abspath(OUTPUT_FILENAME)}")

    except Exception as e:
        print(f"Error writing JSON file: {e}")


if __name__ == "__main__":
    main()