"""
Card database builder.

Converts the tcgdex TypeScript card-database source files into the
card_data_lookup.json that the rest of the app uses for card lookup.

The tcgdex repository (https://github.com/tcgdex/cards-database) organises
its data like this:

    data/
      Scarlet & Violet/                   ← series folder
        Scarlet & Violet Base Set.ts      ← set metadata file
        Scarlet & Violet Base Set/        ← card folder (same name)
          001.ts
          002.ts
          ...
        Paldea Evolved.ts
        Paldea Evolved/
          001.ts
          ...

`build_database()` walks that tree, reads each set's metadata .ts file to get
the official abbreviation (e.g. "SVI"), then reads every numbered card .ts
file and extracts the fields the app needs.  Results are written as a single
JSON object keyed by internal set-id.
"""

import os
import re
import json
from typing import Callable, Optional, Tuple

# ── Regex patterns ────────────────────────────────────────────────────────────
# Each pattern is designed to extract one field from a TypeScript source file.
# re.DOTALL makes '.' match newlines too, which is essential because the
# multi-line objects in these files span many lines.

# English card name inside a `name: { en: "..." }` block
_CARD_NAME_RE = re.compile(
    r'name:\s*\{\s*.*?\ben:\s*"(.*?)"', re.DOTALL | re.IGNORECASE)

# English name of the first attack
_ATTACK_NAME_RE = re.compile(
    r'attacks:\s*\[.*?name:\s*\{\s*.*?\ben:\s*"(.*?)"', re.DOTALL | re.IGNORECASE)

# Set internal id: `id: "sv1"`
_SET_ID_RE = re.compile(r'id:\s*"(.*?)"', re.DOTALL)

# Official set abbreviation: `abbreviations: { official: "SVI" }`
_SET_ABBR_RE = re.compile(
    r'abbreviations:\s*\{\s*.*?\bofficial:\s*"(.*?)"', re.DOTALL | re.IGNORECASE)

# Card category: "Pokemon", "Trainer", "Energy"
_CATEGORY_RE = re.compile(r'category:\s*"(.*?)"', re.DOTALL)

# HP value
_HP_RE = re.compile(r'hp:\s*(\d+)', re.DOTALL)

# Trainer sub-type: "Supporter", "Item", "Stadium", …
_TRAINER_TYPE_RE = re.compile(r'trainerType:\s*"(.*?)"', re.DOTALL)

# Type list: `types: ["Fire", "Water"]`
_TYPES_RE = re.compile(r'types:\s*\[(.*?)\]', re.DOTALL)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read(path: str) -> Optional[str]:
    """Read a file, returning its text or None on error."""
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def _match(regex: re.Pattern, text: str) -> Optional[str]:
    """Return the first capture group of `regex` in `text`, or None."""
    m = regex.search(text)
    return m.group(1).strip() if m else None


def _get_set_metadata(path: str) -> Optional[dict]:
    """Extract set_id and set_abbreviation from a set .ts file."""
    content = _read(path)
    if not content:
        return None
    return {
        'set_id':           _match(_SET_ID_RE,   content),
        'set_abbreviation': _match(_SET_ABBR_RE, content),
    }


def _extract_card(path: str, set_abbreviation: str) -> Optional[dict]:
    """Extract all fields from a single card .ts file."""
    filename = os.path.basename(path)
    # Filenames are zero-padded: "001.ts" → card_number "001", id integer 1
    card_number_full = filename.replace('.ts', '')
    card_number_int  = card_number_full.lstrip('0') or '0'

    content = _read(path)
    if not content:
        return None

    # Parse the types list: `["Fire", "Water"]` → ['Fire', 'Water']
    types = None
    m = _TYPES_RE.search(content)
    if m:
        types = [t.strip().strip('"\'') for t in m.group(1).split(',') if t.strip()]

    hp_str = _match(_HP_RE, content)

    return {
        'id':                    int(card_number_int),
        'card_number':           card_number_full,
        'name_en':               _match(_CARD_NAME_RE,    content),
        'first_attack_name_en':  _match(_ATTACK_NAME_RE,  content),
        'category':              _match(_CATEGORY_RE,     content),
        'hp':                    int(hp_str) if hp_str else None,
        'trainer_type':          _match(_TRAINER_TYPE_RE, content),
        'types':                 types,
        'set_abbreviation':      set_abbreviation,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def build_database(
    input_folder: str,
    output_path: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int, int]:
    """Build card_data_lookup.json from a tcgdex `data/` folder.

    Parameters
    ----------
    input_folder : str
        Path to the `data/` directory inside the cloned tcgdex/cards-database
        repo (e.g. ``/path/to/cards-database/data``).
    output_path : str
        Where to write the resulting JSON file.
    on_progress : callable, optional
        Called with a status string after each set is processed.  Use this to
        update a GUI label from a background thread (via ``self.after(0, ...)``).

    Returns
    -------
    (success, num_cards, num_sets)
    """
    def emit(msg: str):
        if on_progress:
            on_progress(msg)

    if not os.path.isdir(input_folder):
        emit(f"Folder not found: {input_folder}")
        return False, 0, 0

    grouped: dict = {}  # { set_id: [card_dict, ...] }

    for dirpath, _dirs, filenames in os.walk(input_folder):
        # A card folder contains files named like "001.ts", "002.ts", …
        card_files = [f for f in filenames if re.match(r'^\d{3}\.ts$', f)]
        if not card_files:
            continue

        set_folder_name  = os.path.basename(dirpath)
        series_folder    = os.path.dirname(dirpath)
        # The set metadata file sits next to the card folder, with the same name
        set_ts_path      = os.path.join(series_folder, f'{set_folder_name}.ts')

        if not os.path.exists(set_ts_path):
            emit(f"  Skipping '{set_folder_name}' — no metadata file found")
            continue

        meta = _get_set_metadata(set_ts_path)
        if not meta or not meta['set_id']:
            emit(f"  Skipping '{set_folder_name}' — could not read set ID")
            continue

        set_id   = meta['set_id']
        set_abbr = meta['set_abbreviation'] or ''

        # The first Scarlet & Violet base set uses "SV" instead of "SVI"
        # in the metadata file; normalise it to match what is printed on cards.
        if set_abbr == 'SV':
            set_abbr = 'SVI'

        emit(f"Processing: {set_folder_name}  ({set_abbr})")

        if set_id not in grouped:
            grouped[set_id] = []

        for filename in sorted(card_files):
            card = _extract_card(os.path.join(dirpath, filename), set_abbr)
            if card:
                grouped[set_id].append(card)

    if not grouped:
        emit("No card data found — check that the folder contains .ts files.")
        return False, 0, 0

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grouped, f, indent=4, ensure_ascii=False)
    except Exception as e:
        emit(f"Error writing {output_path}: {e}")
        return False, 0, 0

    num_cards = sum(len(v) for v in grouped.values())
    num_sets  = len(grouped)
    emit(f"✓ {num_cards:,} cards across {num_sets} sets → {output_path}")
    return True, num_cards, num_sets


def get_set_abbreviations_from_db(db: dict) -> list:
    """Return a sorted list of unique set abbreviations found in a loaded database."""
    abbrs: set = set()
    for card_list in db.values():
        if not isinstance(card_list, list):
            continue
        for card in card_list:
            if isinstance(card, dict):
                sa = card.get('set_abbreviation')
                if sa:
                    abbrs.add(str(sa))
    return sorted(abbrs)
