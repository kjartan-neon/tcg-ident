"""
Camera capture and OCR scanning for TCG cards.

Wraps the webcam + DocTR/PaddleOCR pipeline from camScan.py into a
reusable class.  All slow operations (model loading, OCR) run in
background threads; results are delivered via the on_status callback.
"""

import cv2
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from ocr_processing import extract_card_info_from_text, find_card_in_database


class CardScanner:
    def __init__(self, on_status: Optional[Callable[[str], None]] = None):
        self._cap: Optional[cv2.VideoCapture] = None  # OpenCV camera handle
        self._doctr = None    # DocTR OCR predictor (primary engine)
        self._paddle = None   # PaddleOCR predictor (fallback engine)
        self._db = None       # Card database loaded from JSON
        self.on_status = on_status
        self.models_ready = False       # True once at least one OCR engine loaded
        self._scan_count = 0            # Lifetime successful scan counter
        self._card_count: int = 0       # Number of cards in the loaded database
        self._doctr_loaded: bool = False   # Set after DocTR loads successfully
        self._paddle_loaded: bool = False  # Set after Paddle loads successfully
        # Crop region in normalised 0–1 coordinates: (x1, y1, x2, y2).
        # Stored as a fraction of the frame size so the crop stays correct
        # when the window or camera resolution changes.
        self._crop_region: Optional[Tuple[float, float, float, float]] = None

    def _emit(self, msg: str):
        """Send a status message to the GUI log via the callback."""
        if self.on_status:
            self.on_status(msg)

    # ── Crop region ──────────────────────────────────────────────────────────

    @property
    def crop_region(self) -> Optional[Tuple[float, float, float, float]]:
        return self._crop_region

    @crop_region.setter
    def crop_region(self, value: Optional[Tuple[float, float, float, float]]):
        self._crop_region = value

    # ── Model / database loading ─────────────────────────────────────────────

    def load_models(self, on_done: Optional[Callable[[bool, bool], None]] = None):
        """Load DocTR and PaddleOCR in a background thread.

        OCR models are large (~50–200 MB) and take 10–30 seconds to load.
        We run the loading in a daemon thread so the GUI stays responsive.
        `on_done(doctr_ok, paddle_ok)` is called from that thread when
        loading finishes — the GUI must use `self.after(0, ...)` to update
        widgets from the callback.
        """
        def _load():
            doctr_ok = False
            paddle_ok = False
            try:
                from doctr.models import ocr_predictor
                self._emit("Loading DocTR (MobileNet)…")
                # MobileNet variants are small enough for fast CPU inference
                self._doctr = ocr_predictor(
                    det_arch='db_mobilenet_v3_large',
                    reco_arch='crnn_mobilenet_v3_small',
                    pretrained=True,
                )
                doctr_ok = True
                self._emit("DocTR ready.")
            except Exception as e:
                self._emit(f"DocTR error: {e}")

            try:
                from paddleocr import PaddleOCR
                self._emit("Loading PaddleOCR (fallback)…")
                self._paddle = PaddleOCR(
                    use_textline_orientation=True,
                    lang='en',
                    enable_mkldnn=False,   # MKL-DNN can cause crashes on some CPUs
                )
                paddle_ok = True
                self._emit("PaddleOCR ready.")
            except Exception as e:
                self._emit(f"PaddleOCR error: {e}")

            # models_ready is True as long as at least one engine loaded
            self.models_ready = doctr_ok or paddle_ok
            self._doctr_loaded = doctr_ok
            self._paddle_loaded = paddle_ok
            if on_done:
                on_done(doctr_ok, paddle_ok)

        threading.Thread(target=_load, daemon=True).start()

    def load_database(self, path: str):
        """Load the card JSON database from disk (runs on the calling thread)."""
        import json
        try:
            with open(path) as f:
                self._db = json.load(f)
            # The database is a dict of lists; count every card across all lists
            total = sum(
                len(v) for v in self._db.values() if isinstance(v, list)
            )
            self._card_count = total
            self._emit(f"Card database loaded: {path} ({total} cards)")
        except FileNotFoundError:
            self._emit(f"Card database not found: {path}")
        except Exception as e:
            self._emit(f"Database error: {e}")

    @property
    def database_loaded(self) -> bool:
        return self._db is not None

    # ── Camera ────────────────────────────────────────────────────────────────

    def open_camera(self, index: int = 0) -> bool:
        # Release any previously open capture before opening a new one
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._emit(f"Could not open camera {index}.")
            self._cap = None
            return False
        self._emit(f"Camera {index} opened.")
        return True

    def close_camera(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            self._emit("Camera closed.")

    @property
    def camera_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def camera_resolution(self):
        """Return (width, height) of the open camera, or None if not open."""
        if not self._cap or not self._cap.isOpened():
            return None
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def get_frame(self):
        """Grab one frame from the camera. Returns a BGR numpy array or None."""
        if not self._cap:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    # ── Image preprocessing ──────────────────────────────────────────────────

    def _crop(self, frame):
        """Crop the frame to the user-defined region, or return it unchanged.

        The crop region is stored in normalised 0–1 coordinates so that it
        stays valid even if the camera resolution changes. We convert to
        pixel coordinates at crop time.
        """
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if self._crop_region:
            x1, y1, x2, y2 = self._crop_region
            # Convert fractional coords to pixel indices
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            cropped = frame[py1:py2, px1:px2]
            # Guard against a degenerate (zero-area) crop
            return cropped if cropped.size > 0 else None
        return frame  # full frame when no region is set

    # ── OCR ──────────────────────────────────────────────────────────────────

    def _ocr_doctr(self, cropped) -> Optional[str]:
        """Run DocTR on a cropped frame. Returns a result string or None on failure."""
        if not self._doctr:
            return None
        try:
            # DocTR expects RGB; OpenCV gives BGR
            img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            result = self._doctr([img_rgb])
            # Walk the DocTR result tree: pages → blocks → lines → words
            texts = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        t = ' '.join(w.value for w in line.words).strip()
                        if t:
                            texts.append(t)
            info = extract_card_info_from_text(texts, self._db)
            return info if "FAILED" not in info else None
        except Exception as e:
            self._emit(f"DocTR scan error: {e}")
            return None

    def _ocr_paddle(self, cropped) -> Optional[str]:
        """Run PaddleOCR on a cropped frame. Returns a result string or None."""
        if not self._paddle:
            return None
        try:
            result = self._paddle.predict(cropped)
            texts = []
            # PaddleOCR v3 returns a list of result objects; handle both
            # attribute-style and dict-style access for forward compatibility.
            if result and isinstance(result, list) and result:
                obj = result[0]
                if hasattr(obj, 'rec_texts'):
                    texts = obj.rec_texts
                elif isinstance(obj, dict) and 'rec_texts' in obj:
                    texts = obj['rec_texts']
            info = extract_card_info_from_text(texts, self._db)
            return f"{info} [Paddle]" if "FAILED" not in info else None
        except Exception as e:
            self._emit(f"PaddleOCR error: {e}")
            return None

    def scan_frame(self, frame) -> str:
        """Run OCR on a single frame. Returns identified card info or a FAILED string.

        DocTR is tried first (faster). If it fails, PaddleOCR is used as
        a fallback. Both engines must fail for a FAILED result to be returned.
        """
        cropped = self._crop(frame)
        if cropped is None:
            return "--- FAILED: no crop ---"
        result = self._ocr_doctr(cropped)
        if result:
            return result
        result = self._ocr_paddle(cropped)
        if result:
            return result
        return "--- FAILED: both OCR engines ---"

    # ── Async scanning ───────────────────────────────────────────────────────

    def scan_async(
        self,
        max_attempts: int = 5,
        on_result: Optional[Callable[[str], None]] = None,
    ):
        """Capture and scan with retries in a background thread.

        OCR can take 1–3 seconds per frame. Running in a background thread
        keeps the GUI responsive during scanning. `on_result` is called from
        that thread — callers must use `self.after(0, ...)` to touch widgets.
        """
        def _scan():
            for i in range(max_attempts):
                self._emit(f"Scan attempt {i + 1}/{max_attempts}…")
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.5)
                    continue
                result = self.scan_frame(frame)
                if "FAILED" not in result:
                    self._scan_count += 1
                    self._emit(f"✓ Identified: {result}")
                    if on_result:
                        on_result(result)
                    return
                if i < max_attempts - 1:
                    time.sleep(0.5)
            msg = "--- FAILED: all attempts exhausted ---"
            self._emit(msg)
            if on_result:
                on_result(msg)

        threading.Thread(target=_scan, daemon=True).start()

    # ── Database helpers ──────────────────────────────────────────────────────

    def get_available_values(self) -> Dict[str, List[str]]:
        """Return unique types, categories, and set abbreviations from the database.

        Used to populate the Sort Settings listboxes so the user can pick
        which values route cards to each cart.
        """
        types: set = set()
        categories: set = set()
        sets: set = set()
        if self._db:
            for card_list in self._db.values():
                if not isinstance(card_list, list):
                    continue
                for card in card_list:
                    if not isinstance(card, dict):
                        continue
                    for t in (card.get('types') or []):
                        if t:
                            types.add(str(t))
                    tt = card.get('trainer_type')
                    if tt:
                        categories.add(str(tt))
                    sa = card.get('set_abbreviation')
                    if sa:
                        sets.add(str(sa))
        return {
            'types':    sorted(types),
            'category': sorted(categories),
            'set':      sorted(sets),
        }

    def find_card_data(self, result_str: str) -> Optional[dict]:
        """Look up card data from a result string like 'SVI-001: Bulbasaur'."""
        m = re.match(r'([A-Z]+)-(\d+)', result_str)
        if not m or not self._db:
            return None
        return find_card_in_database(m.group(1), m.group(2), self._db)

    def determine_cart(
        self,
        card_data: Optional[dict],
        criteria: dict,
        not_found_cart: int,
    ) -> int:
        """Return which cart (1/2/3) a card goes to based on the criteria dict.

        Criteria are checked in cart order (1, 2, 3). The first matching
        cart wins. If no criteria match, `not_found_cart` is returned.

        criteria = {
            1: {'field': 'types',    'values': ['Fire', 'Water']},
            2: {'field': 'category', 'values': ['stadium']},
            3: {'field': 'set',      'values': ['SVI']},
        }
        """
        if not card_data:
            return not_found_cart
        for cart_num in (1, 2, 3):
            c = criteria.get(cart_num, {})
            field  = c.get('field', '')
            # Lowercase both sides so comparison is case-insensitive
            values = {v.lower() for v in c.get('values', [])}
            if not values:
                continue
            if field == 'types':
                card_types = {t.lower() for t in (card_data.get('types') or [])}
                # Set intersection — at least one type must match
                if card_types & values:
                    return cart_num
            elif field == 'category':
                trainer = (card_data.get('trainer_type') or '').lower()
                if trainer in values:
                    return cart_num
            elif field == 'set':
                card_set = (card_data.get('set_abbreviation') or '').lower()
                if card_set in values:
                    return cart_num
        return not_found_cart
