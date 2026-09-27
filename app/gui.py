"""
Scramble Switch Sorter — GUI.

Theme: pastel gradient header, white panels, purple CTAs,
       light neutral secondary buttons, dark camera/log areas.
"""

import io
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, scrolledtext, ttk
from typing import Optional

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, 'source')
sys.path.insert(0, SRC_DIR)

try:
    import cv2
    from PIL import Image, ImageTk
    IMAGING_OK = True
except ImportError:
    IMAGING_OK = False

try:
    from version import __version__ as APP_VERSION
except ImportError:
    APP_VERSION = '1.0.0'

LOGO_PATH     = os.path.join(APP_DIR, 'assets', 'logo.svg')
SETTINGS_PATH = os.path.join(APP_DIR, 'settings.json')
LOGO_H        = 44


def _load_logo(height: int = LOGO_H) -> Optional['ImageTk.PhotoImage']:
    if not IMAGING_OK or not os.path.exists(LOGO_PATH):
        return None
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=LOGO_PATH, output_height=height)
        img = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def _find_db() -> str:
    for p in ['card_data_lookup.json',
              os.path.join(APP_DIR, '..', 'card_data_lookup.json')]:
        if os.path.exists(p):
            return os.path.abspath(p)
    return 'card_data_lookup.json'


from serial_controller import SorterController
from scanner import CardScanner

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG      = '#f0e6ff'
C_PANEL   = '#ffffff'
C_INPUT   = '#f9f0ff'
C_BORDER  = '#ddd0f0'
C_HOVER   = '#ede4fd'

C_TEXT    = '#1e1b30'
C_MUTED   = '#7060a0'
C_ACCENT  = '#7c3aed'
C_SUCCESS = '#10b981'

C_ACC     = '#7c3aed'
C_ACC_H   = '#6d28d9'
C_NEU     = '#ede9fe'
C_NEU_H   = '#ddd6fe'
C_NEU_FG  = '#4c1d95'

C_DARK    = '#1a1030'
C_DARK_FG = '#d8c8f0'
C_DARK_SEL= '#7c3aed'

GRAD_RGB  = [(249, 197, 209), (216, 180, 254), (167, 243, 208)]

FN        = 'Helvetica'
FONT_H1   = (FN, 16, 'bold')
FONT_H2   = (FN, 11, 'bold')
FONT_SECT = (FN, 10, 'bold')
FONT_BODY = (FN, 10)
FONT_SMALL= (FN, 9)
FONT_MONO = ('Courier', 10)


# ── Gradient helper ───────────────────────────────────────────────────────────

def _make_gradient(w: int, h: int, stops: list) -> 'Image.Image':
    img = Image.new('RGB', (max(w, 1), max(h, 1)))
    px  = img.load()
    n   = len(stops) - 1
    for x in range(max(w, 1)):
        t   = x / max(w - 1, 1)
        seg = min(int(t * n), n - 1)
        lt  = t * n - seg
        c   = tuple(int(stops[seg][i] + (stops[seg+1][i] - stops[seg][i]) * lt)
                    for i in range(3))
        for y in range(max(h, 1)):
            px[x, y] = c
    return img


# ── Rounded Canvas button ─────────────────────────────────────────────────────

class _RBtn(tk.Canvas):
    """Canvas button with smooth rounded corners.

    We use tk.Canvas instead of ttk.Button because ttk.Button cannot
    produce truly rounded corners — it is rectangle-clipped by the OS.
    Drawing a filled polygon on a Canvas and overlaying a text item
    gives us full control over the shape.
    """

    def __init__(self, parent, text='', command=None,
                 bg=C_ACC, fg='white', hover=C_ACC_H,
                 radius=10, font=FONT_BODY, **kw):
        self._bg  = bg
        self._hov = hover
        self._fg  = fg
        self._cmd = command

        # Measure the rendered text so we can size the canvas exactly.
        f   = tkfont.Font(family=font[0], size=font[1],
                          weight=font[2] if len(font) > 2 else 'normal')
        tw  = f.measure(text)
        th  = f.metrics('linespace')
        w   = tw + 36   # 18 px horizontal padding each side
        h   = th + 16   # 8 px vertical padding each side

        try:
            parent_bg = parent.cget('background')
        except Exception:
            parent_bg = C_BG

        # cursor='' gives the default system arrow; 'hand2' (pointing finger)
        # would feel wrong here because these buttons are not hyperlinks.
        super().__init__(parent, width=w, height=h,
                         bd=0, highlightthickness=0,
                         bg=parent_bg, cursor='', **kw)

        r   = radius
        # Each pair of points is a corner control-point for the smooth polygon.
        # smooth=True makes tkinter draw Bezier curves through the points,
        # which produces the rounded appearance.
        pts = [r,1, w-r,1, w-1,1, w-1,r, w-1,h-r,
               w-1,h-1, w-r,h-1, r,h-1, 1,h-1, 1,h-r, 1,r, 1,1]
        self._rid = self.create_polygon(pts, smooth=True, fill=bg, outline='')
        self._tid = self.create_text(w // 2, h // 2, text=text, fill=fg, font=font)

        # Bind hover/click events to the canvas AND both canvas items so that
        # clicking on the text or the polygon shape both fire the same handler.
        for ev, fn in (('<Enter>',           self._enter),
                       ('<Leave>',           self._leave),
                       ('<ButtonPress-1>',   self._press),
                       ('<ButtonRelease-1>', self._release)):
            self.bind(ev, fn)
            self.tag_bind(self._rid, ev, fn)
            self.tag_bind(self._tid, ev, fn)

    def _enter(self, _):   self.itemconfig(self._rid, fill=self._hov)
    def _leave(self, _):   self.itemconfig(self._rid, fill=self._bg)
    def _press(self, _):   self.itemconfig(self._rid, fill=self._hov)
    def _release(self, _):
        self.itemconfig(self._rid, fill=self._bg)
        if self._cmd:
            self._cmd()

    def config(self, text=None, command=None, **kw):
        if text is not None:
            self.itemconfig(self._tid, text=text)
        if command is not None:
            self._cmd = command
        if kw:
            super().config(**kw)


# ── Secondary button helper ───────────────────────────────────────────────────

def _nbtn(parent, text, command):
    return ttk.Button(parent, text=text, command=command, style='Neutral.TButton')


# ── App ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Scramble Switch Sorter')
        self.geometry('1280x820')
        self.minsize(960, 660)

        self._queue: queue.Queue = queue.Queue()
        self._controller = SorterController(on_status=self._on_status)
        self._scanner    = CardScanner(on_status=self._on_status)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._cam_img_id = None
        self._crop_rid   = None
        self._sel_rid    = None
        self._sel_start  = None          # (x, y) while mouse is held
        self._draw_mode  = False         # True while Set Scan Area is active

        self._hdr_photo: Optional[ImageTk.PhotoImage] = None
        self._logo_photo: Optional[ImageTk.PhotoImage] = None

        self._sort_active       = False
        self._cart_lb: dict     = {}
        self._cart_field_v: dict = {}
        self._not_found_cart_v  = tk.IntVar(value=3)

        self._pending_criteria: dict = {}
        self._save_timer_id          = None
        self._pulsing:   set = set()
        self._pulse_phase: int = 0

        self._apply_theme()
        self._build_ui()
        self._load_settings()
        self._setup_autosave()
        self._poll_queue()
        self._tick_camera()
        self._update_conn_indicators()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── Settings persistence ──────────────────────────────────────────────────

    def _schedule_save(self, *_):
        # Debounce: cancel any pending save timer, then start a fresh 800 ms
        # countdown.  This means we only write to disk once, 800 ms after the
        # last change — not on every keystroke while the user is still typing.
        if self._save_timer_id:
            self.after_cancel(self._save_timer_id)
        self._save_timer_id = self.after(800, self._save_settings)

    def _save_settings(self):
        raw_port   = self._port_v.get()
        port_clean = raw_port.split(' — ')[0].split(' — ')[0].strip()
        data = {
            'port':          port_clean,
            'baud':          self._baud_v.get(),
            'cam_idx':       self._cam_idx_v.get(),
            'db_path':       self._db_v.get(),
            'servo_open':    self._servo_open_v.get(),
            'servo_close':   self._servo_close_v.get(),
            'step_size':     self._step_size_v.get(),
            'step_delay':    self._delay_v.get(),
            'servo_manual':  self._servo_v.get(),
            'step_n':        self._stepn_v.get(),
            'cycles':        self._cycles_v.get(),
            'current_cart':  self._controller.current_cart,
            'not_found_cart':self._not_found_cart_v.get(),
            'settle':        self._settle_v.get(),
            'drop_wait':     self._drop_wait_v.get(),
            'auto_retries':  self._auto_retries_v.get(),
            'scan_retries':  self._retries_v.get(),
            'crop_region':   list(self._scanner.crop_region)
                             if self._scanner.crop_region else None,
            'cart_criteria': {
                str(n): {
                    'field':  self._cart_field_v[n].get()
                              if n in self._cart_field_v else 'types',
                    'values': [self._cart_lb[n].get(i)
                               for i in self._cart_lb[n].curselection()]
                              if n in self._cart_lb else [],
                }
                for n in (1, 2, 3)
            },
        }
        try:
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        except Exception:
            return

        def _sv(var, key):
            val = data.get(key)
            if val is not None:
                try:
                    var.set(val)
                except Exception:
                    pass

        _sv(self._port_v,          'port')
        _sv(self._baud_v,          'baud')
        _sv(self._cam_idx_v,       'cam_idx')
        _sv(self._db_v,            'db_path')
        _sv(self._servo_open_v,    'servo_open')
        _sv(self._servo_close_v,   'servo_close')
        _sv(self._step_size_v,     'step_size')
        _sv(self._delay_v,         'step_delay')
        _sv(self._servo_v,         'servo_manual')
        _sv(self._stepn_v,         'step_n')
        _sv(self._cycles_v,        'cycles')
        _sv(self._not_found_cart_v,'not_found_cart')
        _sv(self._settle_v,        'settle')
        _sv(self._drop_wait_v,     'drop_wait')
        _sv(self._auto_retries_v,  'auto_retries')
        _sv(self._retries_v,       'scan_retries')

        cart = data.get('current_cart')
        if cart is not None:
            try:
                self._controller.current_cart = int(cart)
                self._cur_cart_v.set(int(cart))
                self._sync_cart_label()
            except (ValueError, TypeError):
                pass

        cr = data.get('crop_region')
        if cr and len(cr) == 4:
            self._scanner.crop_region = tuple(cr)
            self.after(300, lambda: self._restore_crop_overlay(cr))

        cc = data.get('cart_criteria', {})
        # Listbox selections cannot be restored until the database is loaded
        # (the listbox items don't exist yet).  Store them in _pending_criteria
        # and apply them the first time _populate_cart_lb runs after DB load.
        self._pending_criteria = {int(k): v for k, v in cc.items()}
        for n, v in self._pending_criteria.items():
            if n in self._cart_field_v:
                self._cart_field_v[n].set(v.get('field', 'types'))

        self._log('Settings loaded.')

    def _restore_crop_overlay(self, cr: list):
        # scanner.crop_region is already set; _tick_camera draws the rect.
        # Just update the label.
        self._crop_lbl.config(
            text=f'Area: ({cr[0]:.2f},{cr[1]:.2f})–({cr[2]:.2f},{cr[3]:.2f})',
            foreground=C_SUCCESS)

    def _setup_autosave(self):
        # trace_add('write', ...) registers a callback that fires every time
        # the tkinter StringVar/IntVar value changes — even from code, not just
        # user input.  We use this to trigger the debounced save automatically.
        for var in (self._port_v, self._baud_v, self._cam_idx_v, self._db_v,
                    self._servo_open_v, self._servo_close_v, self._step_size_v,
                    self._delay_v, self._servo_v, self._stepn_v, self._cycles_v,
                    self._settle_v, self._drop_wait_v, self._auto_retries_v,
                    self._retries_v, self._not_found_cart_v, self._cur_cart_v):
            var.trace_add('write', self._schedule_save)
        for fv in self._cart_field_v.values():
            fv.trace_add('write', self._schedule_save)
        # Listboxes don't have a tkinter variable; bind to the selection event.
        for lb in self._cart_lb.values():
            lb.bind('<<ListboxSelect>>', self._schedule_save)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        self.configure(bg=C_BG)
        s = ttk.Style(self)
        s.theme_use('clam')

        s.configure('TFrame',         background=C_PANEL)
        s.configure('BG.TFrame',      background=C_BG)
        s.configure('TLabel',         background=C_PANEL, foreground=C_TEXT, font=FONT_BODY)
        s.configure('Muted.TLabel',   background=C_PANEL, foreground=C_MUTED, font=FONT_SMALL)
        s.configure('Accent.TLabel',  background=C_PANEL, foreground=C_ACCENT, font=FONT_SECT)
        s.configure('BigCart.TLabel', background=C_PANEL, foreground=C_ACCENT, font=FONT_H2)
        s.configure('BG.TLabel',      background=C_BG,    foreground=C_TEXT,   font=FONT_BODY)

        s.configure('TEntry',
            fieldbackground=C_INPUT, foreground=C_TEXT,
            bordercolor=C_BORDER, lightcolor=C_BORDER, darkcolor=C_BORDER,
            insertcolor=C_TEXT, padding=4)
        s.configure('TCombobox',
            fieldbackground=C_INPUT, foreground=C_TEXT,
            background=C_NEU, selectbackground=C_ACC, selectforeground='white', padding=4)
        s.map('TCombobox', fieldbackground=[('readonly', C_INPUT)])

        s.configure('TScrollbar',
            background=C_NEU, troughcolor=C_BG,
            arrowcolor=C_ACCENT, bordercolor=C_BG, darkcolor=C_BG, lightcolor=C_BG)
        s.map('TScrollbar', background=[('active', C_NEU_H)])

        for pfx, bg in (('', C_PANEL), ('BG.', C_BG)):
            s.configure(f'{pfx}TLabelframe',
                background=bg, bordercolor=C_BORDER, relief='solid', borderwidth=1)
            s.configure(f'{pfx}TLabelframe.Label',
                background=bg, foreground=C_ACCENT, font=FONT_SECT, padding=(4, 2))

        s.configure('TNotebook', background=C_BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        s.configure('TNotebook.Tab',
            background=C_NEU, foreground=C_MUTED,
            padding=[14, 8], font=FONT_BODY, borderwidth=0)
        s.map('TNotebook.Tab',
            background=[('selected', C_PANEL), ('active', C_HOVER)],
            foreground=[('selected', C_ACCENT), ('active', C_ACC)],
            font=[('selected', (FN, 10, 'bold'))])

        s.configure('Neutral.TButton',
            background=C_NEU, foreground=C_NEU_FG, font=FONT_BODY,
            padding=[12, 7], relief='flat', borderwidth=0, focuscolor=C_NEU)
        s.map('Neutral.TButton',
            background=[('active', C_NEU_H), ('pressed', C_NEU_H)],
            foreground=[('active', C_NEU_FG), ('pressed', C_NEU_FG)])

    # ── UI skeleton ───────────────────────────────────────────────────────────

    def _build_ui(self):
        # The window is a 2-column, 3-row grid:
        #   col 0: camera (rows 1) + log (row 2)
        #   col 1: notebook spanning all content rows (rows 1–2)
        #
        # weight= controls how spare space is distributed when the window is
        # resized.  weight=0 means the row/column never grows; higher weight
        # means it claims a proportionally larger share of the extra space.
        self.columnconfigure(0, weight=3)           # camera column gets 3× more space
        self.columnconfigure(1, weight=1, minsize=300)
        self.rowconfigure(0, weight=0)              # header — fixed height
        self.rowconfigure(1, weight=4)              # camera  |  notebook top
        self.rowconfigure(2, weight=1, minsize=130) # log     |  notebook bottom

        self._build_header()

        # ── Camera ────────────────────────────────────────────────────────────
        cam_wrap = ttk.LabelFrame(
            self, text='Camera  ·  drag to set OCR scan area',
            style='BG.TLabelframe')
        cam_wrap.grid(row=1, column=0, sticky='nsew', padx=(8, 4), pady=(8, 4))
        cam_wrap.rowconfigure(0, weight=1)
        cam_wrap.columnconfigure(0, weight=1)

        self._cam_canvas = tk.Canvas(
            cam_wrap, bg=C_DARK, cursor='crosshair',
            highlightthickness=2, highlightbackground=C_BORDER)
        self._cam_canvas.grid(row=0, column=0, sticky='nsew')
        self._cam_canvas.bind('<ButtonPress-1>',   self._cam_press)
        self._cam_canvas.bind('<B1-Motion>',        self._cam_drag)
        self._cam_canvas.bind('<ButtonRelease-1>',  self._cam_release)

        self._cam_no_text = self._cam_canvas.create_text(
            5, 5, anchor='nw', text='No camera — open one on Connect tab',
            fill='#4a3060', font=FONT_H1, tags='notext')

        self._result_lbl = tk.Label(
            cam_wrap, text='', bg=C_DARK, fg='#c4b5fd',
            font=(FN, 13, 'bold'), anchor='w', padx=12, pady=6)
        self._result_lbl.grid(row=1, column=0, sticky='ew')

        # ── Log (under camera only) ───────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text='Log', style='BG.TLabelframe')
        log_frame.grid(row=2, column=0, sticky='nsew', padx=(8, 4), pady=(0, 8))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log_box = scrolledtext.ScrolledText(
            log_frame, height=7, state='disabled',
            font=FONT_MONO, bg=C_DARK, fg=C_DARK_FG,
            insertbackground=C_DARK_FG,
            selectbackground=C_DARK_SEL, selectforeground='white',
            relief='flat', borderwidth=0)
        self._log_box.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)

        # ── Notebook (spans both rows — full height) ──────────────────────────
        nb_wrap = ttk.Frame(self, style='BG.TFrame')
        # rowspan=2 makes the notebook occupy rows 1 AND 2 in column 1, so it
        # stretches from just below the header all the way to the window bottom,
        # matching the combined height of the camera + log on the left.
        nb_wrap.grid(row=1, column=1, rowspan=2, sticky='nsew',
                     padx=(4, 8), pady=8)
        nb_wrap.rowconfigure(0, weight=1)
        nb_wrap.columnconfigure(0, weight=1)

        nb = ttk.Notebook(nb_wrap)
        nb.grid(row=0, column=0, sticky='nsew')

        self._build_auto_tab(nb)
        self._build_database_tab(nb)
        self._build_conn_tab(nb)
        self._build_sorter_tab(nb)
        self._build_scanner_tab(nb)

    # ── Scrollable tab helper ─────────────────────────────────────────────────

    def _make_tab(self, nb: ttk.Notebook, title: str) -> ttk.Frame:
        """Add a vertically-scrollable tab to nb and return the inner frame.

        tkinter has no built-in scrollable Frame widget.  The standard
        workaround is to embed a ttk.Frame inside a tk.Canvas using
        create_window(), then attach a Scrollbar to the canvas's yview.
        Callers put their widgets in the returned inner frame as normal.
        """
        outer = ttk.Frame(nb)
        nb.add(outer, text=title)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        c  = tk.Canvas(outer, bg=C_PANEL, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(outer, orient='vertical', command=c.yview)
        c.configure(yscrollcommand=sb.set)
        c.grid(row=0, column=0, sticky='nsew')
        sb.grid(row=0, column=1, sticky='ns')

        inner = ttk.Frame(c, padding=(10, 8))
        win   = c.create_window(0, 0, anchor='nw', window=inner)

        # When the canvas is resized (e.g. user drags window edge), force the
        # embedded inner frame to match the canvas width so it fills the tab.
        c.bind('<Configure>', lambda e: c.itemconfig(win, width=e.width))
        # When widgets are added/removed from inner, recalculate the total
        # scrollable area so the scrollbar knows how far the user can scroll.
        inner.bind('<Configure>', lambda e: c.configure(scrollregion=c.bbox('all')))

        # macOS sends <MouseWheel> with e.delta; Linux sends <Button-4/5>.
        def _wheel(e):
            delta = -1 if e.delta > 0 else 1
            c.yview_scroll(delta, 'units')

        def _wheel_lin(e):
            c.yview_scroll(-1 if e.num == 4 else 1, 'units')

        # Bind to both the canvas and the inner frame so scrolling works
        # regardless of which one the mouse pointer is over.
        for w in (c, inner):
            w.bind('<MouseWheel>', _wheel)
            w.bind('<Button-4>',   _wheel_lin)
            w.bind('<Button-5>',   _wheel_lin)

        return inner

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        self._hdr_canvas = tk.Canvas(
            self, height=62, bd=0, highlightthickness=0, bg=C_BG)
        self._hdr_canvas.grid(row=0, column=0, columnspan=2, sticky='ew')
        self._hdr_canvas.bind('<Configure>', self._redraw_header)
        self._logo_photo = _load_logo(LOGO_H)
        self.after(50, self._redraw_header)

    def _redraw_header(self, event=None):
        # Canvas images do not stretch automatically when the window is resized,
        # so we must regenerate the gradient bitmap at the new canvas dimensions
        # every time a <Configure> event fires (which means the size changed).
        w = self._hdr_canvas.winfo_width()  or 1280
        h = self._hdr_canvas.winfo_height() or 62
        self._hdr_canvas.delete('all')   # wipe everything; we redraw from scratch
        if IMAGING_OK:
            img = _make_gradient(w, h, GRAD_RGB)
            self._hdr_photo = ImageTk.PhotoImage(img)
            self._hdr_canvas.create_image(0, 0, anchor='nw', image=self._hdr_photo)
        else:
            self._hdr_canvas.configure(bg='#d8b4fe')

        x = 14
        if self._logo_photo:
            self._hdr_canvas.create_image(x, h // 2, anchor='w', image=self._logo_photo)
            x += self._logo_photo.width() + 10

        self._hdr_canvas.create_text(
            x, h // 2 - 8, anchor='w', text='Scramble Switch Sorter',
            font=FONT_H1, fill=C_TEXT)
        self._hdr_canvas.create_text(
            x, h // 2 + 10, anchor='w', text='Pokémon TCG Automation',
            font=FONT_SMALL, fill=C_MUTED)

        dot = '#10b981' if self._controller.connected else '#9ca3af'
        cx  = w - 18
        self._hdr_canvas.create_oval(cx-6, h//2-6, cx+6, h//2+6, fill=dot, outline='')
        self._hdr_canvas.create_text(
            cx - 10, h // 2, anchor='e',
            text='connected' if self._controller.connected else 'disconnected',
            font=FONT_SMALL, fill=C_MUTED)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _on_status(self, msg: str):
        # Called from background threads (scanner, controller, etc.).
        # We CANNOT call tkinter methods directly from a background thread —
        # tkinter is single-threaded and doing so causes crashes or silent
        # corruption.  Instead we put the message in a thread-safe Queue and
        # let the main thread drain it on a timer (see _poll_queue).
        self._queue.put(msg)

    def _poll_queue(self):
        # Drain every pending message that arrived from background threads
        # since the last poll.  get_nowait() raises queue.Empty when the
        # queue is empty, which is our exit condition.
        try:
            while True:
                self._log(self._queue.get_nowait())
        except queue.Empty:
            pass
        # Re-schedule ourselves 50 ms from now — this creates a repeating
        # timer on the main thread without blocking it.
        self.after(50, self._poll_queue)

    def _log(self, msg: str):
        self._log_box.config(state='normal')
        self._log_box.insert('end', msg + '\n')
        self._log_box.see('end')
        self._log_box.config(state='disabled')

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _le(self, parent, label, var, row, w=18):
        # Shortcut: render a label + text-entry pair on the given grid row.
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=(4, 4), pady=3)
        ttk.Entry(parent, textvariable=var, width=w).grid(
            row=row, column=1, sticky='ew', padx=(0, 4), pady=3)

    def _section(self, parent, text, row):
        # Shortcut: render a full-width accent-coloured section heading and
        # return the next row index so callers can chain: r = self._section(...)
        ttk.Label(parent, text=text, style='Accent.TLabel').grid(
            row=row, column=0, columnspan=2, sticky='w', padx=4, pady=(16, 4))
        return row + 1

    def _rbf(self, parent, row, *buttons):
        """Right-aligned button row. buttons: (text, cmd, 'accent'|'neutral'|'success')

        Shortcut: packs one or more buttons into a right-aligned frame on the
        given grid row, choosing _RBtn (rounded canvas) or _nbtn (neutral ttk)
        based on the style tag.  Returns the next row index.
        """
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky='e', padx=4, pady=(4, 2))
        for i, (text, cmd, kind) in enumerate(buttons):
            if kind == 'accent':
                _RBtn(f, text, cmd).grid(row=0, column=i, padx=(0, 4))
            elif kind == 'success':
                _RBtn(f, text, cmd, bg=C_SUCCESS, fg='white',
                      hover='#0ea572').grid(row=0, column=i, padx=(0, 4))
            else:
                _nbtn(f, text, cmd).grid(row=0, column=i, padx=(0, 4))
        return row + 1

    def _need_conn(self) -> bool:
        if not self._controller.connected:
            self._log('Not connected.'); return False
        return True

    # ── Tab: Connect ─────────────────────────────────────────────────────────

    def _build_conn_tab(self, nb):
        p = self._make_tab(nb, 'Connect')
        p.columnconfigure(1, weight=1)
        r = 0

        # Sorter
        r = self._section(p, 'Sorter', r)
        self._port_v = tk.StringVar(value='')
        self._baud_v = tk.StringVar(value='115200')

        ttk.Label(p, text='Serial port:').grid(
            row=r, column=0, sticky='w', padx=(4, 4), pady=3)
        pf = ttk.Frame(p); pf.grid(row=r, column=1, sticky='ew', padx=(0, 4), pady=3)
        pf.columnconfigure(0, weight=1)
        self._port_cb = ttk.Combobox(pf, textvariable=self._port_v, width=20)
        self._port_cb.grid(row=0, column=0, sticky='ew')
        self._port_cb.bind('<<ComboboxSelected>>', self._on_port_selected)
        _nbtn(pf, '↺', self._scan_ports).grid(row=0, column=1, padx=(4, 0))
        r += 1

        self._le(p, 'Baud rate:', self._baud_v, r, w=10); r += 1

        self._conn_lbl = ttk.Label(p, text='● Disconnected', foreground='#9ca3af')
        self._conn_lbl.grid(row=r, column=0, columnspan=2,
                             sticky='w', padx=4, pady=(0, 2)); r += 1
        r = self._rbf(p, r,
            ('Ping',       self._ping,        'neutral'),
            ('Disconnect', self._disconnect,  'neutral'),
            ('Connect',    self._connect,     'accent'))

        # Camera
        r = self._section(p, 'Camera', r)
        self._cam_idx_v = tk.StringVar(value='0')

        ttk.Label(p, text='Camera:').grid(
            row=r, column=0, sticky='w', padx=(4, 4), pady=3)
        cf = ttk.Frame(p); cf.grid(row=r, column=1, sticky='ew', padx=(0, 4), pady=3)
        cf.columnconfigure(0, weight=1)
        self._cam_cb = ttk.Combobox(cf, textvariable=self._cam_idx_v, values=['0'], width=20)
        self._cam_cb.grid(row=0, column=0, sticky='ew')
        self._cam_cb.bind('<<ComboboxSelected>>', self._on_cam_selected)
        _nbtn(cf, '↺', self._scan_cameras).grid(row=0, column=1, padx=(4, 0))
        r += 1

        self._cam_conn_lbl = ttk.Label(p, text='● Not open', foreground='#9ca3af')
        self._cam_conn_lbl.grid(row=r, column=0, columnspan=2,
                                 sticky='w', padx=4, pady=(0, 2)); r += 1
        r = self._rbf(p, r,
            ('Close Camera', self._close_cam, 'neutral'),
            ('Open Camera',  self._open_cam,  'accent'))

        # OCR
        r = self._section(p, 'OCR Models', r)
        self._models_lbl = ttk.Label(p, text='● Not loaded', foreground='#9ca3af')
        self._models_lbl.grid(row=r, column=0, columnspan=2,
                               sticky='w', padx=4, pady=(0, 2)); r += 1
        r = self._rbf(p, r, ('Load OCR Models', self._load_models, 'accent'))

        # Database
        r = self._section(p, 'Card Database', r)
        self._db_v = tk.StringVar(value=_find_db())

        dbf = ttk.Frame(p)
        dbf.grid(row=r, column=0, columnspan=2, sticky='ew', padx=4, pady=3)
        dbf.columnconfigure(0, weight=1)
        ttk.Entry(dbf, textvariable=self._db_v).grid(row=0, column=0, sticky='ew')
        _nbtn(dbf, '…', self._browse_db).grid(row=0, column=1, padx=(4, 0))
        r += 1

        self._db_status_lbl = ttk.Label(p, text='● Not loaded', foreground='#9ca3af')
        self._db_status_lbl.grid(row=r, column=0, columnspan=2,
                                  sticky='w', padx=4, pady=(0, 2)); r += 1
        r = self._rbf(p, r, ('Load Database', self._load_db, 'accent'))

        ttk.Label(p, text=f'v{APP_VERSION}', style='Muted.TLabel').grid(
            row=r, column=0, columnspan=2, sticky='w', padx=4, pady=(24, 8))

        self.after(100, self._scan_ports)

    # ── Tab: Sorter ───────────────────────────────────────────────────────────

    def _build_sorter_tab(self, nb):
        p = self._make_tab(nb, 'Sorter')
        p.columnconfigure(1, weight=1)
        r = 0

        self._servo_open_v  = tk.StringVar(value=str(self._controller.servo_open_angle))
        self._servo_close_v = tk.StringVar(value=str(self._controller.servo_close_angle))
        self._step_size_v   = tk.StringVar(value=str(self._controller.cart_step_size))
        self._delay_v       = tk.StringVar(value=str(self._controller.step_delay_us))
        self._servo_v       = tk.StringVar(value='90')
        self._stepn_v       = tk.StringVar(value='100')
        self._cycles_v      = tk.StringVar(value='1')
        self._cur_cart_v    = tk.IntVar(value=1)

        # Gate
        r = self._section(p, 'Gate', r)
        self._le(p, 'Open angle °:',  self._servo_open_v,  r, w=5); r += 1
        self._le(p, 'Close angle °:', self._servo_close_v, r, w=5); r += 1
        r = self._rbf(p, r,
            ('Close (hold)', self._close_gate, 'neutral'),
            ('Open (feed)',  self._open_gate,  'accent'))

        # Movement
        r = self._section(p, 'Movement', r)
        self._le(p, 'Cart step size:', self._step_size_v, r, w=8); r += 1
        self._le(p, 'Delay (µs):',     self._delay_v,     r, w=8); r += 1
        self._le(p, 'Servo °:',        self._servo_v,     r, w=5); r += 1
        r = self._rbf(p, r,
            ('Set Servo',      self._set_servo,      'neutral'),
            ('Apply Settings', self._apply_settings, 'neutral'))

        # Position
        r = self._section(p, 'Position', r)

        ttk.Label(p, text='Current cart:').grid(
            row=r, column=0, sticky='w', padx=(4, 4), pady=3)
        spf = ttk.Frame(p)
        spf.grid(row=r, column=1, sticky='e', padx=(0, 4)); r += 1
        tk.Spinbox(spf, from_=1, to=3, textvariable=self._cur_cart_v, width=4,
                   bg=C_INPUT, fg=C_TEXT, relief='flat',
                   highlightthickness=1, highlightbackground=C_BORDER,
                   buttonbackground=C_NEU).pack(side='left')
        _nbtn(spf, 'Set', self._set_current_cart).pack(side='left', padx=(6, 0))

        self._cur_cart_lbl = ttk.Label(p, text='Position: cart 1', style='BigCart.TLabel')
        self._cur_cart_lbl.grid(row=r, column=0, columnspan=2,
                                 sticky='w', padx=4, pady=4); r += 1

        r = self._rbf(p, r,
            ('→ Cart 1', self._cart1, 'neutral'),
            ('→ Cart 2', self._cart2, 'neutral'),
            ('→ Cart 3', self._cart3, 'neutral'))

        self._le(p, 'Step N:', self._stepn_v, r, w=8); r += 1
        r = self._rbf(p, r,
            ('← Back N',    self._back_n, 'neutral'),
            ('Forward N →', self._fwd_n,  'neutral'))

        # Automation
        r = self._section(p, 'Automation', r)
        r = self._rbf(p, r,
            ('Stop',    self._stop,    'neutral'),
            ('Release', self._release, 'neutral'),
            ('Feed',    self._feed,    'accent'))

        ttk.Label(p, text='Full cycles:').grid(
            row=r, column=0, sticky='w', padx=(4, 4), pady=3)
        cyf = ttk.Frame(p)
        cyf.grid(row=r, column=1, sticky='e', padx=(0, 4)); r += 1
        tk.Spinbox(cyf, from_=1, to=99, textvariable=self._cycles_v, width=5,
                   bg=C_INPUT, fg=C_TEXT, relief='flat',
                   highlightthickness=1, highlightbackground=C_BORDER,
                   buttonbackground=C_NEU).pack(side='left')
        _RBtn(cyf, 'Run', self._full_cycle).pack(side='left', padx=(8, 0))

    # ── Tab: Scanner ──────────────────────────────────────────────────────────

    def _build_scanner_tab(self, nb):
        p = self._make_tab(nb, 'Scanner')
        p.columnconfigure(1, weight=1)
        r = 0
        self._retries_v = tk.StringVar(value='5')

        # Scan area
        r = self._section(p, 'Scan Area', r)

        self._crop_mode_btn = _RBtn(p, 'Set Scan Area', self._toggle_crop_mode)
        self._crop_mode_btn.grid(row=r, column=0, columnspan=2,
                                  sticky='e', padx=4, pady=(0, 4)); r += 1

        self._crop_lbl = ttk.Label(p, text='Area: full frame',
                                    style='Muted.TLabel')
        self._crop_lbl.grid(row=r, column=0, columnspan=2,
                             sticky='w', padx=4, pady=(0, 4)); r += 1

        r = self._rbf(p, r, ('Clear — use full frame', self._clear_crop, 'neutral'))

        # Scan
        r = self._section(p, 'Scan', r)
        self._le(p, 'Retries:', self._retries_v, r, w=5); r += 1
        r = self._rbf(p, r,
            ('Scan Card Now',        self._scan_now,          'accent'),
            ('Scan & Identify Cart', self._scan_and_identify, 'accent'))

        self._scan_res_lbl = ttk.Label(p, text='—', foreground=C_MUTED,
                                        wraplength=260, justify='left')
        self._scan_res_lbl.grid(row=r, column=0, columnspan=2,
                                  sticky='w', padx=4, pady=4)

    # ── Tab: Sort Settings ────────────────────────────────────────────────────

    def _build_database_tab(self, nb):
        p = self._make_tab(nb, 'Sort Settings')
        p.columnconfigure(0, weight=1)
        r = 0

        _nbtn(p, '↺ Refresh Lists', self._refresh_criteria_lists).grid(
            row=r, column=0, sticky='e', padx=4, pady=(4, 8)); r += 1

        sub_nb = ttk.Notebook(p)
        sub_nb.grid(row=r, column=0, sticky='ew', pady=3); r += 1

        for cart_num in (1, 2, 3):
            tab = ttk.Frame(sub_nb, padding=8)
            sub_nb.add(tab, text=f'Cart {cart_num}')
            tab.columnconfigure(1, weight=1)

            fv = tk.StringVar(value='types')
            self._cart_field_v[cart_num] = fv

            ttk.Label(tab, text='Sort by:').grid(
                row=0, column=0, sticky='w', padx=4, pady=4)
            cb = ttk.Combobox(tab, textvariable=fv,
                               values=['types', 'category', 'set'],
                               state='readonly', width=12)
            cb.grid(row=0, column=1, sticky='w', padx=4, pady=4)
            cb.bind('<<ComboboxSelected>>',
                    lambda e, n=cart_num: self._on_field_change(n))

            lf = ttk.Frame(tab)
            lf.grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=4)
            lf.columnconfigure(0, weight=1)

            lb = tk.Listbox(lf, selectmode=tk.MULTIPLE, height=8,
                             exportselection=False, font=FONT_BODY,
                             bg=C_INPUT, fg=C_TEXT,
                             selectbackground=C_ACC, selectforeground='white',
                             relief='flat', borderwidth=0,
                             highlightthickness=1, highlightbackground=C_BORDER)
            sb = ttk.Scrollbar(lf, orient='vertical', command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            lb.grid(row=0, column=0, sticky='ew')
            sb.grid(row=0, column=1, sticky='ns')
            self._cart_lb[cart_num] = lb

            ttk.Label(tab, text='Ctrl+click to select multiple',
                      style='Muted.TLabel').grid(
                row=2, column=0, columnspan=2, sticky='w', padx=4, pady=(0, 2))

        nff = ttk.Frame(p)
        nff.grid(row=r, column=0, sticky='ew', pady=(8, 4)); r += 1
        ttk.Label(nff, text='No match → Cart:').pack(side='left', padx=4)
        tk.Spinbox(nff, from_=1, to=3, textvariable=self._not_found_cart_v, width=4,
                   bg=C_INPUT, fg=C_TEXT, relief='flat',
                   highlightthickness=1, highlightbackground=C_BORDER,
                   buttonbackground=C_NEU).pack(side='left')
        ttk.Label(nff, text='fallback for unmatched cards',
                  style='Muted.TLabel').pack(side='left', padx=8)

    # ── Tab: Sort & Scan ──────────────────────────────────────────────────────

    def _build_auto_tab(self, nb):
        p = self._make_tab(nb, 'Sort & Scan')
        p.columnconfigure(1, weight=1)
        r = 0
        self._settle_v       = tk.StringVar(value='1.0')
        self._drop_wait_v    = tk.StringVar(value='2.0')
        self._auto_retries_v = tk.StringVar(value='5')

        # Connection status indicators
        r = self._section(p, 'Connections', r)

        self._ss_serial_lbl = ttk.Label(p, text='● Serial — not connected',
                                         foreground='#9ca3af')
        self._ss_serial_lbl.grid(row=r, column=0, columnspan=2,
                                  sticky='w', padx=4, pady=2); r += 1

        self._ss_cam_lbl = ttk.Label(p, text='● Camera — not open',
                                      foreground='#9ca3af')
        self._ss_cam_lbl.grid(row=r, column=0, columnspan=2,
                               sticky='w', padx=4, pady=2); r += 1

        self._ss_ocr_lbl = ttk.Label(p, text='● OCR models — not loaded',
                                      foreground='#9ca3af')
        self._ss_ocr_lbl.grid(row=r, column=0, columnspan=2,
                               sticky='w', padx=4, pady=2); r += 1

        self._ss_db_lbl = ttk.Label(p, text='● Database — not loaded',
                                     foreground='#9ca3af')
        self._ss_db_lbl.grid(row=r, column=0, columnspan=2,
                              sticky='w', padx=4, pady=2); r += 1

        r = self._rbf(p, r, ('Connect All', self._connect_all, 'accent'))

        # Timing
        r = self._section(p, 'Timing', r)
        self._le(p, 'Settle time (s):', self._settle_v,       r, w=6); r += 1
        self._le(p, 'Drop wait (s):',   self._drop_wait_v,    r, w=6); r += 1
        self._le(p, 'Scan retries:',    self._auto_retries_v, r, w=6); r += 1

        # Control
        r = self._section(p, 'Control', r)
        r = self._rbf(p, r,
            ('■  Stop',              self._stop_auto,  'neutral'),
            ('▶  Start Sort & Scan', self._start_auto, 'success'))

        self._auto_status_lbl = ttk.Label(p, text='Idle',
                                           foreground=C_MUTED, wraplength=260)
        self._auto_status_lbl.grid(row=r, column=0, columnspan=2,
                                    sticky='w', padx=4, pady=4); r += 1

        self._auto_count_lbl = ttk.Label(p, text='Cards sorted: 0',
                                          style='BigCart.TLabel')
        self._auto_count_lbl.grid(row=r, column=0, columnspan=2,
                                   sticky='w', padx=4, pady=4); r += 1

    # ── Camera: drag-to-crop ──────────────────────────────────────────────────

    def _cam_press(self, event):
        # Record the pixel position where the drag started.
        self._sel_start = (event.x, event.y)
        # Hide any rectangle left over from the previous drag so it doesn't
        # confuse the user while they are drawing a new one.
        if self._sel_rid:
            self._cam_canvas.itemconfig(self._sel_rid, state='hidden')

    def _cam_drag(self, event):
        if not self._sel_start:
            return
        x0, y0 = self._sel_start
        if self._sel_rid:
            self._cam_canvas.coords(self._sel_rid, x0, y0, event.x, event.y)
            self._cam_canvas.itemconfig(self._sel_rid, state='normal')
            self._cam_canvas.tag_raise(self._sel_rid)
        else:
            self._sel_rid = self._cam_canvas.create_rectangle(
                x0, y0, event.x, event.y,
                outline='#f9c5d1', width=2, dash=(5, 3))

    def _cam_release(self, event):
        if not self._sel_start:
            return
        x0, y0 = self._sel_start
        x1, y1 = event.x, event.y
        self._sel_start = None
        cw = self._cam_canvas.winfo_width()
        ch = self._cam_canvas.winfo_height()
        # Ignore accidental single-clicks or tiny drags (< 8 px) that would
        # produce a useless crop region.
        if cw > 10 and ch > 10 and abs(x1-x0) > 8 and abs(y1-y0) > 8:
            # Normalise pixel coords to 0-1 fractions of the canvas size.
            # Storing fractions instead of pixels means the crop region stays
            # correct when the window is resized later — _tick_camera converts
            # them back to pixels every frame using the current canvas size.
            nx1 = min(x0, x1) / cw;  ny1 = min(y0, y1) / ch
            nx2 = max(x0, x1) / cw;  ny2 = max(y0, y1) / ch
            self._scanner.crop_region = (nx1, ny1, nx2, ny2)
            self._crop_lbl.config(
                text=f'Area: ({nx1:.2f},{ny1:.2f})–({nx2:.2f},{ny2:.2f})',
                foreground=C_SUCCESS)
            if self._sel_rid:
                self._cam_canvas.itemconfig(self._sel_rid, state='hidden')
            # Exit draw mode automatically after a successful selection
            if self._draw_mode:
                self._draw_mode = False
                self._crop_mode_btn.config(text='Set Scan Area')
                self._cam_canvas.config(
                    highlightbackground=C_BORDER, highlightthickness=2)
            self._schedule_save()

    # ── Camera: frame display ─────────────────────────────────────────────────

    def _tick_camera(self):
        if IMAGING_OK and self._scanner.camera_open:
            frame = self._scanner.get_frame()
            if frame is not None:
                try:
                    cw = self._cam_canvas.winfo_width()
                    ch = self._cam_canvas.winfo_height()
                    if cw > 10 and ch > 10:
                        fh, fw = frame.shape[:2]
                        # Letterbox / pillarbox scaling: use the smaller ratio
                        # so the frame fits entirely inside the canvas without
                        # cropping or stretching in either dimension.
                        scale  = min(cw / fw, ch / fh)
                        new_w  = int(fw * scale)
                        new_h  = int(fh * scale)
                        disp   = cv2.resize(frame, (new_w, new_h))
                        rgb    = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                        if new_w != cw or new_h != ch:
                            # Frame doesn't fill the full canvas — composite it
                            # onto a solid dark background to fill the bars.
                            bg = Image.new('RGB', (cw, ch), (10, 5, 20))
                            bg.paste(Image.fromarray(rgb),
                                     ((cw - new_w) // 2, (ch - new_h) // 2))
                            canvas_img = bg
                        else:
                            canvas_img = Image.fromarray(rgb)
                        self._photo = ImageTk.PhotoImage(image=canvas_img)
                        if self._cam_img_id is None:
                            self._cam_img_id = self._cam_canvas.create_image(
                                0, 0, anchor='nw', image=self._photo, tags='camimg')
                            self._cam_canvas.itemconfig('notext', state='hidden')
                        else:
                            self._cam_canvas.itemconfig(self._cam_img_id,
                                                        image=self._photo)

                        # Crop overlay — recomputed from normalized coords every
                        # frame so it stays correct after resize / camera reopen.
                        # Default (no user crop) = full frame.
                        cr = self._scanner.crop_region
                        if cr:
                            ox1 = int(cr[0] * cw); oy1 = int(cr[1] * ch)
                            ox2 = int(cr[2] * cw); oy2 = int(cr[3] * ch)
                        else:
                            ox1, oy1, ox2, oy2 = 2, 2, cw - 2, ch - 2

                        if self._crop_rid is None:
                            self._crop_rid = self._cam_canvas.create_rectangle(
                                ox1, oy1, ox2, oy2,
                                outline='#a7f3d0', width=2)
                        else:
                            self._cam_canvas.coords(
                                self._crop_rid, ox1, oy1, ox2, oy2)
                            self._cam_canvas.itemconfig(
                                self._crop_rid, state='normal')
                        self._cam_canvas.tag_raise(self._crop_rid)

                        # Live-drag selection rect (dashed pink)
                        if self._sel_rid:
                            self._cam_canvas.tag_raise(self._sel_rid)
                except Exception:
                    pass
        # Re-schedule ~33 ms from now (≈30 fps).  Using after() keeps this on
        # the main thread — no threading required for the display loop.
        self.after(33, self._tick_camera)

    # ── Connect actions ───────────────────────────────────────────────────────

    def _connect(self):
        port = self._port_v.get().strip()
        try:
            baud = int(self._baud_v.get())
        except ValueError:
            baud = 115200
        self._conn_lbl.config(text='● Connecting…', foreground=C_MUTED)

        def _work():
            ok    = self._controller.connect(port, baud)
            text  = '● Connected'  if ok else '● Disconnected'
            color = C_SUCCESS      if ok else '#9ca3af'
            # after(0, ...) schedules the call on the main thread at the next
            # idle moment — the safe way to update tkinter widgets from here.
            self.after(0, lambda: self._conn_lbl.config(text=text, foreground=color))
            self.after(0, self._sync_cart_label)
            self.after(0, self._redraw_header)

        threading.Thread(target=_work, daemon=True).start()

    def _disconnect(self):
        self._controller.disconnect()
        self._conn_lbl.config(text='● Disconnected', foreground='#9ca3af')
        self._redraw_header()

    def _ping(self):
        if not self._controller.connected:
            self._log('Not connected.'); return
        threading.Thread(target=self._controller.ping, daemon=True).start()

    # ── Port / camera discovery ───────────────────────────────────────────────

    def _scan_ports(self):
        def _work():
            try:
                import serial.tools.list_ports
                ports   = serial.tools.list_ports.comports()
                entries = [f"{p.device} — {p.description}" for p in sorted(ports)]
                devices = [p.device for p in sorted(ports)]
            except Exception:
                entries, devices = [], []
            if not entries:
                entries = ['(no ports found)']
                devices = []

            def _update():
                self._port_cb['values'] = entries
                self._port_devices = devices
                if not self._port_v.get() and devices:
                    self._port_v.set(devices[0])
                    self._port_cb.current(0)

            self.after(0, _update)

        self._port_devices: list = []
        threading.Thread(target=_work, daemon=True).start()

    def _on_port_selected(self, event=None):
        raw = self._port_v.get()
        if ' — ' in raw:
            self._port_v.set(raw.split(' — ')[0].strip())

    def _scan_cameras(self):
        if not IMAGING_OK:
            return
        self._cam_cb['values'] = ['scanning…']
        self._cam_conn_lbl.config(text='Scanning cameras…', foreground=C_MUTED)

        def _work():
            found = []
            for i in range(6):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    found.append(str(i)); cap.release()
            entries = found or ['0']

            def _update():
                self._cam_cb['values'] = entries
                if self._cam_idx_v.get() not in entries:
                    self._cam_idx_v.set(entries[0])
                self._cam_conn_lbl.config(
                    text=f'Found: {", ".join(entries)}' if found else '● Not open',
                    foreground=C_MUTED)

            self.after(0, _update)

        threading.Thread(target=_work, daemon=True).start()

    def _on_cam_selected(self, event=None):
        raw = self._cam_idx_v.get().strip()
        self._cam_idx_v.set(raw.split()[0] if raw else '0')

    # ── Sorter actions ────────────────────────────────────────────────────────

    def _sync_cart_label(self):
        n = self._controller.current_cart
        self._cur_cart_lbl.config(text=f'Position: cart {n}')
        self._cur_cart_v.set(n)

    def _set_current_cart(self):
        self._controller.current_cart = self._cur_cart_v.get()
        self._sync_cart_label()
        self._log(f'Current cart set to {self._controller.current_cart} (no movement).')

    def _apply_settings(self):
        if not self._need_conn(): return
        try:
            self._controller.servo_open_angle  = int(self._servo_open_v.get())
            self._controller.servo_close_angle = int(self._servo_close_v.get())
            self._controller.set_steps(int(self._step_size_v.get()))
            self._controller.set_delay(int(self._delay_v.get()))
        except ValueError:
            self._log('Invalid value in settings.')

    def _open_gate(self):
        if not self._need_conn(): return
        try:
            self._controller.servo_open_angle = int(self._servo_open_v.get())
        except ValueError:
            pass
        self._controller.open_gate()

    def _close_gate(self):
        if not self._need_conn(): return
        try:
            self._controller.servo_close_angle = int(self._servo_close_v.get())
        except ValueError:
            pass
        self._controller.close_gate()

    def _set_servo(self):
        if not self._need_conn(): return
        try:
            self._controller.servo(int(self._servo_v.get()))
        except ValueError:
            self._log('Invalid servo angle.')

    def _cart1(self):
        if self._need_conn():
            self._controller.cart1(on_done=lambda _: self.after(0, self._sync_cart_label))

    def _cart2(self):
        if self._need_conn():
            self._controller.cart2(on_done=lambda _: self.after(0, self._sync_cart_label))

    def _cart3(self):
        if self._need_conn():
            self._controller.cart3(on_done=lambda _: self.after(0, self._sync_cart_label))

    def _fwd_n(self):
        if not self._need_conn(): return
        try:
            self._controller.step_async(int(self._stepn_v.get()))
        except ValueError:
            self._log('Invalid step count.')

    def _back_n(self):
        if not self._need_conn(): return
        try:
            self._controller.step_async(-int(self._stepn_v.get()))
        except ValueError:
            self._log('Invalid step count.')

    def _feed(self):
        if self._need_conn(): self._controller.feed()

    def _stop(self):
        if self._need_conn(): self._controller.stop()

    def _release(self):
        if self._need_conn(): self._controller.release()

    def _full_cycle(self):
        if not self._need_conn(): return
        try:
            n = int(self._cycles_v.get())
        except ValueError:
            n = 1
        self._controller.full_cycle(
            n, on_progress=self._on_status,
            on_done=lambda msg: self.after(0, self._log, msg))

    # ── Scanner actions ───────────────────────────────────────────────────────

    def _open_cam(self):
        try:
            idx = int(self._cam_idx_v.get().split()[0])
        except ValueError:
            idx = 0
        ok = self._scanner.open_camera(idx)
        if ok:
            self._cam_conn_lbl.config(text=f'● Camera {idx} open', foreground=C_SUCCESS)
        else:
            self._cam_conn_lbl.config(text=f'● Failed to open {idx}', foreground='#9ca3af')

    def _close_cam(self):
        self._scanner.close_camera()
        if self._cam_img_id:
            self._cam_canvas.delete(self._cam_img_id)
            self._cam_img_id = None
        self._cam_canvas.itemconfig('notext', state='normal')
        self._cam_conn_lbl.config(text='● Not open', foreground='#9ca3af')

    def _toggle_crop_mode(self):
        self._draw_mode = not self._draw_mode
        if self._draw_mode:
            self._crop_mode_btn.config(text='● Drawing — drag on feed')
            self._cam_canvas.config(
                highlightbackground=C_ACC, highlightthickness=3)
            self._log('Draw mode active — drag on camera feed to set the scan area.')
        else:
            self._crop_mode_btn.config(text='Set Scan Area')
            self._cam_canvas.config(
                highlightbackground=C_BORDER, highlightthickness=2)

    def _pulse_tick(self):
        # If no labels are currently connecting, stop the animation loop —
        # we let the method exit rather than rescheduling itself (self-terminating).
        if not self._pulsing:
            return
        # Advance through four blue shades to create a breathing effect.
        self._pulse_phase = (self._pulse_phase + 1) % 4
        blues = ['#2563eb', '#3b82f6', '#60a5fa', '#3b82f6']
        color = blues[self._pulse_phase]
        # Iterate over a snapshot (list) of _pulsing in case a background
        # thread calls discard() while we are iterating.
        for lbl in list(self._pulsing):
            try:
                lbl.config(foreground=color)
            except Exception:
                pass
        self.after(350, self._pulse_tick)

    def _connect_all(self):
        self._log('Connecting all…')
        # Put all four indicator labels into the pulsing set so the animation
        # loop knows which ones to animate blue.
        self._pulsing = {self._ss_serial_lbl, self._ss_cam_lbl,
                         self._ss_ocr_lbl, self._ss_db_lbl}
        self._pulse_phase = 0
        for lbl, txt in (
            (self._ss_serial_lbl, '◉ Serial — connecting…'),
            (self._ss_cam_lbl,    '◉ Camera — opening…'),
            (self._ss_ocr_lbl,    '◉ OCR — loading…'),
            (self._ss_db_lbl,     '◉ Database — loading…'),
        ):
            lbl.config(text=txt, foreground='#3b82f6')
        self._pulse_tick()   # start the colour-cycling animation

        # OCR model loading already manages its own background thread inside
        # scanner.load_models(), so we just pass a callback for when it finishes.
        def _ocr_done(doctr_ok, paddle_ok):
            parts = (['DocTR'] if doctr_ok else []) + (['Paddle'] if paddle_ok else [])
            txt   = '● ' + (', '.join(parts) + ' ready' if parts else 'none loaded')
            color = C_SUCCESS if parts else '#9ca3af'
            # after(0, ...) — safe tkinter update from a background thread.
            self.after(0, lambda: self._models_lbl.config(text=txt, foreground=color))
            # Remove from _pulsing so the indicator stops animating.
            self._pulsing.discard(self._ss_ocr_lbl)
        self._scanner.load_models(on_done=_ocr_done)

        # Serial, camera, and database are all blocking calls (they each wait
        # for hardware or disk), so we run them sequentially in one background
        # thread to keep the UI responsive.  Each step discards its label from
        # _pulsing when it finishes, which stops the pulse for that item.
        def _run():
            # ── Serial ────────────────────────────────────────────────────────
            port = self._port_v.get().split(' — ')[0].strip()
            try:
                baud = int(self._baud_v.get())
            except ValueError:
                baud = 115200
            self.after(0, lambda: self._conn_lbl.config(
                text='● Connecting…', foreground=C_MUTED))
            ok = self._controller.connect(port, baud)   # blocks until done
            txt   = '● Connected'  if ok else '● Disconnected'
            color = C_SUCCESS      if ok else '#9ca3af'
            # Capture txt/color in default args so the lambda closes over the
            # current values, not variables that may change on the next loop.
            self.after(0, lambda t=txt, c=color: self._conn_lbl.config(text=t, foreground=c))
            self.after(0, self._sync_cart_label)
            self.after(0, self._redraw_header)
            self._pulsing.discard(self._ss_serial_lbl)

            # ── Camera ────────────────────────────────────────────────────────
            try:
                idx = int(self._cam_idx_v.get().split()[0])
            except ValueError:
                idx = 0
            ok_cam = self._scanner.open_camera(idx)     # blocks briefly
            txt   = f'● Camera {idx} open'    if ok_cam else f'● Failed to open {idx}'
            color = C_SUCCESS                   if ok_cam else '#9ca3af'
            self.after(0, lambda t=txt, c=color: self._cam_conn_lbl.config(text=t, foreground=c))
            self._pulsing.discard(self._ss_cam_lbl)

            # ── Database ──────────────────────────────────────────────────────
            self.after(0, lambda: self._db_status_lbl.config(
                text='● Loading…', foreground=C_MUTED))
            self._scanner.load_database(self._db_v.get())  # synchronous JSON read
            if self._scanner.database_loaded:
                self.after(0, lambda: self._db_status_lbl.config(
                    text='● Loaded', foreground=C_SUCCESS))
                self.after(0, self._refresh_criteria_lists)
            else:
                self.after(0, lambda: self._db_status_lbl.config(
                    text='● Failed', foreground='#9ca3af'))
            self._pulsing.discard(self._ss_db_lbl)

        threading.Thread(target=_run, daemon=True).start()

    def _update_conn_indicators(self):
        # This method is a self-repeating poll loop (see the after() at the
        # bottom).  It runs on the main thread every 400 ms and refreshes the
        # four connection status labels by reading the live state of each
        # subsystem.  It does NOT use a background thread.
        #
        # Labels currently in self._pulsing are being animated by _pulse_tick.
        # We skip them here so the two loops don't fight over the foreground
        # colour — _update_conn_indicators will pick them up once they leave
        # _pulsing (i.e. once the connection attempt finishes).

        # Serial
        if self._ss_serial_lbl not in self._pulsing:
            if self._controller.connected:
                port = self._port_v.get().split(' — ')[0].strip()
                detail = f' — {port}' if port else ''
                self._ss_serial_lbl.config(
                    text=f'● Serial — connected{detail}',
                    foreground=C_SUCCESS)
            else:
                self._ss_serial_lbl.config(
                    text='● Serial — not connected', foreground='#9ca3af')

        # Camera
        if self._ss_cam_lbl not in self._pulsing:
            if self._scanner.camera_open:
                idx = self._cam_idx_v.get().split()[0] if self._cam_idx_v.get() else '?'
                res = self._scanner.camera_resolution
                res_str = f' · {res[0]}×{res[1]}' if res else ''
                self._ss_cam_lbl.config(
                    text=f'● Camera — open — cam {idx}{res_str}',
                    foreground=C_SUCCESS)
            else:
                self._ss_cam_lbl.config(
                    text='● Camera — not open', foreground='#9ca3af')

        # OCR
        if self._ss_ocr_lbl not in self._pulsing:
            if self._scanner.models_ready:
                parts = []
                if self._scanner._doctr_loaded:  parts.append('DocTR')
                if self._scanner._paddle_loaded: parts.append('Paddle')
                detail = ' + '.join(parts) if parts else 'ready'
                self._ss_ocr_lbl.config(
                    text=f'● OCR — ready — {detail}', foreground=C_SUCCESS)
            else:
                self._ss_ocr_lbl.config(
                    text='● OCR models — not loaded', foreground='#9ca3af')

        # Database
        if self._ss_db_lbl not in self._pulsing:
            if self._scanner.database_loaded:
                n = self._scanner._card_count
                self._ss_db_lbl.config(
                    text=f'● Database — loaded — {n:,} cards',
                    foreground=C_SUCCESS)
            else:
                self._ss_db_lbl.config(
                    text='● Database — not loaded', foreground='#9ca3af')

        # Re-schedule ourselves — this is the "loop" without a thread.
        self.after(400, self._update_conn_indicators)

    def _clear_crop(self):
        self._scanner.crop_region = None
        if self._sel_rid:
            self._cam_canvas.itemconfig(self._sel_rid, state='hidden')
        # _tick_camera will redraw the crop rect as full-frame automatically
        self._crop_lbl.config(text='Area: full frame', foreground=C_MUTED)
        self._log('Scan area cleared — using full frame.')
        self._schedule_save()

    def _load_models(self):
        self._models_lbl.config(text='● Loading…', foreground=C_MUTED)

        def on_done(doctr_ok: bool, paddle_ok: bool):
            parts = (['DocTR'] if doctr_ok else []) + (['Paddle'] if paddle_ok else [])
            txt   = '● ' + (', '.join(parts) + ' ready' if parts else 'none loaded')
            color = C_SUCCESS if parts else '#9ca3af'
            self.after(0, lambda: self._models_lbl.config(text=txt, foreground=color))

        self._scanner.load_models(on_done=on_done)

    def _scan_now(self):
        if not self._scanner.camera_open:
            self._log('Camera not open.'); return
        if not self._scanner.models_ready:
            self._log('OCR models not loaded.'); return
        try:
            retries = int(self._retries_v.get())
        except ValueError:
            retries = 5

        def on_result(result: str):
            color = C_SUCCESS if 'FAILED' not in result else C_MUTED
            self.after(0, lambda: self._scan_res_lbl.config(text=result, foreground=color))
            self.after(0, lambda: self._result_lbl.config(text=result))

        self._scanner.scan_async(max_attempts=retries, on_result=on_result)

    def _scan_and_identify(self):
        if not self._scanner.camera_open:
            self._log('Camera not open.'); return
        if not self._scanner.models_ready:
            self._log('OCR models not loaded.'); return
        if not self._scanner.database_loaded:
            self._log('No database loaded.'); return
        try:
            retries = int(self._retries_v.get())
        except ValueError:
            retries = 5

        def on_result(result: str):
            if 'FAILED' in result:
                msg   = f'Scan failed → Cart {self._not_found_cart_v.get()} (no match)'
                color = C_MUTED
            else:
                card_data = self._scanner.find_card_data(result)
                cart      = self._scanner.determine_cart(
                    card_data, self._get_sort_criteria(), self._not_found_cart_v.get())
                msg   = f'{result}  →  Cart {cart}'
                color = C_SUCCESS
            self.after(0, lambda: self._scan_res_lbl.config(text=msg, foreground=color))
            self.after(0, lambda: self._result_lbl.config(text=msg))

        self._scanner.scan_async(max_attempts=retries, on_result=on_result)

    # ── Database actions ──────────────────────────────────────────────────────

    def _browse_db(self):
        path = filedialog.askopenfilename(
            title='Select Card Database',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
        if path:
            self._db_v.set(path)

    def _load_db(self):
        self._db_status_lbl.config(text='● Loading…', foreground=C_MUTED)
        self._scanner.load_database(self._db_v.get())

        def _check():
            if self._scanner.database_loaded:
                self._db_status_lbl.config(text='● Loaded', foreground=C_SUCCESS)
                self._refresh_criteria_lists()
            else:
                self._db_status_lbl.config(text='● Failed', foreground='#9ca3af')

        self.after(400, _check)

    def _refresh_criteria_lists(self):
        if not self._scanner.database_loaded:
            self._log('No database loaded.'); return
        available = self._scanner.get_available_values()
        for n in (1, 2, 3):
            self._populate_cart_lb(n, available)
        self._log('Criteria lists refreshed.')

    def _on_field_change(self, cart_num: int):
        if not self._scanner.database_loaded:
            return
        self._populate_cart_lb(cart_num, self._scanner.get_available_values())

    def _populate_cart_lb(self, cart_num: int, available: dict):
        lb     = self._cart_lb[cart_num]
        field  = self._cart_field_v[cart_num].get()
        values = available.get(field, [])

        pending = self._pending_criteria.get(cart_num, {})
        if pending.get('values'):
            # First population after DB load: restore selections saved to disk.
            # Clear pending so we don't re-apply them on the next refresh.
            to_select = set(pending['values'])
            pending['values'] = []
        else:
            # Subsequent refreshes (e.g. field change): preserve whatever the
            # user already has selected.
            to_select = {lb.get(i) for i in lb.curselection()}

        lb.delete(0, 'end')
        for v in values:
            lb.insert('end', v)
            if v in to_select:
                lb.selection_set(lb.size() - 1)

    def _get_sort_criteria(self) -> dict:
        criteria = {}
        for n in (1, 2, 3):
            lb    = self._cart_lb.get(n)
            fv    = self._cart_field_v.get(n)
            field = fv.get() if fv else 'types'
            values= [lb.get(i) for i in lb.curselection()] if lb else []
            criteria[n] = {'field': field, 'values': values}
        return criteria

    # ── Sort & Scan automation ────────────────────────────────────────────────

    def _start_auto(self):
        if self._sort_active:
            self._log('Already running.'); return
        if not self._controller.connected:
            self._log('Sorter not connected.'); return
        if not self._scanner.camera_open:
            self._log('Camera not open.'); return
        if not self._scanner.models_ready:
            self._log('OCR models not loaded.'); return
        self._sort_active = True
        self._auto_status_lbl.config(text='Running…', foreground=C_SUCCESS)
        threading.Thread(target=self._auto_loop, daemon=True).start()

    def _stop_auto(self):
        self._sort_active = False
        self._auto_status_lbl.config(text='Stopping…', foreground=C_MUTED)

    def _set_auto_status(self, msg: str):
        self._on_status(msg)
        self.after(0, lambda: self._auto_status_lbl.config(text=msg))

    def _auto_loop(self):
        # This method runs entirely in a background thread (started by
        # _start_auto).  All hardware calls (feed, step, scan) block here
        # while the UI stays responsive on the main thread.

        # Snapshot the criteria and timing values once at the start so that
        # changing the UI mid-run doesn't affect the current batch.
        criteria       = self._get_sort_criteria()
        not_found_cart = self._not_found_cart_v.get()
        count          = 0
        try:
            settle    = float(self._settle_v.get())
        except ValueError:
            settle    = 1.0
        try:
            drop_wait = float(self._drop_wait_v.get())
        except ValueError:
            drop_wait = 2.0
        try:
            retries   = int(self._auto_retries_v.get())
        except ValueError:
            retries   = 5

        # _stop_auto sets _sort_active = False; the loop exits cleanly at the
        # end of the current card cycle without needing a hard interrupt.
        while self._sort_active:
            try:
                self._set_auto_status('Feeding card…')
                self._controller.feed()
                time.sleep(settle)

                self._set_auto_status('Scanning…')
                result = '--- FAILED ---'
                for _ in range(retries):
                    frame = self._scanner.get_frame()
                    if frame is None:
                        time.sleep(0.3); continue
                    r = self._scanner.scan_frame(frame)
                    if 'FAILED' not in r:
                        result = r; break
                    time.sleep(0.3)

                if 'FAILED' not in result:
                    card_data   = self._scanner.find_card_data(result)
                    target_cart = self._scanner.determine_cart(
                        card_data, criteria, not_found_cart)
                    self._set_auto_status(f'{result}  →  Cart {target_cart}')
                else:
                    target_cart = not_found_cart
                    self._set_auto_status(f'Scan failed → Cart {target_cart}')

                # move_to_cart is non-blocking (it fires off its own thread),
                # so we use a threading.Event to wait here until the hardware
                # signals DONE.  timeout=30 prevents an infinite hang if the
                # Arduino never responds.
                done = threading.Event()
                self._controller.move_to_cart(target_cart,
                                               on_done=lambda _: done.set())
                done.wait(timeout=30)
                # after(0, ...) — update the label safely from this thread.
                self.after(0, self._sync_cart_label)

                self._set_auto_status(f'Dropping into cart {target_cart}…')
                self._controller.open_gate()
                time.sleep(drop_wait)
                self._controller.close_gate()

                done2 = threading.Event()
                self._controller.move_to_cart(1, on_done=lambda _: done2.set())
                done2.wait(timeout=30)
                self.after(0, self._sync_cart_label)

                count += 1
                n = count
                self.after(0, lambda c=n: self._auto_count_lbl.config(
                    text=f'Cards sorted: {c}'))

            except Exception as exc:
                self._set_auto_status(f'Error: {exc}')
                time.sleep(1)

        self._set_auto_status('Stopped.')
        self.after(0, lambda: self._auto_status_lbl.config(foreground=C_MUTED))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self._sort_active = False
        self._save_settings()
        self._controller.disconnect()
        self._scanner.close_camera()
        self.destroy()
