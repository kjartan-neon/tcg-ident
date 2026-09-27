#!/usr/bin/env python3
"""
TCG Card Sorter — entry point.

Run with:
    source env/bin/activate
    python app/main.py
"""

import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(APP_DIR, 'source')
# Make `source/` importable as a plain package without an __init__.py.
# sys.path is the list Python searches when you write `import something`.
sys.path.insert(0, SRC_DIR)

from gui import App


def main():
    app = App()
    # mainloop() hands control to tkinter's event loop.
    # It blocks here until the window is closed.
    app.mainloop()


if __name__ == '__main__':
    main()
