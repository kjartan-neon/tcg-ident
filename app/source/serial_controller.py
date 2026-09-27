"""
Serial communication layer for the TCG sorter Arduino.

Talks to sorter/sorter.ino over serial (115200 baud) using the same
command protocol as sorter-test.py (SERVO, STEP, FEED, PING, etc.).
All blocking operations (step + wait-for-DONE) run in background threads;
results are delivered via the on_status callback.
"""

import serial
import threading
import time
from typing import Callable, Optional

PORT_DEFAULT        = '/dev/cu.usbserial-130'
BAUD_DEFAULT        = 115200
CART_STEP_DEFAULT   = 870       # steps between adjacent carts
STEP_DELAY_DEFAULT  = 3600      # microseconds between stepper pulses

# Legacy fullcycle constants (used by the Sorter tab's "Run" button)
SERVO_FEED_ANGLE  = 155
SERVO_DROP_ANGLE  = 50
FULL_CYCLE_STEPS  = 1740
CARD_DROP_WAIT    = 3.0

# Default gate positions — overridable in the Sorter tab
SERVO_OPEN_DEFAULT  = 50
SERVO_CLOSE_DEFAULT = 179


class SorterController:
    def __init__(self, on_status: Optional[Callable[[str], None]] = None):
        self._ser: Optional[serial.Serial] = None
        # A Lock prevents two threads from writing to the serial port at the
        # same time, which would corrupt the data stream.
        self._lock = threading.Lock()
        self.step_delay_us    = STEP_DELAY_DEFAULT
        self.cart_step_size   = CART_STEP_DEFAULT   # steps between adjacent carts
        self.servo_open_angle  = SERVO_OPEN_DEFAULT
        self.servo_close_angle = SERVO_CLOSE_DEFAULT
        # We track cart position in software because the Arduino has no
        # position sensor — we calculate it from the steps we send.
        self.current_cart      = 1
        self.on_status = on_status
        self._busy = False  # True while a move is in progress

    # ── Convenience alias ────────────────────────────────────────────────────
    @property
    def cart2_steps(self) -> int:
        return self.cart_step_size

    @cart2_steps.setter
    def cart2_steps(self, v: int):
        self.cart_step_size = v

    # ── State ────────────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @property
    def busy(self) -> bool:
        return self._busy

    def _emit(self, msg: str):
        """Send a status message to the GUI log via the callback."""
        if self.on_status:
            self.on_status(msg)

    # ── Connection ───────────────────────────────────────────────────────────

    def connect(self, port: str = PORT_DEFAULT, baud: int = BAUD_DEFAULT) -> bool:
        try:
            ser = serial.Serial(port, baud, timeout=5)
            # The Arduino resets when a serial connection opens. We wait 2 s
            # for the bootloader to finish before sending any commands.
            time.sleep(2)
            ser.reset_input_buffer()
            # Wait up to 5 s for the Arduino to print "READY"
            ready = False
            deadline = time.time() + 5
            while time.time() < deadline:
                line = ser.readline().decode(errors='replace').strip()
                if line == 'READY':
                    ready = True
                    break
            self._ser = ser
            # Push our current settings to the Arduino right after connecting
            self._raw(f"DELAY {self.step_delay_us}")
            self._raw(f"STEPS {self.cart_step_size}")
            self._emit(
                f"Connected to {port}" +
                (" (READY)" if ready else " (timeout — continuing anyway)")
            )
            return True
        except Exception as e:
            self._emit(f"Connection error: {e}")
            return False

    def disconnect(self):
        if self._ser:
            try:
                # Release the stepper (cuts hold current) before closing
                self._raw("REL")
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            self._emit("Disconnected.")

    # ── Low-level serial ─────────────────────────────────────────────────────

    def _raw(self, cmd: str) -> str:
        """Send one command and read the immediate acknowledgement line."""
        with self._lock:
            if not self._ser:
                return ""
            self._ser.write((cmd + '\n').encode())
            return self._ser.readline().decode(errors='replace').strip()

    def _wait_done(self) -> str:
        """Block until the Arduino sends DONE or STOPPED.

        Long-running commands (STEP) send an immediate ACK then later send
        DONE when the move finishes. This method reads lines until it sees
        one of those terminal responses.
        """
        while True:
            with self._lock:
                if not self._ser:
                    return "NOT_CONNECTED"
                line = self._ser.readline().decode(errors='replace').strip()
            if line in ('DONE', 'STOPPED'):
                self._emit(line)
                return line
            elif line:
                # Intermediate status messages (e.g. progress counts)
                self._emit(line)

    # ── Basic commands ───────────────────────────────────────────────────────

    def ping(self) -> str:
        resp = self._raw("PING")
        self._emit(f"PING: {resp}")
        return resp

    def servo(self, angle: int) -> str:
        resp = self._raw(f"SERVO {angle}")
        self._emit(f"SERVO {angle}: {resp}")
        return resp

    def open_gate(self) -> str:
        """Move servo to the open (card-drop) angle."""
        return self.servo(self.servo_open_angle)

    def close_gate(self) -> str:
        """Move servo to the closed (holding) angle."""
        return self.servo(self.servo_close_angle)

    def feed(self) -> str:
        resp = self._raw("FEED")
        self._emit(f"FEED: {resp}")
        return resp

    def stop(self) -> str:
        resp = self._raw("STOP")
        self._emit(f"STOP: {resp}")
        return resp

    def release(self) -> str:
        resp = self._raw("REL")
        self._emit(f"REL: {resp}")
        return resp

    def set_delay(self, us: int):
        self.step_delay_us = us
        resp = self._raw(f"DELAY {us}")
        self._emit(f"DELAY {us}µs: {resp}")

    def set_steps(self, n: int):
        self.cart_step_size = n
        resp = self._raw(f"STEPS {n}")
        self._emit(f"Cart step size → {n}: {resp}")

    # ── Step / cart movement ─────────────────────────────────────────────────

    def step_async(self, n: int, on_done: Optional[Callable[[str], None]] = None):
        """Send STEP n and wait for DONE in a background thread.

        The STEP command takes a long time (hundreds of milliseconds to
        several seconds). Running _wait_done in a background thread means
        the GUI stays responsive while the cart is moving.
        """
        resp = self._raw(f"STEP {n}")
        self._emit(f"STEP {n}: {resp}")
        self._busy = True

        def _wait():
            result = self._wait_done()
            self._busy = False
            if on_done:
                on_done(result)

        threading.Thread(target=_wait, daemon=True).start()

    def move_to_cart(self, target: int, on_done: Optional[Callable[[str], None]] = None):
        """Move from current_cart to target (1/2/3) using cart_step_size per step."""
        if target == self.current_cart:
            self._emit(f"Already at cart {target}.")
            if on_done:
                on_done("DONE")
            return
        steps = (target - self.current_cart) * self.cart_step_size
        self._emit(
            f"Moving cart {self.current_cart} → cart {target} ({steps:+d} steps)…"
        )
        prev = self.current_cart
        # Optimistically update the tracked position before the move completes.
        # If the Arduino reports STOPPED (e.g. user pressed Stop), we revert.
        self.current_cart = target

        def _wrapped(result: str):
            if result == "STOPPED":
                # Move was interrupted — we no longer know where the cart is
                self.current_cart = prev
                self._emit(f"Move stopped — cart position uncertain (was {prev})")
            if on_done:
                on_done(result)

        self.step_async(steps, on_done=_wrapped)

    def cart1(self, on_done: Optional[Callable[[str], None]] = None):
        self.move_to_cart(1, on_done)

    def cart2(self, on_done: Optional[Callable[[str], None]] = None):
        self.move_to_cart(2, on_done)

    def cart3(self, on_done: Optional[Callable[[str], None]] = None):
        self.move_to_cart(3, on_done)

    # ── Legacy full-cycle ────────────────────────────────────────────────────

    def full_cycle(
        self,
        n: int = 1,
        on_progress: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str], None]] = None,
    ):
        """Run n full sort cycles (mirrors sorter-test.py 'fullcycle' command)."""
        def _run():
            self._busy = True
            try:
                for i in range(1, n + 1):
                    if on_progress:
                        on_progress(f"Cycle {i}/{n}: servo feed…")
                    self._raw(f"SERVO {SERVO_FEED_ANGLE}")
                    self._raw("FEED")
                    if on_progress:
                        on_progress(f"Cycle {i}/{n}: forward {FULL_CYCLE_STEPS}…")
                    self._raw(f"STEP {FULL_CYCLE_STEPS}")
                    self._wait_done()
                    self._raw(f"SERVO {SERVO_DROP_ANGLE}")
                    time.sleep(CARD_DROP_WAIT)

                    if on_progress:
                        on_progress(f"Cycle {i}/{n}: backward…")
                    self._raw(f"SERVO {SERVO_FEED_ANGLE}")
                    self._raw("FEED")
                    self._raw(f"STEP -{FULL_CYCLE_STEPS}")
                    self._wait_done()
                    self._raw(f"SERVO {SERVO_DROP_ANGLE}")
                    time.sleep(CARD_DROP_WAIT)

                if on_done:
                    on_done(f"Full cycle complete: {n} cycle(s).")
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()
