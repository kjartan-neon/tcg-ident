"""
Stepper motor test for TCG card sorter belt.
Two carts on the belt; use this to tune how many steps move from cart 1 to cart 2.

Pins: Arduino D3→coil1, D4→coil2, D5→coil3, D6→coil4
Port: /dev/cu.usbserial-A600b29u
"""

import time
import sys
from pyfirmata2 import Arduino

# --- CONFIGURATION ---
PORT        = '/dev/cu.usbserial-A600b29u'
PINS        = [3, 4, 5, 6]  # stepper coil pins
STEP_DELAY  = 0.003          # 3 ms per step
CART2_STEPS = 3378           # one full cart movement

# Three common sequences — try each if motor doesn't move
SEQUENCES = {
    # One coil at a time (wave drive) — least torque
    'wave': [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    # Two coils at a time (full step) — more torque, most common
    'full': [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ],
    # Alternating (half step) — smoother but needs 8 steps per cycle
    'half': [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ],
}

current_seq_name = 'wave'
STEP_SEQ = SEQUENCES[current_seq_name]

# --- SETUP ---
print(f"Connecting to Arduino on {PORT}...")
try:
    board = Arduino(PORT)
except Exception as e:
    print(f"Error connecting to {PORT}: {e}")
    sys.exit(1)

print("Waiting for board to initialise...", end=' ', flush=True)
time.sleep(2)
print("ready.")

for pin in PINS:
    board.digital[pin].mode = 1  # OUTPUT

step_index = 0

def step_motor(steps: int, delay: float = None):
    global step_index, STEP_DELAY, STEP_SEQ
    if delay is None:
        delay = STEP_DELAY
    direction = 1 if steps >= 0 else -1
    for _ in range(abs(steps)):
        step_index = (step_index + direction) % len(STEP_SEQ)
        for i, pin in enumerate(PINS):
            board.digital[pin].write(STEP_SEQ[step_index][i])
        time.sleep(delay)

def release_motor():
    for pin in PINS:
        board.digital[pin].write(0)

def print_status():
    print(f"  seq={current_seq_name}  delay={STEP_DELAY*1000:.1f}ms  cart2={CART2_STEPS} steps")

# --- MAIN REPL ---
print(f"\n--- Stepper Belt Test ---")
print_status()
print()
print("Commands:")
print("  f <n>          move forward N steps")
print("  b <n>          move backward N steps")
print("  c2             move to cart-2 position")
print("  c1             return to cart-1 position")
print("  set steps <n>      set cart-2 step count")
print("  set delay <ms>     set step delay in ms")
print("  seq <name>         switch step sequence: wave / full / half")
print("  pins <a> <b> <c> <d>  reorder coil pins, e.g. 'pins 3 5 4 6'")
print("  rel                release / de-energise motor")
print("  status             show current settings")
print("  q                  quit\n")

try:
    while True:
        raw = input("cmd> ").strip().lower()
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0]

        if cmd in ('q', 'quit', 'exit'):
            break

        elif cmd == 'f':
            n = int(parts[1]) if len(parts) > 1 else 10
            print(f"Forward {n} steps...")
            step_motor(n)
            print("Done.")

        elif cmd == 'b':
            n = int(parts[1]) if len(parts) > 1 else 10
            print(f"Backward {n} steps...")
            step_motor(-n)
            print("Done.")

        elif cmd == 'c2':
            print(f"Moving to cart-2 ({CART2_STEPS} steps)...")
            step_motor(CART2_STEPS)
            print("Done.")

        elif cmd == 'c1':
            print(f"Returning to cart-1 ({CART2_STEPS} steps back)...")
            step_motor(-CART2_STEPS)
            print("Done.")

        elif cmd == 'set' and len(parts) == 3:
            if parts[1] == 'steps':
                CART2_STEPS = int(parts[2])
                print(f"Cart-2 steps → {CART2_STEPS}")
            elif parts[1] == 'delay':
                STEP_DELAY = float(parts[2]) / 1000.0
                print(f"Step delay → {STEP_DELAY*1000:.1f} ms")
            else:
                print("Usage: set steps <n>  |  set delay <ms>")

        elif cmd == 'seq':
            name = parts[1] if len(parts) > 1 else ''
            if name in SEQUENCES:
                current_seq_name = name
                STEP_SEQ = SEQUENCES[name]
                step_index = 0
                print(f"Sequence → {name}")
            else:
                print(f"Unknown sequence. Choose: {', '.join(SEQUENCES)}")

        elif cmd == 'rel':
            release_motor()
            print("Motor released.")

        elif cmd == 'pins':
            if len(parts) == 5:
                try:
                    new_pins = [int(p) for p in parts[1:]]
                    PINS = new_pins
                    step_index = 0
                    for pin in PINS:
                        board.digital[pin].mode = 1
                    print(f"Pins → {PINS}")
                except ValueError:
                    print("Usage: pins <a> <b> <c> <d>  e.g. pins 3 5 4 6")
            else:
                # Print orderings worth trying
                print("Current pins:", PINS)
                print("Orderings to try:")
                print("  pins 3 4 5 6   (default)")
                print("  pins 3 5 4 6   (swap middle)")
                print("  pins 6 5 4 3   (reversed)")
                print("  pins 4 3 6 5   (swap pairs)")

        elif cmd == 'status':
            print_status()

        else:
            print("Unknown command. Type 'q' to quit.")

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    release_motor()
    board.exit()
    print("Motor released. Connection closed.")
