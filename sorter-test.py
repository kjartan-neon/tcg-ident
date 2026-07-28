"""
TCG Sorter — interactive test controller.
Talks to sorter/sorter.ino over serial (115200 baud).

Commands:
  servo <angle>      move servo to angle (0-180)
  f <n>              step forward n steps
  b <n>              step backward n steps
  c2                 move to cart-2 position
  c1                 return to cart-1 position
  stop               halt stepper mid-move
  rel                release / de-energise stepper coils
  delay <us>         set step delay in microseconds (default 2200, try lower)
  set steps <n>      set cart-2 step count
  feed               pulse card feeder relay (200 ms)
  ping               check connection
  cycle <pos> <n>    repeat c1↔c2 n times; pos = current position (1 or 2)
  q                  quit
"""

import serial
import time
import sys

PORT        = '/dev/cu.usbserial-A600b29u'
BAUD        = 115200
CART2_STEPS   = 1640
DEFAULT_DELAY = 2200

# --- Connect ---
print(f"Connecting to {PORT} at {BAUD} baud...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=5)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

time.sleep(2)  # wait for Arduino reset after serial open

# Drain any startup noise and wait for READY
ser.reset_input_buffer()
print("Waiting for board...", end=' ', flush=True)
deadline = time.time() + 5
ready = False
while time.time() < deadline:
    line = ser.readline().decode(errors='replace').strip()
    if line == 'READY':
        ready = True
        break
if ready:
    print("ready.")
else:
    print("no READY received, continuing anyway.")

def send(cmd: str) -> str:
    """Send a command and return the first response line."""
    ser.write((cmd + '\n').encode())
    return ser.readline().decode(errors='replace').strip()

def wait_done():
    """Block until DONE or STOPPED, allow Ctrl-C to send STOP."""
    try:
        while True:
            line = ser.readline().decode(errors='replace').strip()
            if line in ('DONE', 'STOPPED'):
                print(line)
                return
            elif line:
                print(line)
    except KeyboardInterrupt:
        ser.write(b'STOP\n')
        print("\nSTOP sent.")
        ser.readline()  # consume STOPPED

# Push defaults to firmware
send(f"DELAY {DEFAULT_DELAY}")
send(f"STEPS {CART2_STEPS}")

# --- REPL ---
print(f"\n--- Sorter Test ---  cart2={CART2_STEPS} steps  delay={DEFAULT_DELAY}µs")
print("Type 'q' to quit, no args for help.\n")

try:
    while True:
        raw = input("cmd> ").strip()
        if not raw:
            print(__doc__)
            continue

        parts = raw.lower().split()
        cmd   = parts[0]

        if cmd in ('q', 'quit', 'exit'):
            break

        elif cmd == 'servo':
            if len(parts) < 2:
                print("Usage: servo <0-180>")
            else:
                resp = send(f"SERVO {parts[1]}")
                print(resp)

        elif cmd == 'f':
            n = int(parts[1]) if len(parts) > 1 else 100
            print(send(f"STEP {n}"))
            wait_done()

        elif cmd == 'b':
            n = int(parts[1]) if len(parts) > 1 else 100
            print(send(f"STEP -{n}"))
            wait_done()

        elif cmd == 'c2':
            print(f"Moving to cart-2 ({CART2_STEPS} steps)...")
            print(send(f"STEP {CART2_STEPS}"))
            wait_done()

        elif cmd == 'c1':
            print(f"Returning to cart-1 ({CART2_STEPS} steps back)...")
            print(send(f"STEP -{CART2_STEPS}"))
            wait_done()

        elif cmd == 'stop':
            resp = send("STOP")
            print(resp)

        elif cmd == 'rel':
            print(send("REL"))

        elif cmd == 'delay':
            if len(parts) < 2:
                print("Usage: delay <microseconds>  e.g. delay 500")
            else:
                print(send(f"DELAY {parts[1]}"))

        elif cmd == 'set' and len(parts) == 3 and parts[1] == 'steps':
            CART2_STEPS = int(parts[2])
            print(f"Cart-2 steps -> {CART2_STEPS}")

        elif cmd == 'feed':
            print(send("FEED"))

        elif cmd == 'ping':
            print(send("PING"))

        elif cmd == 'cycle':
            if len(parts) < 3:
                print("Usage: cycle <pos> <n>  e.g. cycle 1 10")
                print("  pos = current cart position (1 or 2)")
                print("  n   = number of one-way moves to make")
            else:
                cur_pos = parts[1]
                n_moves = int(parts[2])
                if cur_pos not in ('1', '2'):
                    print("pos must be 1 or 2")
                else:
                    pos = int(cur_pos)
                    print(f"Cycle test: starting at c{pos}, {n_moves} moves, {CART2_STEPS} steps each.")
                    errors = 0
                    for i in range(1, n_moves + 1):
                        target = 2 if pos == 1 else 1
                        direction = '+' if target == 2 else '-'
                        step_cmd = f"STEP {CART2_STEPS}" if direction == '+' else f"STEP -{CART2_STEPS}"
                        print(f"  Move {i}/{n_moves}: c{pos} -> c{target} ...", end=' ', flush=True)
                        resp = send(step_cmd)
                        if resp:
                            print(resp, end=' ', flush=True)
                        wait_done()
                        pos = target
                    print(f"\nCycle test done. Final position: c{pos}. Errors: {errors}")

        else:
            print("Unknown command. Press Enter for help.")

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    send("REL")
    ser.close()
    print("Connection closed.")
