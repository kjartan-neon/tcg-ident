from pyfirmata2 import Arduino, SERVO
import sys

# --- CONFIGURATION ---
port = '/dev/cu.usbserial-1110'
servo_pin = 9  # Ensure your servo is on Pin 9

# --- SETUP ---
print(f"Connecting to Arduino on {port}...")
try:
    board = Arduino(port)
except Exception as e:
    print(f"Error: Could not connect to {port}. Check your USB connection.")
    sys.exit()

# Configure the pin mode
board.digital[servo_pin].mode = SERVO

print("\n--- Servo Interactive Mode ---")
print("Enter an angle between 0 and 180.")
print("Type 'q' or 'exit' to quit.\n")

# --- MAIN LOOP ---
try:
    while True:
        user_input = input("Enter angle (0-180): ").strip()
        
        if user_input.lower() in ['q', 'exit', 'quit']:
            print("Exiting...")
            break
            
        try:
            angle = int(user_input)
            
            # Validation logic
            if 0 <= angle <= 180:
                print(f"Moving to {angle}°")
                board.digital[servo_pin].write(angle)
            else:
                print("⚠️ Out of range! Please keep it between 0 and 180.")
                
        except ValueError:
            print("⚠️ Invalid input. Please enter a whole number.")

except KeyboardInterrupt:
    print("\nForced exit.")

finally:
    # Cleanup: Reset servo to 90 degrees and close connection
    if 'board' in locals():
        board.digital[servo_pin].write(90)
        board.exit()
        print("Connection closed.")