/*
 * TCG Card Sorter — combined stepper + servo controller
 *
 * Stepper: pins 3-6 (ULN2003, wave sequence)
 * Servo:   pin 9
 *
 * Serial protocol (115200 baud, newline-terminated commands):
 *   PING          → PONG
 *   SERVO <0-180> → OK  (attach if needed, move, auto-detach after ~1 s)
 *   SREL          → OK  (detach servo so a single light card holds no torque)
 *   STEP <n>      → MOVING … DONE  (negative n = reverse)
 *   STOP          → STOPPED
 *   REL           → OK  (de-energise stepper coils)
 *   DELAY <us>    → OK  (set microseconds between steps, default 1000)
 *   FEED          → OK  (pulse card feeder relay on pin 8 for 200 ms)
 */

#include <Servo.h>

// --- Stepper ---
const int STEP_PINS[4] = {3, 4, 5, 6};
const int WAVE[4][4] = {
  {1, 0, 0, 0},
  {0, 1, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 0, 1},
};

int          stepIndex      = 0;
long         stepsRemaining = 0;
int          stepDir        = 1;
unsigned long lastStepTime  = 0;
unsigned int stepDelayUs    = 1000;  // tune down from 3000 if motor stalls

// --- Servo ---
// Only a single light card is held, so no holding torque is needed.
// After every move the servo is detached (PWM off) to stop buzzing,
// save power and avoid heat. SERVO re-attaches automatically.
Servo sorterServo;
const int SERVO_PIN = 9;
// How long after a SERVO move to keep PWM before auto-detaching.
const unsigned long SERVO_HOLD_MS = 800;
unsigned long servoMoveTime = 0;
bool          servoHolding  = false;

void servoDetach() {
  if (sorterServo.attached()) sorterServo.detach();
  servoHolding = false;
}

void servoMove(int angle) {
  angle = constrain(angle, 0, 180);
  if (!sorterServo.attached()) sorterServo.attach(SERVO_PIN);
  sorterServo.write(angle);
  servoMoveTime = millis();
  servoHolding  = true;
}

// --- Card feeder ---
const int    CARD_PIN       = 8;
const int    CARD_RUN_MS    = 200;
bool         cardActive     = false;
unsigned long cardStartTime = 0;

// ---------------------------------------------------------------

void releaseCoils() {
  for (int i = 0; i < 4; i++) digitalWrite(STEP_PINS[i], LOW);
}

void applyStep(int dir) {
  stepIndex = (stepIndex + dir + 4) % 4;
  for (int i = 0; i < 4; i++) digitalWrite(STEP_PINS[i], WAVE[stepIndex][i]);
}

void handleCommand(String cmd) {
  cmd.trim();

  if (cmd == "PING") {
    Serial.println("PONG");

  } else if (cmd.startsWith("SERVO ")) {
    int angle = cmd.substring(6).toInt();
    servoMove(angle);
    Serial.println("OK");

  } else if (cmd == "SREL") {
    servoDetach();
    Serial.println("OK");

  } else if (cmd.startsWith("STEP ")) {
    long n = cmd.substring(5).toInt();
    stepDir        = (n >= 0) ? 1 : -1;
    stepsRemaining = abs(n);
    lastStepTime   = micros();
    Serial.println("MOVING");

  } else if (cmd == "STOP") {
    stepsRemaining = 0;
    releaseCoils();
    Serial.println("STOPPED");

  } else if (cmd == "REL") {
    stepsRemaining = 0;
    releaseCoils();
    Serial.println("OK");

  } else if (cmd == "FEED") {
    digitalWrite(CARD_PIN, HIGH);
    cardActive    = true;
    cardStartTime = millis();
    Serial.println("OK");

  } else if (cmd.startsWith("DELAY ")) {
    stepDelayUs = (unsigned int) constrain(cmd.substring(6).toInt(), 100, 100000);
    Serial.println("OK");

  } else {
    Serial.println("ERR unknown command");
  }
}

// ---------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    pinMode(STEP_PINS[i], OUTPUT);
    digitalWrite(STEP_PINS[i], LOW);
  }
  pinMode(CARD_PIN, OUTPUT);
  digitalWrite(CARD_PIN, LOW);
  sorterServo.attach(SERVO_PIN);
  sorterServo.write(90);
  servoMoveTime = millis();
  servoHolding  = true;
  Serial.println("READY");
}

void loop() {
  // Auto-release the servo once it has had time to reach position.
  // The gate only holds a single light card, so it stays put unpowered.
  if (servoHolding && millis() - servoMoveTime >= SERVO_HOLD_MS) {
    servoDetach();
  }
  // Non-blocking card feeder
  if (cardActive && millis() - cardStartTime >= CARD_RUN_MS) {
    digitalWrite(CARD_PIN, LOW);
    cardActive = false;
  }

  // Non-blocking stepper
  if (stepsRemaining > 0) {
    unsigned long now = micros();
    if (now - lastStepTime >= stepDelayUs) {
      applyStep(stepDir);
      lastStepTime = now;
      stepsRemaining--;
      if (stepsRemaining == 0) {
        releaseCoils();
        Serial.println("DONE");
      }
    }
  }

  // Serial command reader
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    handleCommand(cmd);
  }
}
