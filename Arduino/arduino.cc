#include <AccelStepper.h>

#define DRIVER_MODE AccelStepper::DRIVER

// Stepper driver pins
#define X_STEP_PIN 2 //aliexpress motor
#define X_DIR_PIN 3
#define Y_STEP_PIN 4
#define Y_DIR_PIN 5

// Stepper driver enable pins
#define X_ENABLE_PIN 7
#define Y_ENABLE_PIN 8

// Limit switch pins (normally open, grounded when triggered)
#define X_LIMIT_PIN 6
#define Y_LIMIT_PIN 9

#define LED_1 13
#define LED_2 12

// Calibrated for motor + microstepping (e.g., 1.8° motor w/ 1/16 microstepping)
const float stepsPerDegreex = 2.222;
const float stepsPerDegreey = 8.888;

AccelStepper stepperX(DRIVER_MODE, X_STEP_PIN, X_DIR_PIN);
AccelStepper stepperY(DRIVER_MODE, Y_STEP_PIN, Y_DIR_PIN);

void setup() {
  Serial.begin(9600);

  pinMode(X_LIMIT_PIN, INPUT_PULLUP);
  pinMode(Y_LIMIT_PIN, INPUT_PULLUP);
  pinMode(X_ENABLE_PIN, OUTPUT);
  pinMode(Y_ENABLE_PIN, OUTPUT);

  enableSteppers();

  stepperX.setMaxSpeed(200);
  stepperX.setAcceleration(50);
  stepperY.setMaxSpeed(4000);
  stepperY.setAcceleration(200);

  // homeBothAxes();
  // Serial.println("HOME_DONE");
  }

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.startsWith("MOVE_TO")) {
      int space1 = input.indexOf(' ');
      int space2 = input.indexOf(' ', space1 + 1);
      long x_deg = input.substring(space1 + 1, space2).toInt();
      long y_deg = input.substring(space2 + 1).toInt();

      enableSteppers();
      moveToAngles(x_deg, y_deg);
      Serial.println("MOVE_DONE");

    } else if (input == "GET_POS") {
      long x_deg = stepperX.currentPosition() / stepsPerDegreex;
      long y_deg = stepperY.currentPosition() / stepsPerDegreey;
      Serial.print("POS ");
      Serial.print(x_deg);
      Serial.print(" ");
      Serial.println(y_deg);

    } else if (input == "HOME") {
      enableSteppers();
      homeBothAxes();
      Serial.println("HOME_DONE");

    } else if (input == "DISABLE") {
      disableSteppers();
      Serial.println("STEPPERS_DISABLED");

    } else if (input == "LED1ON") {
      LED1ON();
      Serial.println("LED_1_ON_DONE");

    } else if (input == "LED1OFF"){
      LED1OFF();
      Serial.println("LED_1_OFF_DONE");

    } else {
      Serial.println("UNKNOWN_CMD");
    }
  }
}

void moveToAngles(long x_deg, long y_deg) {
  long x_steps = x_deg * stepsPerDegreex;
  long y_steps = y_deg * stepsPerDegreey;

  stepperX.moveTo(x_steps);
  stepperY.moveTo(y_steps);

  while (stepperX.isRunning() || stepperY.isRunning()) {
    stepperX.run();
    stepperY.run();
  }
}

void homeBothAxes() {
  homeAxis(stepperX, X_LIMIT_PIN);
  homeAxis(stepperY, Y_LIMIT_PIN);
}

void homeAxis(AccelStepper& stepper, int limitPin) {
  stepper.setMaxSpeed(300);
  stepper.setAcceleration(100);
  stepper.moveTo(-10000);  // Move until limit triggered

  while (digitalRead(limitPin) == HIGH) {
    stepper.run();
  }

  stepper.setCurrentPosition(0);
  stepper.moveTo(stepsPerDegreex * 5);  // Back off 5 degrees
  while (stepper.isRunning()) {
    stepper.run();
  }

  stepper.setCurrentPosition(0);  // Set as zero
}

void enableSteppers() {
  digitalWrite(X_ENABLE_PIN, LOW); // LOW = enabled
  digitalWrite(Y_ENABLE_PIN, LOW);
}

void disableSteppers() {
  digitalWrite(X_ENABLE_PIN, HIGH); // HIGH = disabled
  digitalWrite(Y_ENABLE_PIN, HIGH);
}
void LED1ON(){
digitalWrite(LED_1, HIGH);
} 

void LED1OFF(){
digitalWrite(LED_1, LOW);  
}


