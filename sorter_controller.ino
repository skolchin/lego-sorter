#include <AccelStepper.h>
#include <yasm.h>
#include <Servo.h>

#define DEBUG_MODE           0

// Relay pins
#define CONVEYOR_PIN         4 
#define TABLE_PIN            3

// Servo pins
#define TUBE_GATE_SERVO_PIN  2
#define TURN_SERVO_PIN       10
#define GATE_SERVO_PIN       9

// Stepper calibration pins 
#define TT_NULL_POSITION     8

// Stepper pins
#define PUL_PIN              12
#define DIR_PIN              11

// Debug LEDs
#define CALIBRATION_LED_PIN  A1 // green
#define DROP_LED_PIN         A0 // blue
#define DROP_UNSRT_LED_PIN   A2 // yellow
#define DROP_UNRCG_LED_PIN   A3 // red

// Stepper parameters
#define CALIBRATION_STEPS    10000
#define CALIBRATION_SPEED    200.0
#define WORKING_SPEED        1000.0
#define ACCELERATION         200.0

// Delays and timings
#define CONVEYOR_START_DELAY 5000
#define CONVEYOR_WORK_CYCLE  12000
#define CONVEYOR_PAUSE       10000
#define SERVO_ACTION_DELAY   500
#define DROP_ACTION_DELAY    800
#define STATUS_SEND_TIMING   2500

// define bucket position with two arrays:
//  - bucket table rotation angle in steps 
//  - angle position of recognition camera 
#define BUCKET_COUNT          18
#define INNER_BUCKET_POS      110
#define OUTER_BUCKET_POS      135
#define WAIT_BUCKET_POS       77

#define START_ROTATION_OFFSET 0

// gate angles
#define GATE_OPENED           60
#define GATE_CLOSED           130

const int bucket_table_steps[BUCKET_COUNT] = {0,267,
                                              534,
                                              1068,1335,
                                              1602,
                                              2136,2403,
                                              2670,
                                              3204,3471,
                                              3738,
                                              4272,4539,
                                              4806,
                                              5340,5607,
                                              5874
                                             };
                                             
const int rc_position[BUCKET_COUNT] =        {OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS,
                                              OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS,	
                                              OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS,	
                                              OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS,	
                                              OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS,	
                                              OUTER_BUCKET_POS,	INNER_BUCKET_POS,
                                              OUTER_BUCKET_POS
                                             };
const int unsorted_rc_position = 25;
const int unrecognized_rc_position = 160;

#define START_BUCKET_OFFSET  65; // 'A' symbol
byte currentBucket = 0;

// global states
int conveyorState = LOW;
int vibroTableState = LOW;

// stepper with dedicated driver
AccelStepper stepper(2, PUL_PIN, DIR_PIN);

// servos
Servo turnServo;
Servo gateServo;

enum GateState {OPENED, CLOSED};

// first recognition done flag 
bool firstRecognitionDone = false;

// timing counters
unsigned long timing;
unsigned long stateStartTime;
unsigned long tempTime;

// State machine variables
YASM fsm; 

// calibration variables
bool calibrationBucketAwait = false;
byte calibrationBucket = 0;

void Start() {
  handleTransition();

  conveyorState = LOW;
  vibroTableState = LOW;

  turnServo.write(WAIT_BUCKET_POS);

  fsm.next(Calibration);
}

void Calibration() {
  handleTransition();
  
  if ( fsm.isFirstRun() ) {
    stepper.setMaxSpeed(CALIBRATION_SPEED);
    stepper.setAcceleration(ACCELERATION);
    stepper.setSpeed(CALIBRATION_SPEED);
    stepper.moveTo(CALIBRATION_STEPS);
  }

  if(stepper.distanceToGo() == 0) {
    stepper.setCurrentPosition(0);
    stepper.moveTo(CALIBRATION_STEPS);
  }

  int turn_table_state = digitalRead(TT_NULL_POSITION);
  
  if ( turn_table_state == HIGH ) { 
    stepper.setCurrentPosition(0);
    stepper.moveTo(0);
    stepper.setMaxSpeed(WORKING_SPEED);
    stepper.setAcceleration(ACCELERATION);
    stepper.setSpeed(WORKING_SPEED);
    
    digitalWrite(CALIBRATION_LED_PIN, HIGH);
    if ( DEBUG_MODE == 1 ) fsm.next(IterateBuckets);
    else fsm.next(Wait);
  }

  stepper.run();
}

void IterateBuckets() {
  handleTransition();

  if ( fsm.isFirstRun() ) {
    digitalWrite(CALIBRATION_LED_PIN, LOW);
    moveToBucket(calibrationBucket);
  }
  
  if (stepper.distanceToGo() == 0) {
    if (!calibrationBucketAwait) {
      tempTime = millis();
      calibrationBucketAwait = true;
      digitalWrite(DROP_LED_PIN, HIGH);

      Serial.print("We on bucket ");
      Serial.println(calibrationBucket);

      if ( calibrationBucket + 1 == BUCKET_COUNT )  {
        turnServo.write(WAIT_BUCKET_POS);
        digitalWrite(CALIBRATION_LED_PIN, HIGH);
        fsm.next(Wait);
      }
    } else {
      if (millis() - tempTime > 2000) {   
        calibrationBucketAwait = false; 
        calibrationBucket++;
        digitalWrite(DROP_LED_PIN, LOW);
        moveToBucket(calibrationBucket);
      }
    }
  } else { 
    stepper.run();
  }
}

void Wait() {
  handleTransition();

  // everything stopped
  // waiting for start command from host computer
  //  * Move (M)
  if (Serial.available() > 0) {
    String data = getSerialData();

    if (data[0] == 'M') { 
      confirmationMessage(data);
      fsm.next(Move);
    } else {
      incorrectInputMessage(data);
    }
  } else { 
    conveyorState = LOW;
    vibroTableState = LOW;
  }
}

void Move() { 
  handleTransition();

  digitalWrite(CALIBRATION_LED_PIN, LOW);

  // conveyor and vibrotable enabled
  // waiting for commands from host computer
  //  * recognition (R)
  //  * wait        (W)
  if (Serial.available() > 0) {
    String data = getSerialData();

    if ( data[0] == 'W') {
      confirmationMessage(data);

      fsm.next(Wait);
    }
    else 
    if ( data[0] == 'R' ) {
      confirmationMessage(data);

      fsm.next(Recognize);
    }
    else {
      incorrectInputMessage(data);
    }    
  } else {
    vibroTableState = HIGH;
    
    if ( getTimeOnState() < CONVEYOR_START_DELAY && firstRecognitionDone ) {
      conveyorState = LOW;
    } else { 
      if ( conveyorState == LOW && millis() > tempTime ) { 
        tempTime = millis() + CONVEYOR_WORK_CYCLE;
        conveyorState = HIGH;
      }
      
      if ( conveyorState == HIGH && millis() > tempTime ) { 
        tempTime = millis() + CONVEYOR_PAUSE;
        conveyorState = LOW;
      }
    }
  }
}

void Recognize() {
  handleTransition();

  // conveyor and vibrotable disabled
  // waiting for recognition on host computer done
  //  * select bucket (S)(bucket)
  conveyorState = LOW;
  vibroTableState = LOW;

  if (Serial.available() > 0) {
    String data = getSerialData();

    if (data[0] == 'S') {
      if (data.length()<2) {
        errorMessage("Incorrect message length:"+data);
      } else {
        byte rescBucket = currentBucket;
        currentBucket = byte(data[1]) - START_BUCKET_OFFSET;
        Serial.println(currentBucket);
        
        if ( currentBucket >= 0 && currentBucket < BUCKET_COUNT + 2) {
          confirmationMessage(data);
          fsm.next(SelectBucket);
        } else { 
          errorMessage("Incorrect bucket number");
          currentBucket = rescBucket;
        }
      } 
    } else { 
      incorrectInputMessage(data);
    }
  }

  firstRecognitionDone = true;
}

void SelectBucket() {
  handleTransition();

  digitalWrite(CALIBRATION_LED_PIN, LOW);

  if ( fsm.isFirstRun() ) {
    moveToBucket(currentBucket);
  } else { 
    if(stepper.distanceToGo() == 0 && getTimeOnState() > SERVO_ACTION_DELAY) {
      if (currentBucket < BUCKET_COUNT) digitalWrite(DROP_LED_PIN, HIGH);
      if (currentBucket == BUCKET_COUNT) digitalWrite(DROP_UNSRT_LED_PIN, HIGH);
      if (currentBucket == BUCKET_COUNT+1) digitalWrite(DROP_UNRCG_LED_PIN, HIGH);

      fsm.next(Drop);
    }
  }

  stepper.run();
}

void Drop() {
  handleTransition();

  if ( fsm.isFirstRun() ) 
    setGateState(OPENED); 
  else 
    if (  getTimeOnState() > DROP_ACTION_DELAY )
      fsm.next(CloseGate);
}

void CloseGate() {
  handleTransition();

  if ( fsm.isFirstRun() ) 
    setGateState(CLOSED);
  else
    if ( getTimeOnState() > SERVO_ACTION_DELAY ) {
      digitalWrite(DROP_LED_PIN, LOW);
      digitalWrite(DROP_UNSRT_LED_PIN, LOW);
      digitalWrite(DROP_UNRCG_LED_PIN, LOW);

      turnServo.write(WAIT_BUCKET_POS);

      fsm.next(Move);
    }
}


// Standard arduino functions
void setup() {
  // setup pins mode
  pinMode(CONVEYOR_PIN, OUTPUT); 
  pinMode(TABLE_PIN, OUTPUT); 
 
  pinMode(TURN_SERVO_PIN, OUTPUT); 
  pinMode(GATE_SERVO_PIN, OUTPUT); 

  pinMode(TT_NULL_POSITION, INPUT); 

  pinMode(PUL_PIN, OUTPUT); 
  pinMode(DIR_PIN, OUTPUT); 

  pinMode(CALIBRATION_LED_PIN, OUTPUT); 

  // attache servos
  turnServo.attach(TURN_SERVO_PIN);
  gateServo.attach(GATE_SERVO_PIN);

  // setup serial communiaction 
  Serial.begin(9600);

  setGateState(CLOSED);
  digitalWrite(CONVEYOR_PIN, LOW); 
  digitalWrite(TABLE_PIN, LOW); 
  
  // setup state machine
  fsm.next(Start);

  // start timing
  timing = millis();
}

void loop() {
  fsm.run();

  digitalWrite(CONVEYOR_PIN, conveyorState); 
  digitalWrite(TABLE_PIN, vibroTableState); 

  if ( millis() - timing > STATUS_SEND_TIMING ) {
    statusMessage();
    timing = millis();
  } 
}

// execution 
void moveToBucket(byte bucket) {
  // recognized, sorted
    if (bucket < BUCKET_COUNT) {      
      stepper.moveTo(START_ROTATION_OFFSET + bucket_table_steps[bucket]);
      turnServo.write(rc_position[bucket]);
    } 
    // recognized, unsorted
    if (bucket == BUCKET_COUNT) {
      turnServo.write(unsorted_rc_position);
    } 
    // unrecognized
    if (bucket == BUCKET_COUNT+1) {
      turnServo.write(unrecognized_rc_position);
    } 
}

// getters&setters
void handleTransition() {
  if ( fsm.isFirstRun() ) { 
    stateStartTime = millis();

    if (Serial.available() > 0) {
      String data = getSerialData();
      incorrectInputMessage(data);
    }

    statusMessage();
  }
}

unsigned long getTimeOnState() {
  return millis() - stateStartTime;
}

void setGateState(GateState state) {
  if (state == OPENED)
    gateServo.write(GATE_OPENED);
  else 
    gateServo.write(GATE_CLOSED);
}

String getState() {
  if (fsm.isInState(Start)) return "Start";
  if (fsm.isInState(Calibration)) return "Calibration";
  if (fsm.isInState(IterateBuckets)) return "IterateBuckets";
  if (fsm.isInState(Wait)) return "Wait";
  if (fsm.isInState(Move)) return "Move";
  if (fsm.isInState(Recognize)) return "Recognize";
  if (fsm.isInState(SelectBucket)) return "SelectBucket";
  if (fsm.isInState(Drop)) return "Drop";
  if (fsm.isInState(CloseGate)) return "CloseGate";

  return "Unknown";
}

String getSerialData() {
  String data = Serial.readString();
  debugMessage("Serial data recieved: " + data);

  return data;
}

// serial messaging
void serialMessage(String prefix, String msg) {
  String smsg = prefix + msg;
  Serial.println(smsg);
}

void debugMessage(String msg) {
  if ( DEBUG_MODE == 1 ) serialMessage("D:", msg);
}

void statusMessage() {
  String msg = getState() + ':' + getTimeOnState() + ':' + currentBucket + ':' + vibroTableState + ':' + conveyorState;

  serialMessage("S:", msg);
}

void confirmationMessage(String msg) {
  serialMessage("C:", msg);
}

void incorrectInputMessage(String msg) {
  errorMessage("Incorrect input data:" + msg);  
}

void errorMessage(String msg) {
  serialMessage("E:", msg);
}

