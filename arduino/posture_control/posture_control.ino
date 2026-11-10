/*
 AI posture recognition - Arduino UNO / FSR x4 / IBT-2
 Protocol: DATA,fsr1,fsr2,fsr3,fsr4,total
 Commands: F feedback, S emergency stop, R reset, ? status
 Return is NOT spring/automatic return: Arduino explicitly commands reverse motor direction after 3s hold.
 Sequence: motorExtend 5s -> motorStop 2s -> motorRetract 5s -> motorStop 3s

 IMPORTANT: MOTION_ENABLED stays false until limit protection,
 direction, wiring, travel, and physical emergency power cutoff
 are verified with the actuator UNLOADED.
*/
const int FSR1=A0, FSR2=A1, FSR3=A2, FSR4=A4;
const int RPWM=5, LPWM=6, R_EN=7, L_EN=4;
const unsigned long SENSOR_INTERVAL = 1000; // 1초 측정
const unsigned long EXTEND_MS = 5000;       // 5초 돌출
const unsigned long HOLD_MS = 2000;         // 2초 정지
const unsigned long RETRACT_MS = 5000;      // 5초 복귀
const unsigned long SETTLE_MS = 3000;       // 복귀 후 3초 정지
const int MOTOR_PWM=160;         //PWM 출력 세기
const bool MOTION_ENABLED=true; // 리니어 엑추에이터 모션 구동

enum MotorState {
  IDLE, EXTENDING, HOLDING, RETRACTING, SETTLING, STOPPED
};
MotorState motorState=IDLE;
unsigned long stateStarted=0, lastSensorTime=0;

void motorStop(){ analogWrite(RPWM,0); analogWrite(LPWM,0); digitalWrite(R_EN,LOW); digitalWrite(L_EN,LOW); }
void motorExtend(){ motorStop(); digitalWrite(R_EN,HIGH); digitalWrite(L_EN,HIGH); analogWrite(RPWM,MOTOR_PWM); analogWrite(LPWM,0); }
void motorRetract(){ motorStop(); digitalWrite(R_EN,HIGH); digitalWrite(L_EN,HIGH); analogWrite(RPWM,0); analogWrite(LPWM,MOTOR_PWM); }
void emergencyStop(){ motorStop(); motorState=STOPPED; Serial.println("STATUS,EMERGENCY_STOP"); }

void setup(){
 pinMode(RPWM,OUTPUT); pinMode(LPWM,OUTPUT); pinMode(R_EN,OUTPUT); pinMode(L_EN,OUTPUT);
 motorStop(); Serial.begin(9600); Serial.println("STATUS,READY");
 if(!MOTION_ENABLED) Serial.println("STATUS,MOTION_LOCKED");
}
void sendSensorData(){
 int a=analogRead(FSR1), b=analogRead(FSR2), c=analogRead(FSR3), d=analogRead(FSR4);
 Serial.print("DATA,"); Serial.print(a); Serial.print(','); Serial.print(b);
 Serial.print(','); Serial.print(c); Serial.print(','); Serial.print(d);
 Serial.print(','); Serial.println(a+b+c+d);
}
void readPythonCommand(){
 while(Serial.available()>0){
  char command=Serial.read();
  if(command=='S'){ emergencyStop(); }
  else if(command=='R'){
   if(motorState==STOPPED){ motorStop(); motorState=IDLE; Serial.println("STATUS,RESET"); }
  }
  else if(command=='F'){
   if(motorState==STOPPED) Serial.println("STATUS,STOPPED");
   else if(motorState!=IDLE) Serial.println("STATUS,BUSY");
   else if(!MOTION_ENABLED) Serial.println("STATUS,FEEDBACK_RECEIVED_MOTION_LOCKED");
   else { motorState=EXTENDING; stateStarted=millis(); motorExtend(); Serial.println("STATUS,EXTENDING"); }
  }
  else if(command=='?'){
   if(motorState==STOPPED) Serial.println("STATUS,STOPPED");
   else if(!MOTION_ENABLED) Serial.println("STATUS,MOTION_LOCKED");
   else if(motorState==IDLE) Serial.println("STATUS,READY");
   else Serial.println("STATUS,BUSY");
  }
 }
}
void updateMotor(){
 unsigned long now=millis();
 switch(motorState){
  case EXTENDING:
   if(now-stateStarted>=EXTEND_MS){ motorStop(); motorState=HOLDING; stateStarted=now; Serial.println("STATUS,HOLDING"); }
   break;
  case HOLDING:
   if(now-stateStarted>=HOLD_MS){
    motorRetract(); // Explicit reverse command for powered return
    motorState=RETRACTING;
    stateStarted=now;
    Serial.println("STATUS,RETRACTING");
   }
   break;

   case RETRACTING:
   if (now - stateStarted >= RETRACT_MS) {
    motorStop();
    motorState = SETTLING;
    stateStarted = now;
    Serial.println("STATUS,SETTLING");
   }
   break;

  case SETTLING:
   if (now - stateStarted >= SETTLE_MS) {
    motorStop();
    motorState = IDLE;
    lastSensorTime = now;
    Serial.println("STATUS,COMPLETE");
   }
   break;
 }
}
void loop() {
  readPythonCommand();
  updateMotor();

  unsigned long now = millis();

  // 피드백 중에는 센서 측정·전송을 하지 않음.
  // 명령 처리는 계속하므로 S 긴급 정지는 수신 가능.
  if (motorState == IDLE &&
      now - lastSensorTime >= SENSOR_INTERVAL) {
    lastSensorTime = now;
    sendSensorData();
  }
}