#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Dynamixel2Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

const char* ssid = "";
const char* password = "";
const char* mqtt_server = "172.20.10.11";

WiFiClient espClient;
PubSubClient client(espClient);

const int DXL_DIR_PIN = D3;
const int DXL_TX_PIN  = D6; 
const int DXL_RX_PIN  = D7; 

const uint8_t DXL_ID_1 = 1;
const uint8_t DXL_ID_2 = 2;
const float DXL_PROTOCOL_VERSION = 2.0;
const int32_t BAUDRATE = 115200;

#define DXL_SERIAL Serial1
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

unsigned long lastTelemetryMs = 0;

const int B1_SDA = D4;
const int B1_SCL = D5;
const int B2_SDA = D9;
const int B2_SCL = D8; 

Adafruit_ADS1115 ads_base;   // Bus 1, 0x49
Adafruit_ADS1115 ads_middle; // Bus 2, 0x48
Adafruit_ADS1115 ads_tip;    // Bus 2, 0x49

// Calibration ranges
const int BASE_MIN = 24489;  const int BASE_MAX = 24770;
const int MID_MIN  = 24442;  const int MID_MAX  = 24846;
const int TIP_MIN  = 24509;  const int TIP_MAX  = 24791;

String current_sys_mode = "ui";

int32_t latest_m1_target = -1;
int32_t latest_m2_target = -1;
bool m1_needs_update = false;
bool m2_needs_update = false;


void setupMotors() {
  dxl.begin(BAUDRATE);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  uint8_t motor_ids[] = {DXL_ID_1, DXL_ID_2};
  
  for (int i = 0; i < 2; i++) {
    uint8_t id = motor_ids[i];
    Serial.printf("Pinging Motor %d...\n", id);
    
    if (dxl.ping(id)) {
      dxl.torqueOff(id);
      
      dxl.setOperatingMode(id, OP_EXTENDED_POSITION); 
      
      dxl.writeControlTableItem(ControlTableItem::PROFILE_ACCELERATION, id, 50);
      dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, id, 300);
      
      dxl.torqueOn(id);
      Serial.printf("  -> SUCCESS: Motor %d Ready\n", id);
    } else {
      Serial.printf("  -> FAIL: Motor %d not found.\n", id);
    }
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  if (strcmp(topic, "system/control_mode") == 0) {
    current_sys_mode = String(message);
    Serial.printf("[SYSTEM] Mode changed to: %s\n", message);
    return;
  }

  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, message);
  if (error) return;

  if (strcmp(topic, "fsr/finger/m1") == 0) {
    if (current_sys_mode != "fsr") return; 
    latest_m1_target = constrain((int32_t)doc["value"], 0, 1600);
    m1_needs_update = true;
    return;
  }
  
  if (strcmp(topic, "fsr/finger/m2") == 0) {
    if (current_sys_mode != "fsr") return; 
    latest_m2_target = constrain((int32_t)doc["value"], 2800, 6100);
    m2_needs_update = true;
    return;
  }

  // STANDARD UI & MYO CONTROL
  if (strcmp(topic, "motor/command") == 0) {
    int target_id = doc["id"];
    const char* mode = doc["mode"] | "move";
    
    if (strcmp(mode, "stop") == 0) {
    // Read current position and set it as the goal to stop
      int32_t current_pos = dxl.getPresentPosition(target_id);
      dxl.setGoalPosition(target_id, current_pos);
    // Serial.printf("[UI] Motor %d Halted at %d\n", target_id, current_pos);
      return;
    }

    int32_t target_pos = doc["position"];

    if (target_id == 1) {
      target_pos = constrain(target_pos, 0, 1600);
    } else if (target_id == 2) {
      target_pos = constrain(target_pos, 2800, 6100);
    }

    dxl.setGoalPosition(target_id, target_pos);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  DXL_SERIAL.begin(BAUDRATE, SERIAL_8N1, DXL_RX_PIN, DXL_TX_PIN);
  
  setupMotors();

  // Initialize FSRs (Bus 1)
  Wire.begin(B1_SDA, B1_SCL);     
  Wire.setClock(400000); 
  ads_base.setGain(GAIN_ONE); 
  ads_base.begin(0x49, &Wire);

  // Initialize FSRs (Bus 2)
  Wire.end();
  delay(10); 
  Wire.begin(B2_SDA, B2_SCL);
  Wire.setClock(400000);
  ads_middle.setGain(GAIN_ONE);
  ads_tip.setGain(GAIN_ONE);
  ads_middle.begin(0x48, &Wire);
  ads_tip.begin(0x49, &Wire);

  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  // Serial.printf("\nWiFi IP: %s\n", WiFi.localIP().toString().c_str());

  client.setServer(mqtt_server, 1883);
  client.setCallback(mqttCallback);
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32_Motor_Driver")) {
      Serial.println("connected"); 
      client.subscribe("motor/command");
      client.subscribe("system/control_mode"); 
      client.subscribe("fsr/finger/m1"); 
      client.subscribe("fsr/finger/m2"); 
      client.publish("system/logs", "[ESP32] Motor Driver Ready");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again");
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  
  client.loop(); 

  if (current_sys_mode == "fsr") {
    if (m1_needs_update) {
      dxl.setGoalPosition(1, latest_m1_target);
      m1_needs_update = false; // Reset until the next packet arrives
    }
    if (m2_needs_update) {
      dxl.setGoalPosition(2, latest_m2_target);
      m2_needs_update = false; // Reset until the next packet arrives
    }
  }

  unsigned long now = millis();

  // Send motor position and sensor data to UI every 100ms
  if (now - lastTelemetryMs >= 100) { 
    lastTelemetryMs = now;

    int32_t pos1 = dxl.getPresentPosition(DXL_ID_1);
    int32_t pos2 = dxl.getPresentPosition(DXL_ID_2);

    // Read Bus 1
    Wire.end(); 
    delay(1); 
    Wire.begin(B1_SDA, B1_SCL); 
    delay(1);
    int16_t raw_base = ads_base.readADC_SingleEnded(0);

    // Read Bus 2
    Wire.end(); 
    delay(1);
    Wire.begin(B2_SDA, B2_SCL);
    delay(1);
    int16_t raw_middle = ads_middle.readADC_SingleEnded(0);
    int16_t raw_tip    = ads_tip.readADC_SingleEnded(0);

    // Calibration
    float base_ratio   = constrain((float)(raw_base - BASE_MIN) / (BASE_MAX - BASE_MIN), 0.0, 1.0);
    float middle_ratio = constrain((float)(raw_middle - MID_MIN) / (MID_MAX - MID_MIN), 0.0, 1.0);
    float tip_ratio    = constrain((float)(raw_tip - TIP_MIN) / (TIP_MAX - TIP_MIN), 0.0, 1.0);

    int final_base   = (base_ratio * 100 < 2.0) ? 0 : (int)(base_ratio * 100);
    int final_middle = (middle_ratio * 100 < 2.0) ? 0 : (int)(middle_ratio * 100);
    int final_tip    = (tip_ratio * 100 < 2.0) ? 0 : (int)(tip_ratio * 100);

    StaticJsonDocument<100> motorDoc;
    motorDoc["m1_pos"] = pos1;
    motorDoc["m2_pos"] = pos2;
    char motorBuffer[100];
    serializeJson(motorDoc, motorBuffer);
    client.publish("motor/telemetry", motorBuffer);

    StaticJsonDocument<200> sensorDoc;
    
    JsonArray fsrArray = sensorDoc.createNestedArray("fsr");
    fsrArray.add(final_base);
    fsrArray.add(final_middle);
    fsrArray.add(final_tip);

    JsonArray imuArray = sensorDoc.createNestedArray("imu");
    imuArray.add(0);
    imuArray.add(0);
    imuArray.add(0);

    char sensorBuffer[200];
    serializeJson(sensorDoc, sensorBuffer);
    client.publish("sensor/hardware_telemetry1", sensorBuffer);
  }
}