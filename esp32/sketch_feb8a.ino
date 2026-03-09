#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "*";
const char* password = "*";
const char* mqtt_server = "172.20.10.11";

WiFiClient espClient;
PubSubClient client(espClient);

// Finger Sensors
// const int PIN_FINGER_FSR_BASE = A0; // Replace with actual pin
// const int PIN_FINGER_FSR_MID  = A1; // Replace with actual pin
// const int PIN_FINGER_FSR_TIP  = A2; // Replace with actual pin

// Motor Control Sensors
const int PIN_TOE_FSR_M1 = A1; // Replace with actual pin (Controls Motor 1)
const int PIN_TOE_FSR_M2 = A0; // Replace with actual pin (Controls Motor 2)

// EMA filter state
float emaM1 = 0.0;
float emaM2 = 0.0;
const float EMA_ALPHA = 0.2;
const int FSR_DEADBAND2 = 50;


// Change threshold — only publish if motor target moves by this much
const int CHANGE_THRESHOLD1 = 25;
const int CHANGE_THRESHOLD2 = 50;
int lastTargetM1 = 0;
int lastTargetM2 = 0;

// Timer for sending data at 10Hz (Dynamixel safe rate)
unsigned long lastTelemetryMs = 0;


void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  client.setServer(mqtt_server, 1883);
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Sensor")) {
      Serial.println("Connected to MQTT");
      client.publish("system/logs", "[ESP32] Hardware Ready");
    } else {
      Serial.println("Failed to connect to MQTT, trying again");
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  unsigned long now = millis();

  // ==========================================
  // --- SENSOR READINGS & PROPORTIONAL CONTROL ---
  // ==========================================
  if (now - lastTelemetryMs >= 50) {
    lastTelemetryMs = now;

    // EMA filter on both FSRs to reduce ADC noise
    emaM1 = EMA_ALPHA * analogRead(PIN_TOE_FSR_M1) + (1.0 - EMA_ALPHA) * emaM1;
    emaM2 = EMA_ALPHA * analogRead(PIN_TOE_FSR_M2) + (1.0 - EMA_ALPHA) * emaM2;

    // Deadband: treat near-zero readings as zero to prevent motor hunting at rest
    int rawMotor1Fsr = (int)emaM1;
    int rawMotor2Fsr = (emaM2 < FSR_DEADBAND2) ? 0 : (int)emaM2;

    int fsrBase = 10;
    int fsrMid  = 11;
    int fsrTip  = 12;

    // TODO: Read IMU values from I2C
    int imuBase = 2;
    int imuMid  = 3;
    int imuTip  = 4;

    // Build JSON Payload for UI: {"fsr": [x, y, z], "imu": [a, b, c]}
    StaticJsonDocument<200> doc;
    JsonArray fsrArray = doc.createNestedArray("fsr");
    fsrArray.add(fsrBase);
    fsrArray.add(fsrMid);
    fsrArray.add(fsrTip);

    JsonArray imuArray = doc.createNestedArray("imu");
    imuArray.add(imuBase);
    imuArray.add(imuMid);
    imuArray.add(imuTip);

    JsonArray toeFsrArray = doc.createNestedArray("toe_fsr");
    toeFsrArray.add(rawMotor1Fsr);
    toeFsrArray.add(rawMotor2Fsr);

    char telemetryBuffer[200];
    serializeJson(doc, telemetryBuffer);
    client.publish("sensor/hardware_telemetry", telemetryBuffer);

    // --- TOE CONTROL (For Motors) ---

    //position
    int targetM1 = map(rawMotor1Fsr, 0, 3000, 4300, 3000); 
    targetM1 = constrain(targetM1, 3000, 4300); // Constrain prevents going past bounds if sensor > 3000
    //bending
    int targetM2 = map(rawMotor2Fsr, 0, 3000, 3000, 6900); 
    targetM2 = constrain(targetM2, 3000, 6900);

    // Only publish if either motor target has changed meaningfully
    // Prevents flooding Dynamixel with redundant position commands
    bool m1Changed = abs(targetM1 - lastTargetM1) > CHANGE_THRESHOLD1;
    bool m2Changed = abs(targetM2 - lastTargetM2) > CHANGE_THRESHOLD2;

    if (m1Changed || m2Changed) {
      // Build JSON Payload for comm_bridge.py
      StaticJsonDocument<100> propDoc;
      propDoc["m1"] = targetM1;
      propDoc["m2"] = targetM2;

      char propBuf[100];
      serializeJson(propDoc, propBuf);

      // Send it continuously (comm_bridge.py will ignore it if Myo mode is active)
      client.publish("fsr/finger", propBuf);

      lastTargetM1 = targetM1;
      lastTargetM2 = targetM2;
    }
  }

  // and delay() here can cause MQTT keepalive misses
}