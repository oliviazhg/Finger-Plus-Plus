#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Dynamixel2Arduino.h>

const char* ssid = "*";
const char* password = "*";
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

  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.println("Failed to parse incoming JSON.");
    return;
  }

  int target_id = doc["id"];
  const char* mode = doc["mode"] | "move";
  
  if (strcmp(mode, "stop") == 0) {
    // Read current position and set it as the goal to stop
    int32_t current_pos = dxl.getPresentPosition(target_id);
    dxl.setGoalPosition(target_id, current_pos);
    Serial.printf("[UI] Motor %d Halted at %d\n", target_id, current_pos);
    return;
  }

  int32_t target_pos = doc["position"];

  if (target_id == 1) {
    target_pos = constrain(target_pos, 3000, 4300);
  } else if (target_id == 2) {
    target_pos = constrain(target_pos, 3000, 6900);
  }

  dxl.setGoalPosition(target_id, target_pos);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  DXL_SERIAL.begin(BAUDRATE, SERIAL_8N1, DXL_RX_PIN, DXL_TX_PIN);
  
  setupMotors();

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
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();

  // Send motor position to UI every 50ms
  if (now - lastTelemetryMs >= 50) {
    lastTelemetryMs = now;

    int32_t pos1 = dxl.getPresentPosition(DXL_ID_1);
    int32_t pos2 = dxl.getPresentPosition(DXL_ID_2);

    // {"m1_pos": X, "m2_pos": Y}
    StaticJsonDocument<100> telemetryDoc;
    telemetryDoc["m1_pos"] = pos1;
    telemetryDoc["m2_pos"] = pos2;

    char telemetryBuffer[100];
    serializeJson(telemetryDoc, telemetryBuffer);
    
    client.publish("motor/telemetry", telemetryBuffer);
  }
}