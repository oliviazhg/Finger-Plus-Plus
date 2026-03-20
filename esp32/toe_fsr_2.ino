#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid        = "";
const char* password    = "";
const char* mqtt_server = "172.20.10.11";

WiFiClient espClient;
PubSubClient client(espClient);

const int PIN_TOE_FSR_M2 = A2;

float emaM2 = 0.0f;
const float EMA_ALPHA = 0.5f;

const int FSR_DEADBAND2 = 50;

bool indModeActive = false;

unsigned long lastLoopMs = 0;

const char* TOPIC_IND_MODE     = "fsr/toggle";
const char* TOPIC_FINGER_M2    = "fsr/finger/m2";
const char* TOPIC_TELEMETRY    = "sensor/hardware_telemetry_left";
const char* TOPIC_LOGS         = "system/logs";


void callback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];

  if (String(topic) == TOPIC_IND_MODE) {
    indModeActive = (msg == "1");
    Serial.print(">>> Independent mode: ");
    Serial.println(indModeActive ? "ON" : "OFF");
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32_Left")) {
      Serial.println("Connected to MQTT");
      client.subscribe(TOPIC_IND_MODE);
      client.publish(TOPIC_LOGS, "[ESP2-Left] Hardware Ready");
    } else {
      Serial.println("Failed to connect to MQTT, retrying...");
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  unsigned long now = millis();
  if (now - lastLoopMs < 100) return;
  lastLoopMs = now;

  emaM2 = EMA_ALPHA * analogRead(PIN_TOE_FSR_M2) + (1.0f - EMA_ALPHA) * emaM2;
  int rawM2 = (emaM2 < FSR_DEADBAND2) ? 0 : (int)emaM2;

  StaticJsonDocument<96> doc;
  doc["toe_m2"]   = rawM2;
  doc["ind_mode"] = indModeActive;

  char telBuf[96];
  serializeJson(doc, telBuf);
  client.publish(TOPIC_TELEMETRY, telBuf);

  if (indModeActive) {
    int targetM2 = map(rawM2, 0, 3000, 2000, 5000);
    targetM2 = constrain(targetM2, 2000, 5000);

    StaticJsonDocument<64> m2Doc;
    m2Doc["value"]  = targetM2;
    m2Doc["active"] = indModeActive;

    char m2Buf[64];
    serializeJson(m2Doc, m2Buf);
    client.publish(TOPIC_FINGER_M2, m2Buf);
}

  Serial.print("Toe M2: "); Serial.print(rawM2);
  Serial.print("  |  State: ");
  Serial.println(indModeActive ? "ACTIVE" : "IDLE");
}