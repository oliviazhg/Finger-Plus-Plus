#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid        = "";
const char* password    = "";
const char* mqtt_server = "172.20.10.11";

WiFiClient espClient;
PubSubClient client(espClient);

const int PIN_HEEL_FSR   = A2;
const int PIN_TOE_FSR_M1 = A0;

#define HEEL_ENTRY_THRESHOLD  3050
#define HEEL_HOLD_MS          3000  // Require 3 seconds (3000ms) to activate
#define M1_BASELINE           0   // The resting value of the M1 FSR to zero out

#define FSR1_IDLE_THRESHOLD   50    // FSR 1 rests at 0 (after baseline subtraction)
#define FSR2_IDLE_THRESHOLD   100   // FSR 2 raw value threshold

float emaHeel = 0.0f;
float emaM1   = 0.0f;
const float EMA_ALPHA = 0.5f;

enum HeelState { HEEL_IDLE, HEEL_ACTIVE };
HeelState heelState = HEEL_IDLE;

#define TOE_IDLE_EXIT_MS    5000
unsigned long toeIdleStartMs = 0;
bool toesWereIdle = false;

unsigned long heelPressStartMs = 0;
bool isHeelPressing = false;

// --- Wiretap Tracker for Left Toe (Raw FSR 2) ---
int latest_fsr2_raw = 0;
unsigned long lastFsr2UpdateMs = 0;
#define FSR2_STALE_MS 500   // zero out left toe if no update received for 500ms

unsigned long lastLoopMs = 0;

const char* TOPIC_IND_MODE       = "fsr/toggle";
const char* TOPIC_FINGER_M1      = "fsr/finger/m1";
const char* TOPIC_LEFT_TELEMETRY = "sensor/hardware_telemetry_left";
const char* TOPIC_TELEMETRY      = "sensor/hardware_telemetry";
const char* TOPIC_LOGS           = "system/logs";

void callback(char* topic, byte* payload, unsigned int length) {
  if (String(topic) == TOPIC_LEFT_TELEMETRY) {
    char message[length + 1];
    memcpy(message, payload, length);
    message[length] = '\0';

    StaticJsonDocument<128> doc;
    DeserializationError error = deserializeJson(doc, message);
    
    // Check if it parsed successfully and grab the raw FSR value
    if (!error && doc.containsKey("toe_m2")) {
      latest_fsr2_raw    = doc["toe_m2"];
      lastFsr2UpdateMs   = millis();
    }
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
    Serial.println("[MQTT] Attempting connection...");
    if (client.connect("ESP32_Right")) {
      Serial.println("[MQTT] Connected!");
      client.publish(TOPIC_LOGS, "[ESP1-Right] Hardware Ready");
      
      client.subscribe(TOPIC_LEFT_TELEMETRY); 
    } else {
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();   // called every iteration so incoming messages are never delayed

  unsigned long now = millis();
  if (now - lastLoopMs < 100) return;
  lastLoopMs = now;

  // Zero out stale left-toe value if left ESP32 has gone silent
  if (now - lastFsr2UpdateMs > FSR2_STALE_MS) latest_fsr2_raw = 0;

  emaHeel = EMA_ALPHA * analogRead(PIN_HEEL_FSR) + (1.0f - EMA_ALPHA) * emaHeel;
  emaM1   = EMA_ALPHA * analogRead(PIN_TOE_FSR_M1) + (1.0f - EMA_ALPHA) * emaM1;

  int rawHeel = (int)emaHeel;
  int rawM1   = (int)emaM1;

  // Zero out the M1 FSR by subtracting the 400 resting baseline
  int zeroedM1 = rawM1 - M1_BASELINE;
  if (zeroedM1 < 0) zeroedM1 = 0;

  switch (heelState) {

    case HEEL_IDLE:
      if (rawHeel >= HEEL_ENTRY_THRESHOLD) {
        // Start the 3-second timer
        if (!isHeelPressing) {
          isHeelPressing = true;
          heelPressStartMs = now;
          Serial.println("[HEEL] Press detected... holding for 3 seconds");
        } 
        // If 3 seconds have successfully passed
        else if (now - heelPressStartMs >= HEEL_HOLD_MS) {
          heelState      = HEEL_ACTIVE;
          toesWereIdle   = false;
          toeIdleStartMs = 0;
          isHeelPressing = false;

          Serial.println(">>> TOGGLE ON — Independent mode ACTIVATED");
          client.publish(TOPIC_IND_MODE, "1");
          client.publish(TOPIC_LOGS, "[ESP1] *** Independent mode ON — heel held 3s ***");
        }
      } else {
        // Let go too early
        if (isHeelPressing) {
          Serial.println("[HEEL] Released early — activation cancelled.");
        }
        isHeelPressing = false;
      }
      break;

    case HEEL_ACTIVE: {
      bool fsr1_idle = (zeroedM1 < FSR1_IDLE_THRESHOLD);
      bool fsr2_idle = (latest_fsr2_raw < FSR2_IDLE_THRESHOLD);
      bool bothToesIdle = (fsr1_idle && fsr2_idle);

      if (bothToesIdle && !toesWereIdle) {
        toeIdleStartMs = now;
        Serial.println("[TOE] Both Toes raised — starting 5s exit timer");
      }

      if (bothToesIdle && toeIdleStartMs > 0) {
        unsigned long elapsed = now - toeIdleStartMs;

        if (elapsed >= TOE_IDLE_EXIT_MS) {
          heelState = HEEL_IDLE;
          Serial.println(">>> TOGGLE OFF — Independent mode DEACTIVATED");
          client.publish(TOPIC_IND_MODE, "0");
          client.publish(TOPIC_LOGS, "[ESP1] *** Independent mode OFF — toes raised 5s ***");
        }
      }

      if (!bothToesIdle) {
        if (toesWereIdle) {
          Serial.println("[TOE] Movement detected (FSR1 or FSR2) — exit timer reset");
        }
        toeIdleStartMs = 0;
      }

      toesWereIdle = bothToesIdle;
      break;
    }
  }

  StaticJsonDocument<192> doc;
  doc["heel_fsr"] = rawHeel;
  doc["toe_m1"]   = zeroedM1;
  doc["ind_mode"] = (heelState == HEEL_ACTIVE);
  doc["toe_m2"]   = latest_fsr2_raw;

  char telBuf[192];
  serializeJson(doc, telBuf);
  client.publish(TOPIC_TELEMETRY, telBuf);

  int targetM1 = 0;
  if (heelState == HEEL_ACTIVE) {
    targetM1 = map(zeroedM1, 0, 3000 - M1_BASELINE, 0, 1600);
    targetM1 = constrain(targetM1, 0, 1600);

    StaticJsonDocument<64> m1Doc;
    m1Doc["value"]  = targetM1;
    m1Doc["active"] = (heelState == HEEL_ACTIVE);

    char m1Buf[64];
    serializeJson(m1Doc, m1Buf);
    client.publish(TOPIC_FINGER_M1, m1Buf);
  }
}