#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "*";
const char* password = "*";
const char* mqtt_server = "172.20.10.11";

WiFiClient espClient;
PubSubClient client(espClient);

const int FSR_PIN = A0;

const int PRESS_THRESHOLD = 800;
const int HARD_THRESHOLD  = 3000;
const int REQUIRED_PRESSES = 3;
const unsigned long WINDOW_MS = 2000;

bool fsrModeActive = false;
int hardPressCount = 0;
bool lastWasHard = false;
unsigned long windowStartMs = 0;

String lastFingerState = "unknown";

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
    } else {
      Serial.println("Failed to connect to MQTT, trying again");
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  int fsrValue = analogRead(FSR_PIN);
  unsigned long now = millis();

  // Window timeout logic
  if (hardPressCount > 0 && (now - windowStartMs > WINDOW_MS)) {
    hardPressCount = 0;
    Serial.println("Hard-press window expired -> reset");
  }

  // Rising-edge detect a hard press
  if (fsrValue > HARD_THRESHOLD) {
    if (!lastWasHard) {
      lastWasHard = true;

      if (hardPressCount == 0) {
        windowStartMs = now;
      }

      hardPressCount++;
      Serial.print("Hard press ");
      Serial.print(hardPressCount);
      Serial.print("/");
      Serial.print(REQUIRED_PRESSES);
      Serial.println();

      // Toggle mode when sequence completed
      if (hardPressCount >= REQUIRED_PRESSES) {
        fsrModeActive = !fsrModeActive;
        hardPressCount = 0;

        lastFingerState = "unknown";

        if (fsrModeActive) {
          Serial.println(">>> FSR MODE ACTIVATED <<<");
          client.publish("fsr/mode", "1");
        } else {
          Serial.println(">>> FSR MODE DEACTIVATED <<<");
          client.publish("fsr/mode", "0");
        }
      }
    }
  } else {
    lastWasHard = false;
  }
    // Finger control only when mode active
  if (fsrModeActive) {
    String currentFingerState = "";
    Serial.print("Raw: ");
    Serial.print(fsrValue);
    Serial.print("\t");

    if (fsrValue > PRESS_THRESHOLD) {
      currentFingerState = "close";
    } else {
      currentFingerState = "open";
    }

    if (currentFingerState != lastFingerState) {
      Serial.print("State Change: ");
      Serial.println(currentFingerState);
      
      client.publish("fsr/finger", currentFingerState.c_str());
      
      lastFingerState = currentFingerState;
    }
  }

  delay(50);
}