import os
import time
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
TOPIC_TOGGLE     = "fsr/toggle"
TOPIC_SYS_MODE   = "system/control_mode"
TOPIC_LOGS       = "system/logs"

current_mode = "ui"

def on_message(client, userdata, msg):
    global current_mode
    payload = msg.payload.decode().strip()

    if msg.topic == TOPIC_SYS_MODE:
        current_mode = payload
        print(f"[mode] {current_mode}")
        return

    if msg.topic == TOPIC_TOGGLE:
        if payload == "1":
            new_mode = "fsr"
        elif payload == "0":
            new_mode = "myo"
        else:
            return
            
        client.publish(TOPIC_SYS_MODE, new_mode)
        client.publish(TOPIC_LOGS, f"[Hardware] Mode switched to {new_mode.upper()}")
        print(f"[toggle] Mode switched to: {new_mode}")
        return

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.on_message = on_message
client.subscribe([
    (TOPIC_TOGGLE,    0),
    (TOPIC_SYS_MODE,  0),
])

print("Connected to MQTT")
print("Started bridge")

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping...")