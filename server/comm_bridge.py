import os
import time
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
TOPIC_TOGGLE = "fsr/toggle"
TOPIC_FINGER = 'fsr/finger'
TOPIC_MOTOR  = "motor/command"
TOPIC_SYS_MODE = "system/control_mode"
TOPIC_LOGS = "system/logs"

current_mode = "ui" 

def send_motor_command(client, m1_position, m2_position):
    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1_position}))
    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": m2_position}))

def on_message(client, userdata, msg):
    global current_mode
    payload = msg.payload.decode()

    if msg.topic == TOPIC_SYS_MODE:
        current_mode = payload
        return

    # Handle triple tap (mode toggle between myo and fsr)
    # if msg.topic == TOPIC_TOGGLE:
    #     new_mode = "fsr" if current_mode == "myo" else "myo"
    #     client.publish(TOPIC_SYS_MODE, new_mode)
    #     client.publish(TOPIC_LOGS, f"[Hardware] Triple-Tap: Mode switched to {new_mode.upper()}")
      
    elif msg.topic == TOPIC_FINGER:
        if current_mode == "fsr":
            try:
                data = json.loads(payload)
                # print(f"[fsr] {data}")
                send_motor_command(client, data["m1"], data["m2"])
            except (json.JSONDecodeError, KeyError):
                pass

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.on_message = on_message
client.subscribe([(TOPIC_TOGGLE, 0), (TOPIC_FINGER, 0), (TOPIC_SYS_MODE, 0)])

print("Connected to MQTT")
print("Started... listening for sensors")
client.publish(TOPIC_LOGS, "[FSR] Ready")

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping...")