import os
import time
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
TOPIC_MODE = "fsr/mode"
TOPIC_FINGER = 'fsr/finger'
TOPIC_MOTOR  = "motor/command"
TOPIC_SYS_MODE = "system/control_mode"
TOPIC_LOGS = "system/logs"

system_active = False
last_finger_state = None

current_mode = "myo" 

def send_motor_command(client, m1_position, m2_position):
    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1_position}))
    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": m2_position}))
    print(f"[FSR Executing] motor 1: {m1_position}, motor 2: {m2_position}")

def on_message(client, userdata, msg):
    global system_active, last_finger_state, current_mode
    payload = msg.payload.decode()

    if msg.topic == TOPIC_SYS_MODE:
        current_mode = payload
        return

    # activate/deactivate
    if msg.topic == TOPIC_MODE:
        if payload == "1":
            system_active = True
            print("FSR SENSOR ACTIVATED")
            client.publish(TOPIC_LOGS, "[FSR] Sensor Activated")
        else:
            system_active = False
            last_finger_state = None
            print("FSR SENSOR DEACTIVATED")
            client.publish(TOPIC_LOGS, "[FSR] Sensor Deactivated")
            
    # move finger
    elif msg.topic == TOPIC_FINGER:
        if system_active:
            if payload != last_finger_state:
                
                # Only execute if FSR is the active mode
                if current_mode == "fsr":
                    if payload == "close":
                        send_motor_command(client, -1000, 7000)
                    elif payload == "open":
                        send_motor_command(client, -1000, 3000)
                        
                    client.publish(TOPIC_LOGS, f"[FSR] Executing: {payload.upper()}")
                else:
                    print(f"FSR triggered '{payload}', but current mode is '{current_mode}'. Ignored.")

                last_finger_state = payload
                print(f"FSR state Changed to: {payload}")

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.on_message = on_message
# Subscribe to the new system mode topic as well
client.subscribe([(TOPIC_MODE, 0), (TOPIC_FINGER, 0), (TOPIC_SYS_MODE, 0)])

print("Connected to MQTT")
print("Started... listening for sensors")
client.publish(TOPIC_LOGS, "[FSR] Ready")

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping...")

# while True:
#     # testing by hardcoding
#     target_id = 1
#     target_pos = 0

#     payload = {
#         "id": target_id,
#         "position": target_pos
#     }
    
#     client.publish(TOPIC_MOTOR, json.dumps(payload))
#     print(f"Sent: {payload}")
    
#     time.sleep(3.0)