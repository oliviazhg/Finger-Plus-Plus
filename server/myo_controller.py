import time
import json
import os
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

TOPIC_MOTOR = "motor/command"
TOPIC_SYS_MODE = "system/control_mode"
TOPIC_MYO_STATE = "sensor/myo/state"
TOPIC_FINGER_TELEMETRY = "sensor/hardware_telemetry1"

current_mode = "ui" 
predicted_class = "rest"
target_m2 = 3000

FSR_FORCE_THRESHOLD = 20   # Stop when any FSR hits 20%
CURL_SPEED_STEP = 50
MAX_CYLINDRICAL_CURL = 6100

def on_message(client, userdata, msg):
    global current_mode, predicted_class, target_m2

    if msg.topic == TOPIC_SYS_MODE:
        current_mode = msg.payload.decode().strip()
        print(f"[Myo] System mode changed to: {current_mode}")
        return
        
    elif msg.topic == TOPIC_MYO_STATE:
        new_class = msg.payload.decode().strip()
        
        if new_class != predicted_class:
            predicted_class = new_class
            print(f"[Myo] Detected Grip: {predicted_class.upper()}")
            
            if current_mode == "myo":
                if predicted_class == "rest":
                    m1, target_m2 = 1020, 2950
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1}))
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": target_m2}))
                    
                elif predicted_class == "palm":
                    m1, target_m2 = 505, 2950
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1}))
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": target_m2}))
                    
                elif predicted_class == "lateral":
                    m1, target_m2 = 700, 5000
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1}))
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": target_m2}))
                    
                elif predicted_class == "cylindrical":
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": 480}))
        return

    elif msg.topic == TOPIC_FINGER_TELEMETRY:
        # Only run the tactile loop if we are actively trying to do a cylindrical grip
        if current_mode == "myo" and predicted_class == "cylindrical":
            try:
                data = json.loads(msg.payload.decode())
                if "fsr" in data:
                    fsr_values = data["fsr"] # [base, middle, tip]
                    max_force = max(fsr_values)
                    
                    if max_force < FSR_FORCE_THRESHOLD:
                        target_m2 += CURL_SPEED_STEP
                        target_m2 = min(target_m2, MAX_CYLINDRICAL_CURL)
                        
                        client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": target_m2}))
                    
            except (json.JSONDecodeError, KeyError):
                pass

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe([
            (TOPIC_SYS_MODE, 0),
            (TOPIC_MYO_STATE, 0),
            (TOPIC_FINGER_TELEMETRY, 0)
        ])
        print("Connected to MQTT. Myo Controller Active.")
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\nShutting down Myo Controller...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()