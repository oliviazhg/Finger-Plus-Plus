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

current_mode = "ui" 

def on_message(client, userdata, msg):
    global current_mode
    
    if msg.topic == TOPIC_SYS_MODE:
        current_mode = msg.payload.decode()
        print(f"[Myo Controller] System mode changed to: {current_mode}")
        
    elif msg.topic == TOPIC_MYO_STATE:
        predicted_class = msg.payload.decode()
        print(f"[Myo Controller] Detected Grip: {predicted_class.upper()}")
        
        # Only move motors if the UI has Myo mode selected
        if current_mode == "myo":
            
            # --- MAP ML CLASSES TO MOTOR ANGLES ---
            if predicted_class == "rest":
                m1, m2 = 1020, 2950
            elif predicted_class == "palm":
                m1, m2 = 505, 2950
            elif predicted_class == "cylindrical":
                m1, m2 = 480, 4650
            elif predicted_class == "lateral":
                m1, m2 = 700, 5000
            else:
                return

            # Publish the commands to the motor driver
            client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1}))
            client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": m2}))

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(TOPIC_SYS_MODE)
        client.subscribe(TOPIC_MYO_STATE) # Listen for inference outputs
        client.loop_forever() # Keep the script alive forever
        
    except KeyboardInterrupt:
        print("\nShutting down Myo Controller...")
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()