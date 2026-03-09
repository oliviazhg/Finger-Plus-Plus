import multiprocessing
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
TOPIC_LOGS = "system/logs"
TOPIC_FSR = 'sensor/fsr/raw'

CATEGORY_MAP = {
    0: {"name": "rest",         "ui": "RELAXED",      "m1": 150,   "m2": 6500, "use_fsr": False},
    1: {"name": "cynlindrical", "ui": "CYNLINDRICAL", "m1": -1100, "m2": 7050, "use_fsr": True},
    2: {"name": "ball",         "ui": "BALL",         "m1": -1100, "m2": 6900, "use_fsr": True},
    3: {"name": "lateral",      "ui": "LATERAL",      "m1": -600,  "m2": 7550, "use_fsr": False},
    4: {"name": "flat",         "ui": "FLAT",         "m1": 120,   "m2": 6650, "use_fsr": False}
}

live_fsr_value = 0
FSR_TARGET_FORCE = 80
current_mode = "ui" 

def classifier_worker(shared_position):
    """
    Worker process that runs the Myo classifier isolated in the background.
    """
    from fingerposition import run_classification_mode_with_shared_value

    try:
        run_classification_mode_with_shared_value(shared_position)
    except Exception as e:
        print(f"[Classifier Worker] Error: {e}")

def on_message(client, userdata, msg):
    global current_mode, live_fsr_value
    if msg.topic == TOPIC_SYS_MODE:
        current_mode = msg.payload.decode()
    elif msg.topic == TOPIC_FSR:
        try:
            live_fsr_value = int(msg.payload.decode())
        except ValueError:
            pass

def main():
    global current_mode
    # Create shared value for hand position
    # 'i' = signed integer, initial value = 0 (Relaxed)
    hand_position = multiprocessing.Value('i', 0)

    # Start classifier in background process
    classifier_process = multiprocessing.Process(
        target=classifier_worker,
        args=(hand_position,),
        daemon=True  # Process will terminate when main exits
    )

    client = None
    
    try:
        classifier_process.start()

        # Give classifier time to initialize
        print("Waiting for classifier to initialize...")
        time.sleep(3)
        print("Server ready!\n")
        
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_message = on_message # Attach callback
        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe([(TOPIC_SYS_MODE, 0), (TOPIC_FSR, 0)])
        client.loop_start() # Runs a background thread to process incoming MQTT messages
        
        print(f"Connected to MQTT")
        client.publish(TOPIC_LOGS, "[Myo] Ready")
        last_state = -1
        active_target_m2 = 4000

        while True:
            current_state = hand_position.value

            if current_state != last_state:
                # Look up the state (fallback to REST if unknown)
                state_data = CATEGORY_MAP.get(current_state, CATEGORY_MAP[0])
                
                m1_target = state_data["m1"]
                active_target_m2 = state_data["m2"]
                state_name = state_data["name"]
                ui_state = state_data["ui"]

                client.publish(TOPIC_MYO_STATE, ui_state)

                if current_mode == "myo":
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1_target}))
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": active_target_m2}))
                    print(f"State: {state_name} -> Sending M1:{m1_target}, M2:{active_target_m2}")
                    client.publish(TOPIC_LOGS, f"[Myo] Executing: {ui_state}")
                else:
                    print(f"Myo triggered '{state_name}', but mode is '{current_mode}'. Ignored.")

                last_state = current_state

            # if current_mode == "myo" and CATEGORY_MAP.get(current_state, {}).get("use_fsr"):
            #     if live_fsr_value < FSR_TARGET_FORCE:
            #         active_target_m2 += 50 
            #         active_target_m2 = min(active_target_m2, 8400) 
            #         client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": active_target_m2}))

            #     elif live_fsr_value >= FSR_TARGET_FORCE:
            #         client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "mode": "stop"}))
            #         client.publish(TOPIC_LOGS, f"[System] Grip Force Reached: {live_fsr_value}")
            #         print('FSR target force reached, stopping.')

            # fast polling
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\nCRASH: {e}")
    finally:
        # Clean up background process
        if classifier_process.is_alive():
            print("Terminating classifier process...")
            classifier_process.terminate()
            classifier_process.join(timeout=2)
            if classifier_process.is_alive():
                print("Force killing classifier process...")
                classifier_process.kill()
        if client:
            client.loop_stop()
            client.disconnect()

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()