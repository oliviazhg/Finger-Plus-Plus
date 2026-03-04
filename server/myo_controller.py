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

current_mode = "myo" 

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
    global current_mode
    if msg.topic == TOPIC_SYS_MODE:
        current_mode = msg.payload.decode()

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
        client.subscribe(TOPIC_SYS_MODE) # Listen for mode changes
        client.loop_start() # Runs a background thread to process incoming MQTT messages
        
        print(f"Connected to MQTT")
        client.publish(TOPIC_LOGS, "[Myo] Ready")
        last_state = -1

        while True:
            # 0 = Relaxed, 1 = Fist
            current_state = hand_position.value
            # print(current_state)

            if current_state != last_state:
                if current_state == 1:
                    m1_target = -1000
                    m2_target = 8000
                    state_name = "CLOSED (FIST)"
                    ui_state = "CLOSED"
                else:
                    m1_target = -1000
                    m2_target = 4000
                    state_name = "OPEN (RELAXED)"
                    ui_state = "OPEN"

                client.publish(TOPIC_MYO_STATE, ui_state)

                # Only execute if Myo is the active mode
                if current_mode == "myo":
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 1, "position": m1_target}))
                    client.publish(TOPIC_MOTOR, json.dumps({"id": 2, "position": m2_target}))
                    print(f"State: {state_name} -> Sending M1:{m1_target}, M2:{m2_target}")
                    client.publish(TOPIC_LOGS, f"[Myo] Executing: {state_name}")
                else:
                    print(f"Myo triggered '{state_name}', but mode is '{current_mode}'. Ignored.")

                last_state = current_state

            # fast polling
            time.sleep(0.02)

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