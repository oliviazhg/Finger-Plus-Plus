import asyncio
import websockets
import json
import random
import math
import time
import os
import paho.mqtt.client as mqtt
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

PORT = 8765
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_TOPIC  = "motor/command"
TOPIC_MYO_STATE = "sensor/myo/state"
TOPIC_LOGS = "system/logs"

current_myo_state = "UNKNOWN"
system_logs = ["Starting..."]
LOGS_LENGTH = 30 # Keep only the most recent 30 logs

def on_mqtt_message(client, userdata, msg):
    global current_myo_state, system_logs
    
    if msg.topic == TOPIC_MYO_STATE:
        current_myo_state = msg.payload.decode()
    elif msg.topic == TOPIC_LOGS:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_log = f"[{timestamp}] {msg.payload.decode()}"

        system_logs.insert(0, formatted_log)
        system_logs = system_logs[:LOGS_LENGTH]

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_mqtt_message
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.subscribe([(TOPIC_MYO_STATE, 0), (TOPIC_LOGS, 0)]) 
mqtt_client.loop_start()

async def handle_connection(websocket):
    print(f"React Client Connected")
    
    # send dummy sensor data to React
    async def send_sensor_data():
        global current_myo_state, system_logs
        try:
            while True:
                t = time.time()
                
                base_sweep_factor = math.sin(t * 0.5) * 0.5 + 0.5
                curl_factor = math.sin(t) * 0.5 + 0.5

                payload = {
                    "angles": {
                        "base": base_sweep_factor,
                        "j1": curl_factor * 1.2,
                        "j2": curl_factor * 1.5,
                        "j3": curl_factor * 1.0 
                    },
                    "sensors": {
                        "flex": int(curl_factor * 90),
                        "force": round(curl_factor * 5, 2)
                    },
                    "myo": {
                        "state": current_myo_state
                    },
                    "logs": system_logs
                }
                await websocket.send(json.dumps(payload))
                await asyncio.sleep(0.02)
        except websockets.exceptions.ConnectionClosed:
            pass

    # receive manual control commands from React
    async def receive_commands():
        try:
            async for message in websocket:
                try:
                    command = json.loads(message)
                except json.JSONDecodeError:
                    print("Received invalid JSON.")
                    continue 

                if command.get("type") == "control":
                    motor_id = command.get("motor")
                    action = command.get("action")
                    direction = command.get("dir", "forward")                    
                    mqtt_payload = {}

                    if action == "start":
                        # Forward = Max Position, Backward = Min Position
                        target_pos = 0
                        if motor_id == 1:
                            target_pos = 500 if direction == "forward" else -1000
                        elif motor_id == 2:
                            target_pos = 8000 if direction == "forward" else 2000

                        mqtt_payload = {
                            "id": motor_id, 
                            "mode": "move", 
                            "position": target_pos
                        }
                        print(f"-> MQTT: Motor {motor_id} GO {direction.upper()} ({target_pos})")
                        mqtt_client.publish(TOPIC_LOGS, f"[UI] Motor {motor_id} {direction.upper()}")
                        
                    elif action == "stop":
                        # Stop = Tell motor driver to hold current position
                        mqtt_payload = {
                            "id": motor_id, 
                            "mode": "stop"
                        }
                        print(f"-> MQTT: Motor {motor_id} STOP")
                        mqtt_client.publish(TOPIC_LOGS, f"[UI] Motor {motor_id} STOP")

                    if mqtt_payload:
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(mqtt_payload))

                elif command.get("type") == "set_mode":
                    new_mode = command.get("mode")
                    print(f"Control mode changed to: {new_mode}")
                    mqtt_client.publish("system/control_mode", new_mode)
                    mqtt_client.publish(TOPIC_LOGS, f"[System] Mode changed to {new_mode.upper()}")
                
        except websockets.exceptions.ConnectionClosed:
            pass

    # run tasks concurrently
    await asyncio.gather(send_sensor_data(), receive_commands())

async def main():
    print(f"Starting Finger_OS Server on ws://localhost:{PORT}")
    async with websockets.serve(handle_connection, "localhost", PORT):
        await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        mqtt_client.loop_stop()
        print("Stopped.")
