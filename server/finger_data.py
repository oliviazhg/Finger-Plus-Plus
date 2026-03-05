import asyncio
import websockets
import json
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
TOPIC_TELEMETRY = "motor/telemetry"
TOPIC_HARDWARE_SENSORS = "sensor/hardware_telemetry"

current_myo_state = "UNKNOWN"
system_logs = ["Starting..."]
LOGS_LENGTH = 30

# Store live hardware values
live_m1_pos = 150
live_m2_pos = 4000
live_fsr = [0, 0, 0]
live_imu = [0, 0, 0]

def map_range(x, in_min, in_max, out_min, out_max):
    """Maps a number from one range to another, with strict clamping"""
    clamped_x = max(min(x, max(in_min, in_max)), min(in_min, in_max))
    return (clamped_x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def on_mqtt_message(client, userdata, msg):
    global current_myo_state, system_logs, live_m1_pos, live_m2_pos, live_fsr, live_imu
    
    if msg.topic == TOPIC_MYO_STATE:
        current_myo_state = msg.payload.decode()
    elif msg.topic == TOPIC_LOGS:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_log = f"[{timestamp}] {msg.payload.decode()}"

        system_logs.insert(0, formatted_log)
        system_logs = system_logs[:LOGS_LENGTH]
    elif msg.topic == TOPIC_TELEMETRY:
        try:
            data = json.loads(msg.payload.decode())
            if "m1_pos" in data:
                live_m1_pos = data["m1_pos"]
            if "m2_pos" in data:
                live_m2_pos = data["m2_pos"]
        except json.JSONDecodeError:
            pass
    elif msg.topic == TOPIC_HARDWARE_SENSORS:
        try:
            data = json.loads(msg.payload.decode())
            if "fsr" in data:
                live_fsr = data["fsr"]
            if "imu" in data:
                live_imu = data["imu"]
        except json.JSONDecodeError:
            pass

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_mqtt_message
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.subscribe([(TOPIC_MYO_STATE, 0), (TOPIC_LOGS, 0), (TOPIC_TELEMETRY, 0), (TOPIC_HARDWARE_SENSORS, 0)]) 
mqtt_client.loop_start()

async def handle_connection(websocket):
    print(f"React Client Connected")
    
    async def send_sensor_data():
        global current_myo_state, system_logs, live_m1_pos, live_m2_pos, live_fsr, live_imu
        try:
            while True:
                # Motor 1: Resting at 150 (0.0), Sweep to -1100 (1.0)
                base_sweep_factor = map_range(live_m1_pos, 150, -1100, 0.0, 1.0)
                
                # Motor 2: Resting at 4000 (0.0), Curl to 8400 (1.0)
                curl_factor = map_range(live_m2_pos, 4000, 8400, 0.0, 1.0)

                payload = {
                    "angles": {
                        "base": base_sweep_factor,
                        "j1": curl_factor * 0.45,
                        "j2": curl_factor * 0.9,
                        "j3": curl_factor * 0.8
                    },
                    "sensors": {
                        "fsr": live_fsr,
                        "imu": live_imu,
                        "motors": [live_m1_pos, live_m2_pos]
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
                            target_pos = 150 if direction == "forward" else -1100
                        elif motor_id == 2:
                            target_pos = 8400 if direction == "forward" else 4000

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
