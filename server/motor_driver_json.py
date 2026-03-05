import time
import json
import os
import threading
import paho.mqtt.client as mqtt
from dynamixel_sdk import *
from dotenv import load_dotenv

load_dotenv()

# MQTT config
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT   = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = "motor/command"
TOPIC_TELEMETRY = "motor/telemetry"

# motor config
DXL_ID_1 = 1
DXL_ID_2 = 2
BAUDRATE = 1000000
DEVICENAME = 'COM3' # /dev/ttyUSB0 on Pi
PROTOCOL_VERSION = 2.0

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Prevent serial collisions between the read loop and incoming MQTT commands
port_lock = threading.Lock()

def get_signed_position(unsigned_val):
    if unsigned_val > 2147483647:
        return unsigned_val - 4294967296
    return unsigned_val

def setup_motors():
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        quit()

    # set baud rate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        quit()

    # enable torque
    with port_lock:
        for motor_id in [DXL_ID_1, DXL_ID_2]:
            packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)
            packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_OPERATING_MODE, 4)
            packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_ACCELERATION, 50)
            packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, 300)
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 1)
            
            if dxl_comm_result != COMM_SUCCESS:
                print(f"ID {motor_id} Setup Error: {packetHandler.getTxRxResult(dxl_comm_result)}")
            else:
                print(f"Dynamixel {motor_id} Ready (Extended Mode)")

def move_motor(motor_id, position):
    if position < 0:
        position = 4294967296 + position

    # goal position
    with port_lock:
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, position)
    
        if dxl_comm_result != COMM_SUCCESS:
            print(f"move motor {motor_id} Comm Error: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"move motor {motor_id} Packet Error: {packetHandler.getRxPacketError(dxl_error)}")

def stop_motor(motor_id):
    # stops motor by reading current position and setting it as goal
    with port_lock:
        current_pos, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
        packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, current_pos)
    print(f"Motor {motor_id} Halted at {current_pos}")

def shutdown_motors():
    # disable torque before quitting
    with port_lock:
        for motor_id in [DXL_ID_1, DXL_ID_2]:
            packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)
        portHandler.closePort()
    print("motors shutdown")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        target_id = int(payload['id'])
        mode = payload.get('mode', 'move')
        
        if mode == "stop":
            print(f"UI Command: Stopping Motor {target_id}")
            stop_motor(target_id)
            return

        target_position = int(payload['position'])
        
        print(f"motor id: {target_id}, position: {target_position}")

        # define physical bounds
        if target_id == 1:
            if target_position < -1100: target_position = -1100
            if target_position > 150: target_position = 150
        if target_id == 2:
            if target_position < 4000: target_position = 4000
            if target_position > 8400: target_position = 8400

        move_motor(target_id, target_position)
        
        # delay to let the bus stabilize
        # time.sleep(0.01)

        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, target_id, ADDR_PRESENT_POSITION)
        
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[WARNING] Read Failed ID {target_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[CRITICAL] Hardware Error ID {target_id}: {packetHandler.getRxPacketError(dxl_error)}")
            print("Action: Please check if motor is stalled or overheated.")
        else:
            # print current position if success
            # real_position = get_signed_position(dxl_present_position)
            # print(f"[SUCCESS] ID {target_id} at {real_position}")
            pass

    except Exception as e:
        # prevents "list index out of range" crash
        print(f"[ERROR] Logic skipped: {e}")
        
if __name__ == "__main__":
    setup_motors()

    # setup MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    print(f"Connecting to MQTT...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        print(f"Listening for commands on '{MQTT_TOPIC}'...")

        client.loop_start()

        while True:
            with port_lock:
                pos1, comm1, err1 = packetHandler.read4ByteTxRx(portHandler, DXL_ID_1, ADDR_PRESENT_POSITION)
                pos2, comm2, err2 = packetHandler.read4ByteTxRx(portHandler, DXL_ID_2, ADDR_PRESENT_POSITION)
            
            if comm1 == COMM_SUCCESS and comm2 == COMM_SUCCESS:
                telemetry_payload = {
                    "m1_pos": get_signed_position(pos1),
                    "m2_pos": get_signed_position(pos2)
                }
                client.publish(TOPIC_TELEMETRY, json.dumps(telemetry_payload))
            
            time.sleep(0.05) # add delay to poll
       
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        client.loop_stop()
        shutdown_motors()