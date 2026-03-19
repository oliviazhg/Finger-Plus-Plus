import subprocess
import time
import sys

def main():
    processes = []
    print("Starting Finger Plus Plus...")

    try:
        # Start Motor Driver
        # print("Starting Motor Driver...")
        # p_motor = subprocess.Popen([sys.executable, "motor_driver_json.py"])
        # processes.append(p_motor)
        # time.sleep(2)

        # Start FSR Controller (comm_bridge)
        print("Starting ESP32 Sensor Bridge...")
        p_fsr = subprocess.Popen([sys.executable, "comm_bridge.py"])
        processes.append(p_fsr)

        # Start Backend
        print("Starting Server...")
        p_ui = subprocess.Popen([sys.executable, "finger_data.py"]) 
        processes.append(p_ui)

        # Start Classification Model
        print("Starting Classification Model...")
        p_inference = subprocess.Popen([sys.executable, "run_inference.py"])
        processes.append(p_inference)

        # Start Myo Controller
        print("Starting Myo Controller...")
        p_myo = subprocess.Popen([sys.executable, "myo_controller.py"])
        processes.append(p_myo)

        print("\nSYSTEM RUNNING")
        print("Ctrl+C to stop everything.\n")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")

        for p in processes:
            try:
                # Give each process up to 3 seconds to shut down
                p.wait(timeout=3.0) 
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait()
                
        print("System shutdown complete.")

if __name__ == "__main__":
    main()