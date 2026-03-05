import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, Grid } from "@react-three/drei";
import { Cpu, Gamepad2, Settings2 } from "lucide-react";
import "./App.css";
import DataSidebar from "./components/DataSidebar";
import FingerModel from "./components/FingerModel";
import ControlOverlay from "./components/ControlOverlay";

export default function FingerDashboard() {
  const [data, setData] = useState({
    angles: { base: 0, j1: 0, j2: 0, j3: 0 },
    sensors: {
      fsr: [0, 0, 0],
      imu: [0, 0, 0],
      motors: [150, 4000],
    },
    myo: { state: "UNKNOWN" },
    logs: ["Started..."],
  });

  const [isSimulating, setIsSimulating] = useState(false);
  const [controlMode, setControlMode] = useState("myo"); // 'ui', 'myo', 'fsr'

  const [activeKeys, setActiveKeys] = useState({
    up: false,
    down: false,
    left: false,
    right: false,
  });
  const socket = useRef(null);

  // WebSocket Logic
  useEffect(() => {
    socket.current = new WebSocket("ws://localhost:8765");
    socket.current.onmessage = (event) => {
      const incoming = JSON.parse(event.data);
      setData((prev) => ({
        ...prev,
        ...incoming,
      }));
    };
    return () => socket.current.close();
  }, []);

  const handleModeChange = (newMode) => {
    setControlMode(newMode);
    setIsSimulating(false);
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: "set_mode", mode: newMode }));
    }
  };

  // Keyboard Event Listeners (Only active if controlMode === 'ui')
  useEffect(() => {
    if (controlMode !== "ui") return;

    const handleKeyDown = (e) => {
      let key = null;
      if (e.key === "ArrowUp") key = "up";
      if (e.key === "ArrowDown") key = "down";
      if (e.key === "ArrowLeft") key = "left";
      if (e.key === "ArrowRight") key = "right";

      if (key && !activeKeys[key]) {
        setActiveKeys((prev) => ({ ...prev, [key]: true }));
        const motor = key === "up" || key === "down" ? 1 : 2;
        const dir = key === "up" || key === "right" ? "forward" : "backward";
        socket.current.send(
          JSON.stringify({ type: "control", motor, dir, action: "start" }),
        );
      }
    };

    const handleKeyUp = (e) => {
      let key = null;
      if (e.key === "ArrowUp") key = "up";
      if (e.key === "ArrowDown") key = "down";
      if (e.key === "ArrowLeft") key = "left";
      if (e.key === "ArrowRight") key = "right";

      if (key) {
        setActiveKeys((prev) => ({ ...prev, [key]: false }));
        const motor = key === "up" || key === "down" ? 1 : 2;
        socket.current.send(
          JSON.stringify({ type: "control", motor, action: "stop" }),
        );
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [controlMode, activeKeys]);

  // Simulation Logic
  useEffect(() => {
    let interval;
    if (isSimulating && controlMode === "ui") {
      interval = setInterval(() => {
        const time = Date.now() * 0.002;

        const baseSweepFactor = Math.sin(time * 0.5) * 0.5 + 0.5;
        const curlFactor = Math.sin(time) * 0.5 + 0.5;

        setData((prev) => ({
          ...prev,
          angles: {
            base: baseSweepFactor,
            j1: curlFactor * 1.2,
            j2: curlFactor * 1.5,
            j3: curlFactor * 1.0,
          },
          sensors: {
            fsr: [
              Math.floor(curlFactor * 10),
              Math.floor(curlFactor * 30),
              Math.floor(curlFactor * 80),
            ],
            imu: [
              Math.floor(curlFactor * 1.2 * 57.3),
              Math.floor(curlFactor * 1.1 * 57.3),
              Math.floor(curlFactor * 0.4 * 57.3),
            ],
            motors: [
              Math.floor(baseSweepFactor * -1500 + 500),
              Math.floor(curlFactor * 4000 + 3000),
            ],
          },
          myo: { state: curlFactor > 0.6 ? "CLOSED" : "OPEN" },
        }));
      }, 50);
    }
    return () => clearInterval(interval);
  }, [isSimulating, controlMode]);

  return (
    <div className="dashboard-root">
      <div className="viewer-container">
        <div
          style={{
            position: "absolute",
            top: "24px",
            left: "24px",
            zIndex: 10,
          }}
        >
          <h1
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              margin: 0,
              fontSize: "20px",
              marginBottom: "15px",
            }}
          >
            FINGER MODEL
          </h1>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                background: "#1e293b",
                padding: "8px 12px",
                borderRadius: "8px",
                border: "1px solid #334155",
              }}
            >
              <Settings2
                size={16}
                color="#94a3b8"
                style={{ marginRight: "8px" }}
              />
              <select
                value={controlMode}
                onChange={(e) => {
                  handleModeChange(e.target.value);
                  e.target.blur();
                }}
                style={{
                  background: "transparent",
                  color: "#94a3b8",
                  border: "none",
                  outline: "none",
                  cursor: "pointer",
                  fontSize: "14px",
                }}
              >
                <option value="ui">Arrow keys</option>
                <option value="fsr">FSR sensor</option>
                <option value="myo">Myo band</option>
              </select>
            </div>

            <button
              onClick={() => setIsSimulating(!isSimulating)}
              disabled={controlMode !== "ui"}
              className="btn-primary"
              style={{
                backgroundColor: isSimulating ? "#ef4444" : "#10b981",
                opacity: controlMode !== "ui" ? 0.5 : 1,
              }}
            >
              {isSimulating ? "STOP SIM" : "TEST MODE"}
            </button>
          </div>
        </div>

        {/* Manual Control Visualizer Overlay - Only show if UI mode is active */}
        {controlMode === "ui" && <ControlOverlay activeKeys={activeKeys} />}

        <div className="canvas-wrapper">
          <Canvas camera={{ position: [0.3, 0.4, 0.5], fov: 45 }}>
            <Stage environment="city" intensity={0.5} adjustCamera={false}>
              <FingerModel
                angles={data.angles}
                rotation={[-Math.PI / 2, 0, -Math.PI / 2]}
              />
            </Stage>
            <Grid infiniteGrid sectionColor="#334155" cellColor="#1f2937" />
            <OrbitControls
              makeDefault
              enablePan={true}
              panSpeed={2.0}
              enableDamping={true}
              dampingFactor={0.05}
              target={[0, -0.1, 0]}
            />
          </Canvas>
        </div>
      </div>
      <DataSidebar data={data} />
    </div>
  );
}
