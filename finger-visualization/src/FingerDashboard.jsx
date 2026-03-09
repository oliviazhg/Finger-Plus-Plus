import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, Grid } from "@react-three/drei";
import { Cpu, Gamepad2, Settings2, Mic } from "lucide-react";
import "./App.css";
import DataSidebar from "./components/DataSidebar";
import FingerModel from "./components/FingerModel";
import ControlOverlay from "./components/ControlOverlay";

// Map the pitch to the motor encoding
function mapRange(value, inMin, inMax, outMin, outMax) {
  const clamped = Math.max(inMin, Math.min(value, inMax));
  return Math.floor(
    ((clamped - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin,
  );
}

export default function FingerDashboard() {
  const [data, setData] = useState({
    angles: { base: 0, j1: 0, j2: 0, j3: 0 },
    sensors: {
      fsr: [0, 0, 0],
      imu: [0, 0, 0],
      toe_fsr: [0, 0],
      motors: [150, 4000],
    },
    myo: { state: "UNKNOWN" },
    // system: { mode: "myo" },
    logs: ["Started..."],
  });

  const [isSimulating, setIsSimulating] = useState(false);
  const [controlMode, setControlMode] = useState("ui"); // 'ui', 'myo', 'fsr', 'voice'
  const [livePitch, setLivePitch] = useState(0);

  const [activeKeys, setActiveKeys] = useState({
    up: false,
    down: false,
    left: false,
    right: false,
  });
  const socket = useRef(null);
  const lastTimestamp = useRef(0); // Track previous message time
  const [metrics, setMetrics] = useState({ ping: 0, delta: 0 }); // Store the math results

  // useEffect(() => {
  //   if (data.system.mode !== controlMode) {
  //     setControlMode(data.system.mode);
  //   }
  // }, [data.system.mode]);

  // WebSocket Logic
  useEffect(() => {
    socket.current = new WebSocket("ws://localhost:8765");
    socket.current.onmessage = (event) => {
      const incoming = JSON.parse(event.data);
      const now = Date.now();

      // Calculate the metrics if a timestamp exists in the payload
      if (incoming.timestamp) {
        // True Latency: Current React Time - Backend Sent Time
        const ping = Math.max(0, now - incoming.timestamp);

        // Update Interval: Current Backend Time - Previous Backend Time
        const delta = lastTimestamp.current
          ? incoming.timestamp - lastTimestamp.current
          : 0;

        lastTimestamp.current = incoming.timestamp;
        setMetrics({ ping, delta });
      }

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
    setLivePitch(0);
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: "set_mode", mode: newMode }));
    }
  };

  const handleSetPosition = (motor, position) => {
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      socket.current.send(
        JSON.stringify({
          type: "set_position",
          motor,
          position: Number(position),
        }),
      );
    }
  };

  useEffect(() => {
    if (controlMode !== "voice") return;

    let audioCtx;
    let analyser;
    let microphone;
    let animationFrameId;
    let lastSendTime = 0;

    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;

        microphone = audioCtx.createMediaStreamSource(stream);
        microphone.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const detectPitch = () => {
          analyser.getByteFrequencyData(dataArray);

          let maxVal = 0;
          let maxIndex = 0;

          // Find the dominant frequency bin
          for (let i = 0; i < bufferLength; i++) {
            if (dataArray[i] > maxVal) {
              maxVal = dataArray[i];
              maxIndex = i;
            }
          }

          // Noise Gate: Only trigger if the volume is loud enough (max 255)
          if (maxVal > 130) {
            const freq = maxIndex * (audioCtx.sampleRate / analyser.fftSize);

            // Only react to a sensible whistling/humming range
            if (freq > 150 && freq < 1200) {
              setLivePitch(Math.floor(freq));

              const now = Date.now();
              if (now - lastSendTime > 100) {
                const mappedPos = mapRange(freq, 250, 900, 3000, 6900);
                handleSetPosition(2, mappedPos);
                lastSendTime = now;
              }
            }
          } else {
            setLivePitch(0); // Not loud enough
          }

          animationFrameId = requestAnimationFrame(detectPitch);
        };
        detectPitch();
      } catch (err) {
        console.error("Microphone access denied or failed", err);
        alert("Please allow microphone access to use Voice Control.");
      }
    };

    initAudio();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (audioCtx && audioCtx.state !== "closed") audioCtx.close();
    };
  }, [controlMode]);

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
                <option value="ui">[Manual] Arrow keys</option>
                <option value="fsr">[Toe] FSR sensor</option>
                <option value="myo">[EMG] Myo band</option>
                <option value="voice">[Manual] Voice pitch</option>
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

        {/* Manual Control Overlay */}
        {controlMode === "ui" && (
          <ControlOverlay
            activeKeys={activeKeys}
            onSetPosition={handleSetPosition}
          />
        )}

        {/* Voice Control Overlay */}
        {controlMode === "voice" && (
          <div
            style={{
              position: "absolute",
              bottom: "24px",
              left: "24px",
              zIndex: 10,
              background: "rgba(30, 41, 59, 0.8)",
              padding: "16px 24px",
              borderRadius: "12px",
              border: `1px solid ${livePitch > 0 ? "#10b981" : "#334155"}`,
              backdropFilter: "blur(4px)",
              display: "flex",
              alignItems: "center",
              gap: "16px",
            }}
          >
            <Mic
              size={24}
              color={livePitch > 0 ? "#10b981" : "#94a3b8"}
              className={livePitch > 0 ? "pulse-animation" : ""}
            />
            <div>
              <div
                style={{
                  fontSize: "12px",
                  color: "#94a3b8",
                  fontWeight: "bold",
                  marginBottom: "4px",
                }}
              >
                LISTENING FOR PITCH...
              </div>
              <div
                style={{
                  fontSize: "20px",
                  fontWeight: "bold",
                  color: livePitch > 0 ? "#10b981" : "#fff",
                }}
              >
                {livePitch > 0 ? `${livePitch} Hz` : "--- Hz"}
              </div>
            </div>
          </div>
        )}

        {/* Connection Metrics Overlay */}
        <div
          style={{
            position: "absolute",
            top: "24px",
            right: "24px",
            zIndex: 10,
            background: "rgba(15, 23, 42, 0.8)",
            padding: "8px 12px",
            borderRadius: "8px",
            border: "1px solid #334155",
            fontSize: "11px",
            color: "#94a3b8",
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: "4px",
            backdropFilter: "blur(4px)",
            fontFamily: "monospace",
          }}
        >
          <div>
            UPDATE RATE:{" "}
            <span
              style={{ color: "#34d399", fontSize: "13px", fontWeight: "bold" }}
            >
              {metrics.delta}
            </span>{" "}
            ms
          </div>
          <div>
            LATENCY:{" "}
            <span
              style={{ color: "#60a5fa", fontSize: "13px", fontWeight: "bold" }}
            >
              {metrics.ping}
            </span>{" "}
            ms
          </div>
        </div>

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
