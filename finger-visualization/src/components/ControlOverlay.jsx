import React, { useState } from "react";

export default function ControlOverlay({ activeKeys, onSetPosition }) {
  const [m1Pos, setM1Pos] = useState("");
  const [m2Pos, setM2Pos] = useState("");

  const handleM1Submit = (e) => {
    e.preventDefault();
    if (m1Pos !== "") onSetPosition(1, m1Pos);
  };

  const handleM2Submit = (e) => {
    e.preventDefault();
    if (m2Pos !== "") onSetPosition(2, m2Pos);
  };

  return (
    <div
      style={{
        position: "absolute",
        bottom: "24px",
        left: "24px",
        zIndex: 10,
        display: "flex",
        gap: "24px",
        alignItems: "flex-end",
        pointerEvents: "none", // Let clicks pass through to the 3D canvas...
      }}
    >
      {/* Arrow Key Visualizer */}
      {/* <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "6px",
        }}
      >
        <div className={`key-box ${activeKeys.up ? "active" : ""}`}>↑</div>
        <div style={{ display: "flex", gap: "6px" }}>
          <div className={`key-box ${activeKeys.left ? "active" : ""}`}>←</div>
          <div className={`key-box ${activeKeys.down ? "active" : ""}`}>↓</div>
          <div className={`key-box ${activeKeys.right ? "active" : ""}`}>→</div>
        </div>
      </div> */}

      <div
        style={{
          pointerEvents: "auto", // catch clicks for the input boxes
          background: "rgba(30, 41, 59, 0.8)",
          padding: "16px",
          borderRadius: "12px",
          border: "1px solid #334155",
          backdropFilter: "blur(4px)",
        }}
      >
        <h4
          style={{
            margin: "0 0 12px 0",
            fontSize: "12px",
            color: "#94a3b8",
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          ABSOLUTE POSITIONING
        </h4>

        <form
          onSubmit={handleM1Submit}
          style={{ display: "flex", gap: "8px", marginBottom: "12px" }}
        >
          <label
            style={{
              fontSize: "12px",
              width: "55px",
              alignContent: "center",
              fontWeight: "bold",
            }}
          >
            M1 (Base)
          </label>
          <input
            type="number"
            value={m1Pos}
            onChange={(e) => setM1Pos(e.target.value)}
            placeholder="4300 to 3000"
            style={{
              background: "#0f172a",
              border: "1px solid #334155",
              color: "#fff",
              padding: "6px 10px",
              borderRadius: "6px",
              width: "110px",
              fontSize: "12px",
            }}
          />
          <button
            type="submit"
            className="btn-primary"
            style={{ padding: "6px 12px", fontSize: "12px" }}
          >
            GO
          </button>
        </form>

        <form onSubmit={handleM2Submit} style={{ display: "flex", gap: "8px" }}>
          <label
            style={{
              fontSize: "12px",
              width: "55px",
              alignContent: "center",
              fontWeight: "bold",
            }}
          >
            M2 (Curl)
          </label>
          <input
            type="number"
            value={m2Pos}
            onChange={(e) => setM2Pos(e.target.value)}
            placeholder="3000 to 6900"
            style={{
              background: "#0f172a",
              border: "1px solid #334155",
              color: "#fff",
              padding: "6px 10px",
              borderRadius: "6px",
              width: "110px",
              fontSize: "12px",
            }}
          />
          <button
            type="submit"
            className="btn-primary"
            style={{ padding: "6px 12px", fontSize: "12px" }}
          >
            GO
          </button>
        </form>
      </div>
    </div>
  );
}
