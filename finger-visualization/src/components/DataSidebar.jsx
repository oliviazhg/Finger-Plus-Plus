export default function DataSidebar({ data }) {
  return (
    <div className="data-sidebar">
      <section className="card">
        <h3
          style={{ color: "#60a5fa", fontSize: "12px", marginBottom: "15px" }}
        >
          HARDWARE DATA
        </h3>

        <div style={{ display: "flex", flexDirection: "column", gap: "15px" }}>
          <div>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                marginBottom: "4px",
                fontWeight: "bold",
              }}
            >
              MOTORS ( Base / Tendon )
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.motors[0]}
                </div>
              </div>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.motors[1]}
                </div>
              </div>
            </div>
          </div>

          <div>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                marginBottom: "4px",
                fontWeight: "bold",
              }}
            >
              FSR FORCE ( Base / Mid / Tip )
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.fsr[0]}
                </div>
              </div>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.fsr[1]}
                </div>
              </div>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.fsr[2]}
                </div>
              </div>
            </div>
          </div>

          <div>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                marginBottom: "4px",
                fontWeight: "bold",
              }}
            >
              IMU ANGLES ( Base / Mid / Tip )
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.imu[0]}°
                </div>
              </div>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.imu[1]}°
                </div>
              </div>
              <div className="sensor-box" style={{ flex: 1, padding: "8px" }}>
                <div style={{ fontSize: "14px", fontWeight: "bold" }}>
                  {data.sensors.imu[2]}°
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="card">
        <h3 style={{ color: "#60a5fa", fontSize: "12px" }}>MYO BAND STATUS</h3>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            padding: "10px 0",
          }}
        >
          <div
            style={{
              fontSize: "24px",
              fontWeight: "bold",
              letterSpacing: "2px",
              padding: "10px 20px",
              borderRadius: "8px",
              backgroundColor: "rgba(96, 165, 250, 0.2)",
              border: `1px solid #60a5fa`,
              minWidth: "120px",
              textAlign: "center",
            }}
          >
            {data.myo.state || "UNKNOWN"}
          </div>
        </div>
      </section>

      <section
        className="card"
        style={{ flex: 1, display: "flex", flexDirection: "column" }}
      >
        <h3 style={{ color: "#60a5fa", fontSize: "12px" }}>DATA LOG</h3>
        <div className="log-stream">
          {data.logs.map((log, i) => (
            <div
              key={i}
              style={{
                marginBottom: "4px",
                borderLeft: "1px solid #334155",
                paddingLeft: "8px",
              }}
            >
              {log}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
