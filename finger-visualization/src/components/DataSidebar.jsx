export default function DataSidebar({ data }) {
  const getPercentage = (val) => Math.min(100, Math.max(0, (val / 3000) * 100));

  return (
    <div className="data-sidebar">
      <section className="card">
        <h3 style={{ color: "#60a5fa", fontSize: "12px", marginBottom: "4px" }}>
          HARDWARE DATA
        </h3>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <div>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              TOE CONTROLLERS ( M1 / M2 )
            </div>

            {/* Motor 1 Bar */}
            <div style={{ marginBottom: "8px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "10px",
                  marginBottom: "4px",
                  fontWeight: "bold",
                }}
              >
                <span>M1 Force</span>
                <span>
                  {Math.round(getPercentage(data.sensors.toe_fsr?.[0] || 0))}%
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "#1e293b",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${getPercentage(data.sensors.toe_fsr?.[0] || 0)}%`,
                    background: "#60a5fa",
                    transition: "width 0.1s ease-out",
                  }}
                />
              </div>
            </div>

            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "10px",
                  marginBottom: "4px",
                  fontWeight: "bold",
                }}
              >
                <span>M2 Force</span>
                <span>
                  {Math.round(getPercentage(data.sensors.toe_fsr?.[1] || 0))}%
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "#1e293b",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${getPercentage(data.sensors.toe_fsr?.[1] || 0)}%`,
                    background: "#60a5fa",
                    transition: "width 0.1s ease-out",
                  }}
                />
              </div>
            </div>
          </div>

          <div>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                marginBottom: "8px",
                fontWeight: "bold",
              }}
            >
              FINGER SENSORS ( Base / Mid / Tip )
            </div>

            <div style={{ marginBottom: "8px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "10px",
                  marginBottom: "4px",
                  fontWeight: "bold",
                }}
              >
                <span>Base Force</span>
                <span>
                  {Math.round(getPercentage(data.sensors.fsr?.[0] || 0))}%
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "#1e293b",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${getPercentage(data.sensors.fsr?.[0] || 0)}%`,
                    background: "#60a5fa",
                    transition: "width 0.1s ease-out",
                  }}
                />
              </div>
            </div>

            <div style={{ marginBottom: "8px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "10px",
                  marginBottom: "4px",
                  fontWeight: "bold",
                }}
              >
                <span>Mid Force</span>
                <span>
                  {Math.round(
                    getPercentage(
                      Math.floor(
                        (data.sensors.fsr?.[0] + data.sensors.fsr?.[2]) / 3,
                      ) || 0,
                    ),
                  )}
                  %
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "#1e293b",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${getPercentage(data.sensors.fsr?.[1] || 0)}%`,
                    background: "#60a5fa",
                    transition: "width 0.1s ease-out",
                  }}
                />
              </div>
            </div>

            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "10px",
                  marginBottom: "4px",
                  fontWeight: "bold",
                }}
              >
                <span>Tip Force</span>
                <span>
                  {Math.round(getPercentage(data.sensors.fsr?.[2] || 0))}%
                </span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "#1e293b",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${getPercentage(data.sensors.fsr?.[2] || 0)}%`,
                    background: "#60a5fa",
                    transition: "width 0.1s ease-out",
                  }}
                />
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
          }}
        >
          <div
            style={{
              fontSize: "20px",
              fontWeight: "bold",
              letterSpacing: "2px",
              padding: "8px 16px",
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
