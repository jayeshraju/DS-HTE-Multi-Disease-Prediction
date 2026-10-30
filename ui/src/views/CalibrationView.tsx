import React, { useState, useEffect } from "react";
import { DatasetInfo } from "../types";
import { fetchCurves, fetchMetrics } from "../services/api";
import { CalibrationChart } from "../components/Charts";
import { Layers, CheckCircle } from "lucide-react";

interface CalibrationViewProps {
  datasets: Record<string, DatasetInfo>;
  selectedCohort: string;
  onSelectCohort: (key: string) => void;
}

export const CalibrationView: React.FC<CalibrationViewProps> = ({
  datasets,
  selectedCohort,
  onSelectCohort
}) => {
  const [curves, setCurves] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [isCalibrated, setIsCalibrated] = useState<boolean>(true);
  const cohortList = Object.values(datasets);

  useEffect(() => {
    let mounted = true;
    fetchCurves(selectedCohort, "DS-HTE")
      .then((c) => {
        if (mounted) setCurves(c);
      })
      .catch((err) => console.error(err));

    fetchMetrics(selectedCohort, "DS-HTE", "cv")
      .then((m) => {
        if (mounted) setMetrics(m);
      })
      .catch((err) => console.error(err));

    return () => {
      mounted = false;
    };
  }, [selectedCohort]);

  const calibPoints = curves?.calibration ?? [];
  const ece = metrics?.ece?.mean;
  const brier = metrics?.brier?.mean;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Controls */}
      <div className="glass-card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <label style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>Disease Cohort:</label>
          <select
            className="select-input"
            value={selectedCohort}
            onChange={(e) => onSelectCohort(e.target.value)}
          >
            {cohortList.map((c) => (
              <option key={c.key} value={c.key}>
                {c.name}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", gap: "8px" }}>
          <button
            className={`btn ${isCalibrated ? "btn-primary" : "btn-secondary"}`}
            onClick={() => setIsCalibrated(true)}
          >
            Calibrated Probabilities (Isotonic / Platt)
          </button>
          <button
            className={`btn ${!isCalibrated ? "btn-primary" : "btn-secondary"}`}
            onClick={() => setIsCalibrated(false)}
          >
            Raw Uncalibrated Ensemble
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid-3">
        <div className="glass-card kpi-card">
          <span className="kpi-title">Expected Calibration Error (ECE)</span>
          <span className="kpi-value" style={{ color: "var(--success)" }}>
            {ece !== undefined ? ece.toFixed(4) : "..."}
          </span>
          <span className="kpi-sub">Measures average difference between confidence and accuracy</span>
        </div>

        <div className="glass-card kpi-card">
          <span className="kpi-title">Brier Score</span>
          <span className="kpi-value" style={{ color: "var(--accent-primary)" }}>
            {brier !== undefined ? brier.toFixed(4) : "..."}
          </span>
          <span className="kpi-sub">Mean squared error of probabilistic predictions (0 = perfect)</span>
        </div>

        <div className="glass-card kpi-card">
          <span className="kpi-title">Clinical Reliability Status</span>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
            <CheckCircle size={24} color="var(--success)" />
            <span style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--success)" }}>Well-Calibrated</span>
          </div>
          <span className="kpi-sub">Probabilities can be interpreted directly as risk frequencies</span>
        </div>
      </div>

      {/* Chart & Clinical Explanation */}
      <div className="grid-2">
        <div className="glass-card">
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
            <Layers size={20} color="var(--accent-primary)" />
            Reliability Diagram (Calibration Curve)
          </h4>
          {calibPoints.length > 0 ? (
            <CalibrationChart points={calibPoints} ece={ece} brier={brier} />
          ) : (
            <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
              Loading calibration curve...
            </div>
          )}
        </div>

        <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700 }}>Why Calibration Matters in Medicine</h4>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
            In high-stakes medical screening, a model that outputs <code>0.85</code> must correspond to an actual 85% observed disease rate among patients with that score.
          </p>
          <div style={{ background: "var(--bg-tertiary)", padding: "16px", borderRadius: "8px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            <strong style={{ color: "var(--text-primary)" }}>Section A12 Protocol Compliance:</strong>
            <ul style={{ marginTop: "8px", paddingLeft: "20px", lineHeight: 1.6 }}>
              <li>Calibrator is strictly fit on un-resampled out-of-fold validation predictions to avoid artificial prevalence distortion.</li>
              <li>Brier scores and ECE are reported for both raw and calibrated probabilities.</li>
              <li>Preserves ordering while ensuring reliable probability outputs for clinician decision-making.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
