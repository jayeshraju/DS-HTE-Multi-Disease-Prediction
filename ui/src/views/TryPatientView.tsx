import React, { useState } from "react";
import { DatasetInfo } from "../types";
import { predictPatient, explainPatient } from "../services/api";
import { WaterfallChart } from "../components/Charts";
import { UserCheck, FileDown, AlertTriangle, CheckCircle2 } from "lucide-react";

interface TryPatientViewProps {
  datasets: Record<string, DatasetInfo>;
  selectedCohort: string;
  onSelectCohort: (key: string) => void;
}

export const TryPatientView: React.FC<TryPatientViewProps> = ({
  datasets,
  selectedCohort,
  onSelectCohort
}) => {
  const cohortList = Object.values(datasets);
  const activeDataset = datasets[selectedCohort];

  // Default patient values per cohort
  const defaultPresets: Record<string, Record<string, number>> = {
    pima: { Glucose: 165, BMI: 34.2, Age: 48, Pregnancies: 3, BloodPressure: 82, Insulin: 180, SkinThickness: 28, DiabetesPedigreeFunction: 0.65 },
    cleveland: { age: 58, sex: 1, cp: 3, trestbps: 145, chol: 260, fbs: 1, restecg: 1, thalach: 125, exang: 1, oldpeak: 2.4, slope: 2, ca: 2, thal: 3 },
    ckd: { age: 62, bp: 90, bgr: 190, bu: 65, sc: 3.2, sod: 132, pot: 5.1, hemo: 9.5, pcv: 30, wc: 9800, rc: 3.4 },
    ilpd: { age: 52, gender: 1, tot_bilirubin: 2.8, direct_bilirubin: 1.4, tot_proteins: 6.2, albumin: 2.8, ag_ratio: 0.8, sgpt: 55, sgot: 72, alkphos: 290 },
    wbcd: { radius_mean: 18.2, texture_mean: 21.5, perimeter_mean: 122.0, area_mean: 1040.0, smoothness_mean: 0.108, compactness_mean: 0.165, concavity_mean: 0.185, concave_points_mean: 0.098 }
  };

  const [features, setFeatures] = useState<Record<string, number>>(
    defaultPresets[selectedCohort] || { Glucose: 140, BMI: 30, Age: 50 }
  );
  const [patientId, setPatientId] = useState<string>("PT-DEMO-01");
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [attributions, setAttributions] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleCohortChange = (cohortKey: string) => {
    onSelectCohort(cohortKey);
    setFeatures(defaultPresets[cohortKey] || {});
    setPredictionResult(null);
    setAttributions(null);
  };

  const handleFeatureChange = (name: string, val: number) => {
    setFeatures((prev) => ({ ...prev, [name]: val }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const pred = await predictPatient(selectedCohort, patientId, features);
      setPredictionResult(pred);
      const exp = await explainPatient(selectedCohort, patientId, features);
      setAttributions(exp.shap_attributions);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = async () => {
    const res = await fetch(`/api/report/${selectedCohort}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_id: patientId, features })
    });
    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Clinical_Report_${patientId}_${selectedCohort}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Cohort Selector */}
      <div className="glass-card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <label style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>Disease Assessment Model:</label>
          <select
            className="select-input"
            value={selectedCohort}
            onChange={(e) => handleCohortChange(e.target.value)}
          >
            {cohortList.map((c) => (
              <option key={c.key} value={c.key}>
                {c.name} Assessment
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>Patient ID:</label>
          <input
            type="text"
            className="text-input"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            style={{ width: "140px" }}
          />
        </div>
      </div>

      <div className="grid-2">
        {/* Input Form Card */}
        <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700, display: "flex", alignItems: "center", gap: "8px" }}>
            <UserCheck size={20} color="var(--accent-primary)" />
            Clinical Biomarkers & Patient Parameters
          </h4>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", maxHeight: "420px", overflowY: "auto", paddingRight: "8px" }}>
            {Object.entries(features).map(([name, val]) => (
              <div key={name} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)" }}>{name}</label>
                <input
                  type="number"
                  step="any"
                  className="text-input"
                  value={val}
                  onChange={(e) => handleFeatureChange(name, parseFloat(e.target.value) || 0)}
                />
              </div>
            ))}
          </div>

          <button className="btn btn-primary" onClick={handlePredict} disabled={loading} style={{ marginTop: "8px" }}>
            {loading ? "Running DS-HTE Inference..." : "Evaluate Patient Risk Profile"}
          </button>
        </div>

        {/* Prediction Results & Explanation Card */}
        <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700 }}>Decision Support Risk Output</h4>

          {predictionResult ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div
                style={{
                  padding: "16px",
                  borderRadius: "8px",
                  background: predictionResult.prediction === 1 ? "rgba(239, 68, 68, 0.15)" : "rgba(16, 185, 129, 0.15)",
                  border: `1px solid ${predictionResult.prediction === 1 ? "var(--danger)" : "var(--success)"}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  {predictionResult.prediction === 1 ? (
                    <AlertTriangle size={32} color="var(--danger)" />
                  ) : (
                    <CheckCircle2 size={32} color="var(--success)" />
                  )}
                  <div>
                    <h5 style={{ fontSize: "1.1rem", fontWeight: 800 }}>
                      {predictionResult.prediction === 1 ? "HIGH PATHOLOGICAL RISK" : "NORMAL / LOW RISK PROFILE"}
                    </h5>
                    <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                      Calibrated Threshold: {predictionResult.calibrated_threshold}
                    </p>
                  </div>
                </div>

                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "1.75rem", fontWeight: 800, fontFamily: "var(--font-mono)" }}>
                    {(predictionResult.risk_probability * 100).toFixed(1)}%
                  </div>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Risk Probability</span>
                </div>
              </div>

              {/* Local Feature Explanations */}
              <div>
                <h5 style={{ fontSize: "0.9rem", fontWeight: 700, marginBottom: "8px" }}>
                  Individual Risk Attributions (SHAP Consensus)
                </h5>
                {attributions && (
                  <WaterfallChart attributions={attributions} finalProb={predictionResult.risk_probability} />
                )}
              </div>

              {/* Download Report Button */}
              <button className="btn btn-secondary" onClick={handleDownloadPdf} style={{ display: "flex", justifyContent: "center" }}>
                <FileDown size={16} />
                Download Signed Physician PDF Report
              </button>
            </div>
          ) : (
            <div style={{ padding: "60px 20px", textAlign: "center", color: "var(--text-muted)" }}>
              Adjust patient parameters and click "Evaluate Patient Risk Profile" to inspect calibrated risk and feature attributions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
