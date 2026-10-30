import React, { useState, useEffect } from "react";
import { DatasetInfo, ConfusionData } from "../types";
import { fetchConfusion, fetchCurves } from "../services/api";
import { ConfusionMatrixView, DcaChart } from "../components/Charts";
import { Sliders, Sparkles } from "lucide-react";

interface ThresholdExplorerViewProps {
  datasets: Record<string, DatasetInfo>;
  selectedCohort: string;
  onSelectCohort: (key: string) => void;
}

export const ThresholdExplorerView: React.FC<ThresholdExplorerViewProps> = ({
  datasets,
  selectedCohort,
  onSelectCohort
}) => {
  const [threshold, setThreshold] = useState<number>(0.5);
  const [confusionData, setConfusionData] = useState<ConfusionData | null>(null);
  const [dcaPoints, setDcaPoints] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchConfusion(selectedCohort, "DS-HTE", threshold)
      .then((data) => {
        if (mounted) {
          setConfusionData(data);
          if (threshold === 0.5 && data.default_threshold) {
            setThreshold(data.default_threshold);
          }
        }
      })
      .catch((err) => console.error(err))
      .finally(() => {
        if (mounted) setLoading(false);
      });

    fetchCurves(selectedCohort, "DS-HTE", "dca")
      .then((pts) => {
        if (mounted && Array.isArray(pts)) {
          setDcaPoints(pts);
        }
      })
      .catch((err) => console.error(err));

    return () => {
      mounted = false;
    };
  }, [selectedCohort, threshold]);

  const cohortList = Object.values(datasets);

  const handleSetYoudenOptimal = () => {
    if (confusionData?.default_threshold) {
      setThreshold(confusionData.default_threshold);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Control Banner */}
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
                {c.name} ({c.n_samples} samples)
              </option>
            ))}
          </select>
        </div>

        <button className="btn btn-primary" onClick={handleSetYoudenOptimal}>
          <Sparkles size={16} />
          Snap to Youden-J Optimal Threshold ({confusionData?.default_threshold ?? 0.5})
        </button>
      </div>

      {/* Interactive Slider Card */}
      <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h4 style={{ fontSize: "1.1rem", fontWeight: 700, display: "flex", alignItems: "center", gap: "8px" }}>
              <Sliders size={20} color="var(--accent-primary)" />
              Clinical Decision Threshold (Cutoff): <span style={{ color: "var(--accent-primary)", fontFamily: "var(--font-mono)" }}>{threshold.toFixed(2)}</span>
            </h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Drag the slider to adjust patient risk sensitivity. Lower thresholds catch more cases (high sensitivity) at the cost of more false alarms (lower specificity).
            </p>
          </div>
          <span className="badge badge-info" style={{ fontSize: "0.85rem" }}>
            Current Threshold: {threshold.toFixed(2)}
          </span>
        </div>

        <div className="slider-container">
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="slider-input"
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            <span>0.05 (Aggressive Screening / High Sensitivity)</span>
            <span>0.50 (Default Neutral)</span>
            <span>0.95 (Conservative Confirmatory / High Specificity)</span>
          </div>
        </div>
      </div>

      {/* Matrix and DCA Grid */}
      <div className="grid-2">
        <div className="glass-card">
          <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "16px" }}>
            Dynamic Confusion Matrix (at threshold = {threshold.toFixed(2)})
          </h4>
          {confusionData ? (
            <ConfusionMatrixView data={confusionData} />
          ) : (
            <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>Loading confusion matrix...</div>
          )}
        </div>

        <div className="glass-card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <h4 style={{ fontSize: "1rem", fontWeight: 700 }}>Decision Curve Analysis (Net Clinical Benefit)</h4>
            {confusionData && (
              <span className="badge badge-success">
                Net Benefit: {confusionData.net_benefit.toFixed(3)}
              </span>
            )}
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "12px" }}>
            Quantifies the clinical utility of making treatment decisions using DS-HTE versus treating all or treating none.
          </p>
          {dcaPoints.length > 0 ? (
            <DcaChart points={dcaPoints} />
          ) : (
            <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>Loading DCA curve...</div>
          )}
        </div>
      </div>
    </div>
  );
};
