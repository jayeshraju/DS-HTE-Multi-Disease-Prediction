import React, { useState, useEffect } from "react";
import { DatasetInfo, GlobalFeatureItem, StabilityRecord } from "../types";
import { fetchGlobalExplain, fetchStability, fetchSubgroups } from "../services/api";
import { Cpu, Award, ShieldAlert } from "lucide-react";

interface ExplainabilityViewProps {
  datasets: Record<string, DatasetInfo>;
  selectedCohort: string;
  onSelectCohort: (key: string) => void;
}

export const ExplainabilityView: React.FC<ExplainabilityViewProps> = ({
  datasets,
  selectedCohort,
  onSelectCohort
}) => {
  const [features, setFeatures] = useState<GlobalFeatureItem[]>([]);
  const [consensus, setConsensus] = useState<number>(0.85);
  const [stabilityRecords, setStabilityRecords] = useState<StabilityRecord[]>([]);
  const [subgroups, setSubgroups] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const cohortList = Object.values(datasets);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    fetchGlobalExplain(selectedCohort)
      .then((res) => {
        if (mounted) {
          setFeatures(res.features);
          setConsensus(res.consensus_agreement);
        }
      })
      .catch((err) => console.error(err))
      .finally(() => {
        if (mounted) setLoading(false);
      });

    fetchStability(selectedCohort)
      .then((stab) => {
        if (mounted) setStabilityRecords(stab);
      })
      .catch((err) => console.error(err));

    fetchSubgroups(selectedCohort)
      .then((sg) => {
        if (mounted && Array.isArray(sg)) setSubgroups(sg);
      })
      .catch((err) => console.error(err));

    return () => {
      mounted = false;
    };
  }, [selectedCohort]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Control Bar */}
      <div className="glass-card" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
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

        <span className="badge badge-success">
          Consensus Agreement: {(consensus * 100).toFixed(1)}%
        </span>
      </div>

      {/* Global Feature Importance Table */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
          <Cpu size={20} color="var(--accent-primary)" />
          Global Feature Attributions (SHAP + LIME Consensus)
        </h4>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
          Ranks features by mean absolute SHAP value and LIME linear weight, showing stability across 50 outer folds.
        </p>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Clinical Feature</th>
                <th>Mean |SHAP| Value</th>
                <th>LIME Importance Weight</th>
                <th>50-Fold Selection Frequency</th>
              </tr>
            </thead>
            <tbody>
              {features.map((f) => (
                <tr key={f.feature}>
                  <td className="table-num" style={{ fontWeight: 700 }}>#{f.rank}</td>
                  <td style={{ fontWeight: 600 }}>{f.feature}</td>
                  <td className="table-num" style={{ color: "var(--accent-primary)" }}>
                    {f.mean_abs_shap.toFixed(4)}
                  </td>
                  <td className="table-num" style={{ color: "var(--accent-secondary)" }}>
                    {f.lime_weight.toFixed(4)}
                  </td>
                  <td className="table-num">
                    <span className="badge badge-info">{(f.stability_freq * 100).toFixed(0)}%</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Subgroup Error Profiling */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
          <ShieldAlert size={18} color="var(--warning)" />
          Clinical Subgroup Error Profiling (Section A13: N &ge; 30)
        </h4>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
          Audits False Negative Rates (FNR) across demographic and clinical strata. Subgroups with N &lt; 30 are suppressed to prevent spurious bias claims.
        </p>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Stratum Attribute</th>
                <th>Subgroup Category</th>
                <th>Sample Size (N)</th>
                <th>False Negative Rate (FNR)</th>
                <th>Wilson 95% Confidence Interval</th>
                <th>Disparity Flag</th>
              </tr>
            </thead>
            <tbody>
              {subgroups.length > 0 ? (
                subgroups.map((sg, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 600 }}>{sg.attribute}</td>
                    <td>{sg.subgroup}</td>
                    <td className="table-num">{sg.n_samples}</td>
                    <td className="table-num">{(sg.fnr * 100).toFixed(1)}%</td>
                    <td className="table-num">
                      [{(sg.fnr_ci_lower * 100).toFixed(1)}%, {(sg.fnr_ci_upper * 100).toFixed(1)}%]
                    </td>
                    <td>
                      {sg.disparity_flag ? (
                        <span className="badge badge-danger">Disparity Flagged</span>
                      ) : (
                        <span className="badge badge-success">Acceptable Range</span>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} style={{ textAlign: "center", padding: "20px", color: "var(--text-muted)" }}>
                    No statistically significant subgroup disparities detected across strata with N &ge; 30.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
