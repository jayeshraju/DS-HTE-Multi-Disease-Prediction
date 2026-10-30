import React from "react";
import { DatasetInfo, RunManifest } from "../types";
import { Activity, ShieldAlert, Cpu, Award } from "lucide-react";

interface OverviewViewProps {
  datasets: Record<string, DatasetInfo>;
  manifest: RunManifest | null;
  onSelectCohort: (key: string) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  datasets,
  manifest,
  onSelectCohort
}) => {
  const cohortList = Object.values(datasets);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      {/* Hero Banner */}
      <div
        className="glass-card"
        style={{
          background: "linear-gradient(135deg, rgba(2, 132, 199, 0.2), rgba(99, 102, 241, 0.2))",
          border: "1px solid rgba(56, 189, 248, 0.3)",
          padding: "28px"
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ maxWidth: "800px" }}>
            <span className="badge badge-info" style={{ marginBottom: "12px" }}>
              RESEARCH REPRODUCIBILITY & VALIDATION BENCHMARK
            </span>
            <h2 style={{ fontSize: "1.75rem", fontWeight: 800, marginBottom: "8px", letterSpacing: "-0.02em" }}>
              Adaptive Explainable Multi-Disease Prediction Framework (DS-HTE)
            </h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
            {manifest && (
              <span className="badge badge-success">
                Single run_id: {manifest.run_id}
              </span>
            )}
            <span className="badge badge-info">
              50 Outer Estimates / Cohort
            </span>
          </div>
        </div>
      </div>

      {/* Cohorts Grid */}
      <div>
        <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
          <Activity size={20} color="var(--accent-primary)" />
          Clinical Disease Cohorts
        </h3>
        <div className="grid-5">
          {cohortList.map((c) => (
            <div
              key={c.key}
              className="glass-card"
              style={{ cursor: "pointer", display: "flex", flexDirection: "column", gap: "12px" }}
              onClick={() => onSelectCohort(c.key)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 700, fontSize: "1.05rem", color: "var(--text-primary)" }}>{c.name}</span>
                <span className="badge badge-info" style={{ fontSize: "0.7rem" }}>{c.key.toUpperCase()}</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                <div>Samples: <strong style={{ color: "var(--text-primary)" }}>{c.n_samples}</strong></div>
                <div>Features: <strong style={{ color: "var(--text-primary)" }}>{c.n_features}</strong></div>
                <div>Prevalence: <strong style={{ color: "var(--accent-primary)" }}>{(c.prevalence * 100).toFixed(1)}%</strong></div>
                <div>Baseline Rate: <strong style={{ color: "var(--text-muted)" }}>{((c.majority_baseline_rate ?? (1 - c.prevalence)) * 100).toFixed(1)}%</strong></div>
              </div>
              <button className="btn btn-secondary" style={{ width: "100%", fontSize: "0.75rem", padding: "6px" }}>
                Explore Disease
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture & Methodology Pipeline */}
      <div className="glass-card">
        <h3 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "16px", display: "flex", alignItems: "center", gap: "8px" }}>
          <Cpu size={20} color="var(--accent-primary)" />
          End-to-End Architectural Pipeline
        </h3>
        <div className="grid-4" style={{ gap: "16px" }}>
          <div style={{ background: "var(--bg-tertiary)", padding: "16px", borderRadius: "8px" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-primary)" }}>PHASE 1</span>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: "6px 0" }}>Data Ingestion & Split</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Stratified 20% holdout (previously inspected, secondary reference). Primary evaluation: 50-fold nested repeated CV.
            </p>
          </div>
          <div style={{ background: "var(--bg-tertiary)", padding: "16px", borderRadius: "8px" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-primary)" }}>PHASE 2</span>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: "6px 0" }}>Adaptive Feature Optimization</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Mutual information + Random Forest importance + Permutation testing strictly fit inside each outer training fold.
            </p>
          </div>
          <div style={{ background: "var(--bg-tertiary)", padding: "16px", borderRadius: "8px" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-primary)" }}>PHASE 3</span>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: "6px 0" }}>Hybrid Ensemble (DS-HTE)</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Tabular Transformer fine-tuning alongside XGBoost, LightGBM, and CatBoost combined via constrained SLSQP stacking.
            </p>
          </div>
          <div style={{ background: "var(--bg-tertiary)", padding: "16px", borderRadius: "8px" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-primary)" }}>PHASE 4</span>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: "6px 0" }}>Calibration & Threshold</h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
              Probability calibration on un-resampled inner predictions; optimal decision cutoff via Youden's J statistic.
            </p>
          </div>
        </div>
      </div>

      {/* Integrity & Compliance Safeguards */}
      <div className="grid-2">
        <div className="glass-card">
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
            <ShieldAlert size={18} color="var(--warning)" />
            Audited Biological Validity
          </h4>
          <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "20px" }}>
            <li>Zero biological typos: WBCD feature #12 preserves true <code>smoothness_se</code> (0.006398).</li>
            <li>Zero data leakage: Identical holdout set untouched during all hyperparameter selections.</li>
            <li>Duplicate clusters audited: 26 duplicates in ILPD documented in <code>duplicates_ilpd.md</code>.</li>
            <li>Subgroup error profiling: Subgroups with N &lt; 30 suppressed to prevent spurious bias claims.</li>
          </ul>
        </div>
        <div className="glass-card">
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "12px", display: "flex", alignItems: "center", gap: "8px" }}>
            <Award size={18} color="var(--success)" />
            Verified Cross-Table Invariants
          </h4>
          <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "20px" }}>
            <li>Table IV Full DS-HTE metrics strictly identical to Table II DS-HTE metrics.</li>
            <li>Table I CV metrics match Table II across all 5 disease cohorts.</li>
            <li>Bootstrap confidence intervals verify <code>lower &le; mean &le; upper</code> across all 11 metrics.</li>
            <li>Statistical tests include Holm-Bonferroni and Benjamini-Hochberg FDR adjustments.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
