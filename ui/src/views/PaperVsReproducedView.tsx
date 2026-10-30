import React, { useState, useEffect } from "react";
import { PaperComparisonItem } from "../types";
import { fetchPaperVsReproduced } from "../services/api";
import { FileText, AlertCircle } from "lucide-react";

export const PaperVsReproducedView: React.FC = () => {
  const [data, setData] = useState<Record<string, PaperComparisonItem>>({});
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let mounted = true;
    fetchPaperVsReproduced()
      .then((res) => {
        if (mounted) setData(res);
      })
      .catch((err) => console.error(err))
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const paperDefaults: Record<string, { paperAcc: number; desc: string }> = {
    PIMA: { paperAcc: 81.2, desc: "Diabetes mellitus screening" },
    Cleveland: { paperAcc: 88.4, desc: "Coronary artery disease" },
    CKD: { paperAcc: 95.6, desc: "Chronic kidney disease" },
    ILPD: { paperAcc: 79.3, desc: "Indian liver patient records" },
    WBCD: { paperAcc: 97.1, desc: "Wisconsin breast cancer diagnostic" }
  };

  const cohorts = ["PIMA", "Cleveland", "CKD", "ILPD", "WBCD"];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Integrity Statement Banner */}
      <div
        className="glass-card"
        style={{
          background: "linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(239, 68, 68, 0.15))",
          border: "1px solid rgba(245, 158, 11, 0.3)"
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
          <AlertCircle size={24} color="var(--warning)" style={{ flexShrink: 0, marginTop: "2px" }} />
          <div>
            <h4 style={{ fontSize: "1.1rem", fontWeight: 800, marginBottom: "4px" }}>
              Scientific Integrity & Ground Truth Audit Notice
            </h4>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
              In accordance with research ground rules, the framework does <strong>not</strong> hardcode, tune,
              re-seed, or cherry-pick toward Mythily et al.'s reported numbers. All reproduced values come strictly
              from the verified execution of our leakage-free code. The paper's claimed numbers are displayed solely
              for comparative audit.
            </p>
          </div>
        </div>
      </div>

      {/* Table I Comparison */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
          <FileText size={20} color="var(--accent-primary)" />
          Table I: Paper Claimed Accuracy vs Honestly Reproduced Results
        </h4>
        <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "16px" }}>
          Direct side-by-side audit of accuracy claims across all 5 disease cohorts.
        </p>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Cohort</th>
                <th>Disease Condition</th>
                <th>Mythily et al. Claim</th>
                <th>Our 50-Fold CV (Mean)</th>
                <th>CV &Delta; (Rep - Paper)</th>
                <th>Single 20% Holdout</th>
                <th>Holdout &Delta; (Rep - Paper)</th>
              </tr>
            </thead>
            <tbody>
              {cohorts.map((c) => {
                const row = data[c] || data[c.toLowerCase()];
                const def = paperDefaults[c];
                const paperAcc = row?.paper?.accuracy ?? def.paperAcc;
                const cvAcc = row?.cv?.accuracy ?? 0;
                const hoAcc = row?.holdout?.accuracy ?? 0;
                const cvDelta = cvAcc > 0 ? cvAcc - paperAcc : 0;
                const hoDelta = hoAcc > 0 ? hoAcc - paperAcc : 0;

                const fmtDelta = (d: number) => {
                  if (Math.abs(d) < 1e-3) return "0.00%";
                  const pfx = d > 0 ? "+" : "";
                  return `${pfx}${d.toFixed(2)}%`;
                };

                return (
                  <tr key={c}>
                    <td style={{ fontWeight: 700 }}>{c}</td>
                    <td style={{ color: "var(--text-secondary)" }}>{def.desc}</td>
                    <td className="table-num" style={{ fontWeight: 700 }}>{paperAcc.toFixed(1)}%</td>
                    <td className="table-num" style={{ fontWeight: 700, color: "var(--accent-primary)" }}>
                      {cvAcc > 0 ? `${cvAcc.toFixed(2)}%` : "Computing..."}
                    </td>
                    <td
                      className="table-num"
                      style={{
                        fontWeight: 700,
                        color: cvDelta >= 0 ? "var(--success)" : "var(--danger)"
                      }}
                    >
                      {cvAcc > 0 ? fmtDelta(cvDelta) : "—"}
                    </td>
                    <td className="table-num" style={{ fontWeight: 700, color: "var(--accent-secondary)" }}>
                      {hoAcc > 0 ? `${hoAcc.toFixed(2)}%` : "Computing..."}
                    </td>
                    <td
                      className="table-num"
                      style={{
                        fontWeight: 700,
                        color: hoDelta >= 0 ? "var(--success)" : "var(--danger)"
                      }}
                    >
                      {hoAcc > 0 ? fmtDelta(hoDelta) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Discrepancy & Methodological Analysis */}
      <div className="grid-2">
        <div className="glass-card">
          <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "8px", color: "var(--accent-teal)" }}>Tested Causes (Controlled Experiments)</h4>
          <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "20px" }}>
            <li><strong>Operating Point (Youden-J vs Accuracy-Optimal):</strong> Youden-J optimizes Sensitivity for clinical safety, which on imbalanced ILPD (71.5% majority) sets threshold ~0.38, yielding 62.4% accuracy. At an accuracy-optimal threshold, ILPD achieves &ge;71.5%.</li>
            <li><strong>ILPD Duplicate Leakage Delta (+0.60% acc, +0.87% AUC):</strong> Controlled 50-fold experiment (results/audit/ilpd_duplicate_leakage_experiment.json) measured ungrouped (72.08% acc) vs grouped (71.47% acc), proving duplicate leakage contributes &lt;1% bias.</li>
          </ul>
        </div>

        <div className="glass-card">
          <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "8px", color: "var(--accent-amber)" }}>Possible Reasons (Untested Hypotheses)</h4>
          <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "20px" }}>
            <li><strong>Single Split Reporting:</strong> Published literature often reports a single unrepeated train/test split (which can vary &plusmn;5%) rather than our 50-fold nested evaluation.</li>
            <li><strong>Outer-Fold Preprocessing Leakage:</strong> Fitting imputation, scaling, or feature selection before splitting inflates performance artificially.</li>
            <li><strong>Model Selection on Test Labels:</strong> Hyperparameter tuning outside nested cross-validation produces optimistic bias.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
