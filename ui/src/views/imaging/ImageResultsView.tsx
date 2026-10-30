import React, { useState, useEffect } from "react";
import { ShieldCheck, AlertCircle, TrendingUp, CheckCircle, HelpCircle } from "lucide-react";

export const ImageResultsView: React.FC<{
  selectedJobId?: string | null;
}> = ({ selectedJobId }) => {
  const [jobId, setJobId] = useState<string>(selectedJobId || "bench_pneumoniamnist");
  const [data, setData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"metrics" | "curves" | "gradcam">("metrics");

  useEffect(() => {
    if (selectedJobId) {
      setJobId(selectedJobId);
    }
  }, [selectedJobId]);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/image/jobs/${jobId}`)
      .then((res) => {
        if (!res.ok) throw new Error("Job not found");
        return res.json();
      })
      .then((d) => {
        setData(d.metrics || d);
      })
      .catch((e) => console.error(e))
      .finally(() => setLoading(false));
  }, [jobId]);

  if (loading) {
    return <div className="glass-card" style={{ padding: "40px", textAlign: "center" }}>Loading model results...</div>;
  }

  if (!data || !data.metrics) {
    return (
      <div className="glass-card" style={{ padding: "40px", textAlign: "center" }}>
        <h4>No Model Results Found</h4>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Train a model in the Training Jobs page or run the MedMNIST benchmark to view results.
        </p>
      </div>
    );
  }

  const m = data.metrics;
  const deltaMaj = Math.round((m.accuracy - m.majority_rate) * 1000) / 10;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Disclaimer Bar */}
      <div className="glass-card" style={{ borderLeft: "4px solid var(--accent-teal)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-teal)", textTransform: "uppercase" }}>Model Card & Provenance</span>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 700, margin: "4px 0" }}>
            {data.backbone?.toUpperCase()} on {data.dataset || "Clinical Cohort"}
          </h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
            Job ID: <code>{data.job_id}</code> | Classes: {data.classes?.join(", ")} | Samples: {data.n_samples}
          </p>
        </div>
        <div style={{ textAlign: "right", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <span className="badge badge-info" style={{ marginBottom: "4px", display: "inline-block" }}>Temperature Scaled: T={data.temperature}</span>
          <div>Research prototype, not a diagnostic device.</div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid var(--card-border)", paddingBottom: "12px" }}>
        <button
          className={`btn ${activeTab === "metrics" ? "btn-primary" : "btn-secondary"}`}
          onClick={() => setActiveTab("metrics")}
        >
          Clinical Performance & CIs
        </button>
        <button
          className={`btn ${activeTab === "curves" ? "btn-primary" : "btn-secondary"}`}
          onClick={() => setActiveTab("curves")}
        >
          Learning Curves & Calibration
        </button>
        <button
          className={`btn ${activeTab === "gradcam" ? "btn-primary" : "btn-secondary"}`}
          onClick={() => setActiveTab("gradcam")}
        >
          Grad-CAM Saliency Gallery
        </button>
      </div>

      {/* TAB 1: METRICS */}
      {activeTab === "metrics" && (
        <>
          <div className="grid-4">
            <div className="metric-box">
              <span className="metric-label">Accuracy vs Majority Baseline</span>
              <span className="metric-value">{(m.accuracy * 100).toFixed(1)}%</span>
              <span className="metric-sub" style={{ color: deltaMaj >= 0 ? "var(--accent-teal)" : "#ef4444" }}>
                Majority Baseline: {(m.majority_rate * 100).toFixed(1)}% ({deltaMaj >= 0 ? `+${deltaMaj}%` : `${deltaMaj}%`})
              </span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Balanced Accuracy</span>
              <span className="metric-value">{(m.balanced_accuracy * 100).toFixed(1)}%</span>
              <span className="metric-sub">Mean Sensitivity Across Classes</span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Discrimination (ROC-AUC)</span>
              <span className="metric-value">{m.roc_auc?.toFixed(4)}</span>
              <span className="metric-sub">PR-AUC: {m.pr_auc?.toFixed(4)}</span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Calibration (Brier & ECE)</span>
              <span className="metric-value">{m.brier?.toFixed(4)}</span>
              <span className="metric-sub">ECE: {m.ece?.toFixed(4)}</span>
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="glass-card">
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "16px" }}>Validation Confusion Matrix</h4>
            <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
              <div style={{ overflowX: "auto" }}>
                <table style={{ borderCollapse: "collapse", textAlign: "center", fontSize: "0.85rem" }}>
                  <thead>
                    <tr>
                      <th style={{ padding: "8px" }}></th>
                      {data.classes?.map((c: string) => (
                        <th key={c} style={{ padding: "8px 16px", color: "var(--accent-primary)" }}>Pred: {c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.confusion_matrix?.map((row: number[], rIdx: number) => (
                      <tr key={rIdx}>
                        <td style={{ padding: "8px 16px", fontWeight: 600, color: "var(--text-secondary)" }}>
                          True: {data.classes?.[rIdx]}
                        </td>
                        {row.map((val: number, cIdx: number) => (
                          <td
                            key={cIdx}
                            style={{
                              padding: "12px 20px",
                              background: rIdx === cIdx ? "rgba(2, 132, 199, 0.2)" : "rgba(239, 68, 68, 0.1)",
                              border: "1px solid var(--card-border)",
                              fontWeight: 700
                            }}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ flex: 1, fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                <strong>Error Pattern Diagnostic:</strong>
                <p style={{ margin: "6px 0 0 0" }}>
                  Diagonal cells represent correctly calibrated predictions. Off-diagonal counts indicate misclassification frequency. 
                  Temperature scaling ensures predicted probabilities reflect true empirical empirical error rates.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* TAB 2: CURVES */}
      {activeTab === "curves" && (
        <div className="grid-2">
          {/* Learning Curve */}
          <div className="glass-card">
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
              <TrendingUp size={18} /> Learning Curve (Data Saturation)
            </h4>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "16px" }}>
              Shows validation accuracy as a function of training dataset volume. Positive slope indicates gathering more images will improve performance.
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {data.learning_curve?.map((point: any, idx: number) => (
                <div key={idx}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "4px" }}>
                    <span>{point.training_fraction * 100}% Training Data ({point.training_samples} samples)</span>
                    <span style={{ fontWeight: 700 }}>{(point.validation_accuracy * 100).toFixed(1)}% Acc</span>
                  </div>
                  <div style={{ height: "8px", borderRadius: "4px", background: "var(--bg-tertiary)", overflow: "hidden" }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${point.validation_accuracy * 100}%`,
                        background: "var(--accent-teal)"
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* What to do next */}
          <div className="glass-card">
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
              <HelpCircle size={18} /> What to Do Next (Artifact-Grounded Guidance)
            </h4>
            <ul style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8, paddingLeft: "20px", margin: 0 }}>
              <li>
                <strong>Sample Adequacy:</strong> {m.accuracy > m.majority_rate + 0.05 ? "Model strongly separates classes above baseline." : "Accuracy is near majority baseline; consider larger sample counts."}
              </li>
              <li>
                <strong>Patient Clustering:</strong> Ensure patient IDs are supplied in labels.csv to prevent data leakage across train/validation folds.
              </li>
              <li>
                <strong>Saliency Inspection:</strong> Review the Grad-CAM gallery to verify the backbone is attending to clinical pathology rather than edge artifacts.
              </li>
            </ul>
          </div>
        </div>
      )}

      {/* TAB 3: GRAD-CAM */}
      {activeTab === "gradcam" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          {/* Confident Correct */}
          <div className="glass-card">
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", color: "var(--accent-teal)" }}>
              Confident Correct Predictions (High Probability, Correct Ground Truth)
            </h4>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "16px" }}>
              {data.galleries?.confident_correct?.map((item: any, idx: number) => (
                <div key={idx} style={{ background: "var(--bg-tertiary)", padding: "10px", borderRadius: "8px", textAlign: "center" }}>
                  <img src={item.overlay_base64} alt="Grad-CAM" style={{ width: "100%", borderRadius: "6px", marginBottom: "8px" }} />
                  <div style={{ fontSize: "0.8rem", fontWeight: 700 }}>{item.true_class}</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Confidence: {(item.confidence * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          {/* Confident Wrong */}
          <div className="glass-card">
            <h4 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "12px", color: "#ef4444" }}>
              Confident Misclassifications (Model Confident But Wrong)
            </h4>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "16px" }}>
              {data.galleries?.confident_wrong?.length > 0 ? (
                data.galleries.confident_wrong.map((item: any, idx: number) => (
                  <div key={idx} style={{ background: "var(--bg-tertiary)", padding: "10px", borderRadius: "8px", textAlign: "center" }}>
                    <img src={item.overlay_base64} alt="Grad-CAM" style={{ width: "100%", borderRadius: "6px", marginBottom: "8px" }} />
                    <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "#ef4444" }}>Pred: {item.pred_class} (True: {item.true_class})</div>
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Confidence: {(item.confidence * 100).toFixed(1)}%</div>
                  </div>
                ))
              ) : (
                <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", padding: "12px" }}>
                  No confident misclassifications detected in validation subset.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
