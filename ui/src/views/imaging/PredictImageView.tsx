import React, { useState } from "react";
import { UploadCloud, AlertTriangle, ShieldCheck, Download, Eye, FileText } from "lucide-react";

export const PredictImageView: React.FC<{
  selectedModelId?: string | null;
}> = ({ selectedModelId }) => {
  const [modelId, setModelId] = useState<string>(selectedModelId || "bench_pneumoniamnist");
  const [file, setFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [result, setResult] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [overlayAlpha, setOverlayAlpha] = useState<number>(0.5);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setErrorMsg(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setPredicting(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`/api/image/models/${modelId}/predict`, {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        let msg = "Prediction failed";
        try {
          const contentType = res.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
            const err = await res.json();
            msg = err.detail || err.message || msg;
          } else {
            const text = await res.text();
            msg = text || `Server error (${res.status})`;
          }
        } catch {
          msg = `HTTP Error ${res.status}: ${res.statusText}`;
        }
        throw new Error(msg);
      }
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to process image inference.");
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Configuration & Upload */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "16px" }}>Medical Image Prediction & Saliency Analysis</h4>
        
        <div style={{ marginBottom: "16px" }}>
          <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
            Select Trained Model ID
          </label>
          <input
            type="text"
            className="select-input"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="e.g. bench_pneumoniamnist"
            style={{ width: "300px", padding: "8px 12px", background: "var(--bg-tertiary)", border: "1px solid var(--card-border)", color: "var(--text-primary)", borderRadius: "6px" }}
          />
        </div>

        <div
          style={{
            border: "2px dashed var(--card-border)",
            borderRadius: "12px",
            padding: "36px 20px",
            textAlign: "center",
            background: "var(--bg-tertiary)",
            cursor: "pointer"
          }}
          onClick={() => document.getElementById("predict-image-input")?.click()}
        >
          <input
            id="predict-image-input"
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <UploadCloud size={40} style={{ color: "var(--accent-primary)", margin: "0 auto 10px auto" }} />
          <h5 style={{ fontSize: "0.95rem", fontWeight: 600, margin: "0 0 4px 0" }}>
            {file ? file.name : "Select or drag clinical image (PNG/JPG/TIFF)"}
          </h5>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Supports radiographic slices and histological 2D images</span>
        </div>

        {file && (
          <div style={{ marginTop: "16px", display: "flex", justifyContent: "flex-end" }}>
            <button
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={predicting}
            >
              {predicting ? "Analyzing & Computing Grad-CAM..." : "Run Clinical Inference"}
            </button>
          </div>
        )}

        {errorMsg && (
          <div style={{ marginTop: "16px", padding: "12px", borderRadius: "8px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444", color: "#ef4444", fontSize: "0.85rem" }}>
            {errorMsg}
          </div>
        )}
      </div>

      {/* Results Card */}
      {result && (
        <div className="glass-card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
            <div>
              <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-teal)", textTransform: "uppercase" }}>Inference Decision</span>
              <h3 style={{ fontSize: "1.4rem", fontWeight: 800, margin: "4px 0" }}>
                Class: {result.predicted_class}
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
                File: {result.filename} | Model: {result.model_id}
              </p>
            </div>

            {/* Disclaimer Chip */}
            <div style={{ textAlign: "right", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              <div style={{ fontWeight: 700, color: "var(--accent-teal)" }}>Research prototype, not a diagnostic device.</div>
              <div>Single-touch calibrated probability</div>
            </div>
          </div>

          {/* OOD & Uncertainty Warning Banner */}
          {result.needs_expert_review && (
            <div style={{ marginBottom: "20px", padding: "14px 18px", borderRadius: "8px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444", display: "flex", alignItems: "center", gap: "12px" }}>
              <AlertTriangle size={24} style={{ color: "#ef4444", flexShrink: 0 }} />
              <div>
                <strong style={{ color: "#ef4444", fontSize: "0.9rem" }}>Safety Guardrail Flag: Needs Expert Review</strong>
                <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "4px 0 0 0" }}>
                  {result.safety_banner || "High predictive uncertainty or feature deviation detected. Model abstains from automated recommendation."}
                </p>
              </div>
            </div>
          )}

          {/* Probabilities and Grad-CAM Grid */}
          <div className="grid-2">
            <div>
              <h4 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "12px" }}>Calibrated Class Probabilities</h4>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "20px" }}>
                {Object.entries(result.class_probabilities || {}).map(([cName, prob]: [string, any]) => (
                  <div key={cName}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", marginBottom: "4px" }}>
                      <span style={{ fontWeight: 600 }}>{cName}</span>
                      <span style={{ fontFamily: "var(--font-mono)", fontWeight: 700 }}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: "8px", borderRadius: "4px", background: "var(--bg-tertiary)", overflow: "hidden" }}>
                      <div
                        style={{
                          height: "100%",
                          width: `${prob * 100}%`,
                          background: prob >= 0.5 ? "var(--accent-primary)" : "var(--accent-teal)"
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                <div><strong>Uncertainty Entropy:</strong> {result.uncertainty_score}</div>
                <div><strong>Out-of-Distribution:</strong> {result.is_ood ? "YES (Exceeds 95th Percentile)" : "NO (In-Distribution)"}</div>
              </div>
            </div>

            {/* Grad-CAM Viewer */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
                <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: 0, display: "flex", alignItems: "center", gap: "6px" }}>
                  <Eye size={16} /> Grad-CAM Attention Heatmap
                </h4>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem" }}>
                  <span>Overlay:</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={overlayAlpha}
                    onChange={(e) => setOverlayAlpha(parseFloat(e.target.value))}
                    style={{ width: "80px" }}
                  />
                </div>
              </div>

              {result.gradcam_overlay ? (
                <div style={{ textAlign: "center", background: "var(--bg-tertiary)", padding: "12px", borderRadius: "8px" }}>
                  <img
                    src={result.gradcam_overlay}
                    alt="Grad-CAM Overlay"
                    style={{ maxWidth: "100%", maxHeight: "260px", borderRadius: "6px" }}
                  />
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "8px", margin: 0 }}>
                    Warm regions highlight visual anatomical patterns most responsible for the decision.
                  </p>
                </div>
              ) : (
                <div style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>Heatmap not available</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
