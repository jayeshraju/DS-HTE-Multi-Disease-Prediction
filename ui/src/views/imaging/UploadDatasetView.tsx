import React, { useState } from "react";
import { UploadCloud, CheckCircle2, AlertTriangle, XCircle, ArrowRight, ShieldAlert } from "lucide-react";

export const UploadDatasetView: React.FC<{
  onStartTraining: (datasetId: string) => void;
}> = ({ onStartTraining }) => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [validation, setValidation] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setValidation(null);
      setErrorMsg(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/image/datasets", {
        method: "POST",
        body: formData
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setDatasetId(data.dataset_id);
      setValidation(data.validation);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to upload and validate dataset.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Scope and Disclaimer Banner */}
      <div className="glass-card" style={{ borderLeft: "4px solid var(--accent-primary)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: "12px" }}>
          <ShieldAlert size={22} style={{ color: "var(--accent-primary)", flexShrink: 0, marginTop: "2px" }} />
          <div>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 6px 0" }}>Image Lab v1 Scope & Privacy Protocol</h3>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6, margin: 0 }}>
              Supports <strong>2D image classification</strong> (PNG/JPG/TIFF and single-slice DICOM). Segmentation, detection, and 3D volumes are out of scope. 
              An unlabelled image cannot be classified by an untrained system; performance strictly depends on your labelled ground truth. 
              All processing is local; DICOM PHI tags (name, ID, birth date, institution) are stripped before storage.
            </p>
            <div style={{ marginTop: "8px", fontSize: "0.75rem", fontWeight: 700, color: "var(--accent-teal)" }}>
              Research prototype, not a diagnostic device.
            </div>
          </div>
        </div>
      </div>

      {/* Upload Zone */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "16px" }}>Upload Labelled Medical Image Dataset</h4>
        
        <div
          style={{
            border: "2px dashed var(--card-border)",
            borderRadius: "12px",
            padding: "40px 20px",
            textAlign: "center",
            background: "var(--bg-tertiary)",
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
          onClick={() => document.getElementById("dataset-file-input")?.click()}
        >
          <input
            id="dataset-file-input"
            type="file"
            accept=".zip"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <UploadCloud size={44} style={{ color: "var(--accent-primary)", margin: "0 auto 12px auto" }} />
          <h5 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "6px" }}>
            {file ? file.name : "Click to select or drag and drop dataset .ZIP"}
          </h5>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", margin: 0 }}>
            Accepts folder structure (e.g. <code>benign/*.png</code>, <code>malignant/*.png</code>) or zip with <code>labels.csv</code>
          </p>
        </div>

        {file && !validation && (
          <div style={{ marginTop: "16px", display: "flex", justifyContent: "flex-end" }}>
            <button
              className="btn btn-primary"
              onClick={handleUpload}
              disabled={uploading}
            >
              {uploading ? "Analyzing & De-identifying..." : "Validate Dataset Integrity"}
            </button>
          </div>
        )}

        {errorMsg && (
          <div style={{ marginTop: "16px", padding: "12px", borderRadius: "8px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444", color: "#ef4444", fontSize: "0.85rem" }}>
            {errorMsg}
          </div>
        )}
      </div>

      {/* Live Validation Report */}
      {validation && (
        <div className="glass-card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <h4 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0 }}>Dataset Validation Report</h4>
            {validation.can_train ? (
              <span className="badge badge-success" style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <CheckCircle2 size={14} /> Ready to Train
              </span>
            ) : (
              <span className="badge badge-danger" style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <XCircle size={14} /> Training Blocked
              </span>
            )}
          </div>

          {/* Quick Metrics */}
          <div className="grid-3" style={{ marginBottom: "20px" }}>
            <div className="metric-box">
              <span className="metric-label">Total Valid Images</span>
              <span className="metric-value">{validation.total_valid_images}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Classes Detected</span>
              <span className="metric-value">{validation.num_classes}</span>
            </div>
            <div className="metric-box">
              <span className="metric-label">Patient ID Present</span>
              <span className="metric-value" style={{ color: validation.has_patient_id ? "var(--accent-teal)" : "var(--accent-amber)" }}>
                {validation.has_patient_id ? "YES (Grouped)" : "NO (Warning)"}
              </span>
            </div>
          </div>

          {/* Class Distribution */}
          <div style={{ marginBottom: "20px" }}>
            <h5 style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "8px" }}>Class Distribution</h5>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {Object.entries(validation.class_distribution || {}).map(([cName, count]: [string, any]) => (
                <div key={cName}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "3px" }}>
                    <span>{cName}</span>
                    <span style={{ fontWeight: 600 }}>{count} images</span>
                  </div>
                  <div style={{ height: "6px", borderRadius: "3px", background: "var(--bg-tertiary)", overflow: "hidden" }}>
                    <div
                      style={{
                        height: "100%",
                        width: `${Math.min(100, (count / Math.max(1, validation.total_valid_images)) * 100)}%`,
                        background: "var(--accent-primary)"
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Blocking Issues */}
          {validation.blocking_issues && validation.blocking_issues.length > 0 && (
            <div style={{ marginBottom: "16px", padding: "12px", borderRadius: "8px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#ef4444", fontWeight: 700, marginBottom: "6px" }}>
                <XCircle size={16} /> Blocking Issues (Must fix to proceed)
              </div>
              <ul style={{ margin: 0, paddingLeft: "20px", fontSize: "0.85rem", color: "#ef4444" }}>
                {validation.blocking_issues.map((issue: string, idx: number) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Warnings */}
          {validation.warnings && validation.warnings.length > 0 && (
            <div style={{ marginBottom: "20px", padding: "12px", borderRadius: "8px", background: "rgba(245, 158, 11, 0.1)", border: "1px solid #f59e0b" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#f59e0b", fontWeight: 700, marginBottom: "6px" }}>
                <AlertTriangle size={16} /> Integrity Warnings
              </div>
              <ul style={{ margin: 0, paddingLeft: "20px", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                {validation.warnings.map((warn: string, idx: number) => (
                  <li key={idx}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Actions */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Exact Duplicates: {validation.exact_duplicates_count} | Near Duplicates: {validation.near_duplicates_count}
            </span>
            <button
              className="btn btn-primary"
              disabled={!validation.can_train}
              onClick={() => datasetId && onStartTraining(datasetId)}
              style={{ display: "flex", alignItems: "center", gap: "8px" }}
            >
              <span>Configure & Start Training</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
