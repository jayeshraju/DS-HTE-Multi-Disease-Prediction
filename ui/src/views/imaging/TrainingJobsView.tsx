import React, { useState, useEffect } from "react";
import { Play, CheckCircle2, Clock, XCircle, ArrowRight, RefreshCw, Cpu } from "lucide-react";

export const TrainingJobsView: React.FC<{
  initialDatasetId?: string | null;
  onSelectJobResults: (jobId: string) => void;
}> = ({ initialDatasetId, onSelectJobResults }) => {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [datasetId, setDatasetId] = useState<string>(initialDatasetId || "");
  const [backbone, setBackbone] = useState<string>("resnet18");
  const [headType, setHeadType] = useState<string>("lr");
  const [submitting, setSubmitting] = useState<boolean>(false);

  const fetchJobs = async () => {
    try {
      const res = await fetch("/api/image/jobs");
      if (res.ok) {
        const data = await res.json();
        setJobs(data);
      }
    } catch (e) {
      console.error("Failed to load jobs", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 2500);
    return () => clearInterval(interval);
  }, []);

  const handleStartJob = async () => {
    if (!datasetId) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/image/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          backbone: backbone,
          head_type: headType
        })
      });
      if (res.ok) {
        fetchJobs();
      }
    } catch (e) {
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await fetch(`/api/image/jobs/${jobId}/cancel`, { method: "POST" });
      fetchJobs();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Launch Job Card */}
      <div className="glass-card">
        <h4 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "16px" }}>Configure & Launch Training Job</h4>
        
        <div className="grid-3" style={{ marginBottom: "16px" }}>
          <div>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
              Dataset ID
            </label>
            <input
              type="text"
              className="select-input"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              placeholder="e.g. ds_4a2f8b1c"
              style={{ width: "100%", padding: "8px 12px", background: "var(--bg-tertiary)", border: "1px solid var(--card-border)", color: "var(--text-primary)", borderRadius: "6px" }}
            />
          </div>

          <div>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
              Vision Backbone
            </label>
            <select
              className="select-input"
              value={backbone}
              onChange={(e) => setBackbone(e.target.value)}
              style={{ width: "100%", padding: "8px 12px", background: "var(--bg-tertiary)", border: "1px solid var(--card-border)", color: "var(--text-primary)", borderRadius: "6px" }}
            >
              <option value="resnet18">ResNet-18 (Fast, CPU recommended)</option>
              <option value="efficientnet_b0">EfficientNet-B0 (Lightweight)</option>
              <option value="densenet121">DenseNet-121 (Radiology Standard)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "6px" }}>
              Classification Head
            </label>
            <select
              className="select-input"
              value={headType}
              onChange={(e) => setHeadType(e.target.value)}
              style={{ width: "100%", padding: "8px 12px", background: "var(--bg-tertiary)", border: "1px solid var(--card-border)", color: "var(--text-primary)", borderRadius: "6px" }}
            >
              <option value="lr">Balanced Logistic Regression (Fastest)</option>
              <option value="gbdt">Gradient Boosted Decision Trees (GBDT)</option>
            </select>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            Includes temperature scaling calibration on validation folds & Mahalanobis OOD safety fitting.
          </span>
          <button
            className="btn btn-primary"
            onClick={handleStartJob}
            disabled={!datasetId || submitting}
            style={{ display: "flex", alignItems: "center", gap: "6px" }}
          >
            <Play size={16} />
            <span>{submitting ? "Starting..." : "Start Training"}</span>
          </button>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="glass-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <h4 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0 }}>Active & Recent Training Jobs</h4>
          <button className="btn btn-secondary" onClick={fetchJobs} style={{ padding: "6px 12px", fontSize: "0.8rem" }}>
            <RefreshCw size={14} style={{ marginRight: "4px" }} /> Refresh
          </button>
        </div>

        {loading ? (
          <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)" }}>Loading jobs...</div>
        ) : jobs.length === 0 ? (
          <div style={{ padding: "40px", textAlign: "center", color: "var(--text-muted)" }}>
            No jobs found. Launch a training job above or run the benchmark.
          </div>
        ) : (
          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Backbone</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Action / Details</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", fontWeight: 600 }}>
                      {job.job_id}
                    </td>
                    <td>
                      <span className="badge badge-info">{job.backbone?.toUpperCase() || "RESNET18"}</span>
                    </td>
                    <td>
                      {job.status === "completed" && (
                        <span className="badge badge-success" style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                          <CheckCircle2 size={12} /> Completed
                        </span>
                      )}
                      {job.status === "running" && (
                        <span className="badge badge-info" style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                          <RefreshCw size={12} className="spin" /> Running
                        </span>
                      )}
                      {job.status === "queued" && (
                        <span className="badge badge-neutral" style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}>
                          <Clock size={12} /> Queued
                        </span>
                      )}
                      {job.status === "cancelled" && (
                        <span className="badge badge-danger">Cancelled</span>
                      )}
                    </td>
                    <td style={{ minWidth: "160px" }}>
                      <div style={{ fontSize: "0.75rem", marginBottom: "4px", color: "var(--text-muted)" }}>
                        {job.message || `${Math.round((job.progress || 0) * 100)}%`}
                      </div>
                      <div style={{ height: "6px", borderRadius: "3px", background: "var(--bg-tertiary)", overflow: "hidden" }}>
                        <div
                          style={{
                            height: "100%",
                            width: `${Math.round((job.progress || 0) * 100)}%`,
                            background: job.status === "completed" ? "var(--accent-teal)" : "var(--accent-primary)",
                            transition: "width 0.3s ease"
                          }}
                        />
                      </div>
                    </td>
                    <td>
                      {job.status === "completed" ? (
                        <button
                          className="btn btn-primary"
                          onClick={() => onSelectJobResults(job.job_id)}
                          style={{ padding: "6px 12px", fontSize: "0.8rem", display: "inline-flex", alignItems: "center", gap: "4px" }}
                        >
                          <span>Results</span>
                          <ArrowRight size={14} />
                        </button>
                      ) : job.status === "running" ? (
                        <button
                          className="btn btn-secondary"
                          onClick={() => handleCancelJob(job.job_id)}
                          style={{ padding: "6px 12px", fontSize: "0.8rem", color: "#ef4444" }}
                        >
                          Cancel
                        </button>
                      ) : (
                        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
