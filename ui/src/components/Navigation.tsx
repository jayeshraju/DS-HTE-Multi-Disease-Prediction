import React from "react";
import {
  Activity,
  Layers,
  Sliders,
  BarChart3,
  FlaskConical,
  Cpu,
  UserCheck,
  Sun,
  Moon,
  Database,
  UploadCloud,
  PlayCircle,
  Image as ImageIcon,
  Crosshair,
  Bot,
  Sparkles
} from "lucide-react";
import { RunManifest } from "../types";

export type ViewType =
  | "overview"
  | "performance"
  | "threshold"
  | "calibration"
  | "ablation"
  | "explain"
  | "patient"
  | "copilot"
  | "image-upload"
  | "image-jobs"
  | "image-results"
  | "image-predict"
  | "image-agent";

interface NavigationProps {
  currentView: ViewType;
  onSelectView: (view: ViewType) => void;
  manifest: RunManifest | null;
  theme: "dark" | "light";
  onToggleTheme: () => void;
}

export const Sidebar: React.FC<NavigationProps> = ({
  currentView,
  onSelectView,
  manifest
}) => {
  const tabularNavItems: { id: ViewType; label: string; icon: React.ReactNode }[] = [
    { id: "overview", label: "Overview & Cohorts", icon: <Activity size={18} /> },
    { id: "performance", label: "Model Performance", icon: <BarChart3 size={18} /> },
    { id: "threshold", label: "Threshold Explorer", icon: <Sliders size={18} /> },
    { id: "calibration", label: "Calibration & Reliability", icon: <Layers size={18} /> },
    { id: "ablation", label: "Ablation Study", icon: <FlaskConical size={18} /> },
    { id: "explain", label: "Model Explainability", icon: <Cpu size={18} /> },
    { id: "patient", label: "Try a Patient", icon: <UserCheck size={18} /> },
  ];

  const imageLabNavItems: { id: ViewType; label: string; icon: React.ReactNode }[] = [
    { id: "image-upload", label: "Upload Dataset", icon: <UploadCloud size={18} /> },
    { id: "image-jobs", label: "Training Jobs", icon: <PlayCircle size={18} /> },
    { id: "image-results", label: "Image Results", icon: <ImageIcon size={18} /> },
    { id: "image-predict", label: "Predict Image", icon: <Crosshair size={18} /> },
    { id: "image-agent", label: "Analysis Agent", icon: <Bot size={18} /> },
  ];

  return (
    <aside className="sidebar">
      <div style={{ padding: "24px 20px", borderBottom: "1px solid var(--card-border)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "8px",
              background: "linear-gradient(135deg, #0284c7, #6366f1)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "white"
            }}
          >
            <Activity size={22} />
          </div>
          <div>
            <h1 style={{ fontSize: "1.1rem", fontWeight: 800, letterSpacing: "-0.02em" }}>DS-HTE</h1>
            <p style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Multi-Disease Framework</p>
          </div>
        </div>
      </div>

      <div className="nav-group">
        <div className="nav-label">Tabular Multi-Disease</div>
        {tabularNavItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${currentView === item.id ? "active" : ""}`}
            onClick={() => onSelectView(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      <div className="nav-group" style={{ marginTop: "16px" }}>
        <div className="nav-label" style={{ color: "var(--accent-purple, #a855f7)" }}>AI Assistant</div>
        <div
          className={`nav-item ${currentView === "copilot" ? "active" : ""}`}
          onClick={() => onSelectView("copilot")}
          style={{ background: currentView === "copilot" ? "rgba(168, 85, 247, 0.15)" : undefined }}
        >
          <Sparkles size={18} color={currentView === "copilot" ? "#c084fc" : undefined} />
          <span style={{ fontWeight: 600 }}>DS-HTE Copilot</span>
        </div>
      </div>

      <div className="nav-group" style={{ marginTop: "16px" }}>
        <div className="nav-label" style={{ color: "var(--accent-teal)" }}>Image Lab (Medical Vision)</div>
        {imageLabNavItems.map((item) => (
          <div
            key={item.id}
            className={`nav-item ${currentView === item.id ? "active" : ""}`}
            onClick={() => onSelectView(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </div>
        ))}
      </div>

      {manifest && (
        <div style={{ padding: "16px 20px", borderTop: "1px solid var(--card-border)", fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
            <Database size={14} />
            <span style={{ fontWeight: 600 }}>Active Run</span>
          </div>
          <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {manifest.run_id}
          </div>
        </div>
      )}
    </aside>
  );
};

export const Header: React.FC<{
  currentView: ViewType;
  theme: "dark" | "light";
  onToggleTheme: () => void;
  manifest: RunManifest | null;
}> = ({ currentView, theme, onToggleTheme, manifest }) => {
  const titles: Record<ViewType, string> = {
    overview: "Multi-Disease Clinical Decision Support Overview",
    performance: "Model Performance (50 Outer Folds Repeated CV & Holdout)",
    threshold: "Interactive Threshold & Net Benefit Explorer",
    calibration: "Probability Calibration & Clinical Reliability",
    ablation: "Architectural Component Ablation Study",
    explain: "Explainability Consensus (SHAP + LIME) & Subgroups",
    patient: "Live Patient Risk Profiling & Decision Support",
    copilot: "DS-HTE Clinical & ML AI Assistant Agent",
    "image-upload": "Image Lab: Dataset Ingestion & Pre-Training Validation",
    "image-jobs": "Image Lab: Training Jobs & Real-Time Progress",
    "image-results": "Image Lab: Deep Model Performance & Grad-CAM Interpretability",
    "image-predict": "Image Lab: Interactive Inference, OOD Detection & Saliency",
    "image-agent": "Medical Imaging Analysis Agent (Grounded Q&A)",
  };

  return (
    <header className="header">
      <div>
        <h2 style={{ fontSize: "1.15rem", fontWeight: 700 }}>{titles[currentView]}</h2>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        {manifest && (
          <span className="badge badge-info">
            Profile: {manifest.profile.toUpperCase()}
          </span>
        )}
        <span className="badge badge-success">
          50 Outer Folds
        </span>

        <button
          className="btn btn-secondary"
          onClick={onToggleTheme}
          style={{ width: "36px", height: "36px", padding: 0 }}
          title={`Switch to ${theme === "dark" ? "Light" : "Dark"} Mode`}
        >
          {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
};
