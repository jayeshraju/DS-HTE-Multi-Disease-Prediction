import React from "react";
import { CurvePoint, ConfusionData } from "../types";

// ==========================================
// ROC & PR CURVE
// ==========================================
export const RocPrChart: React.FC<{
  points: CurvePoint[];
  type: "roc" | "pr";
  auc?: number;
  width?: number;
  height?: number;
}> = ({ points, type, auc, width = 450, height = 300 }) => {
  const pad = 40;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;

  const isRoc = type === "roc";
  const xKey = isRoc ? "fpr" : "recall";
  const yKey = isRoc ? "tpr" : "precision";

  const pathD = points.reduce((acc, pt, i) => {
    const xVal = (pt as any)[xKey] ?? 0;
    const yVal = (pt as any)[yKey] ?? 0;
    const x = pad + xVal * innerW;
    const y = pad + (1 - yVal) * innerH;
    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, "");

  return (
    <div style={{ position: "relative", width, height }}>
      <svg width={width} height={height} className="chart-svg">
        {/* Background Grid */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
          <React.Fragment key={tick}>
            <line
              x1={pad + tick * innerW}
              y1={pad}
              x2={pad + tick * innerW}
              y2={height - pad}
              className="chart-grid"
            />
            <line
              x1={pad}
              y1={pad + (1 - tick) * innerH}
              x2={width - pad}
              y2={pad + (1 - tick) * innerH}
              className="chart-grid"
            />
            <text x={pad + tick * innerW} y={height - pad + 18} textAnchor="middle" className="chart-label">
              {tick.toFixed(2)}
            </text>
            <text x={pad - 10} y={pad + (1 - tick) * innerH + 4} textAnchor="end" className="chart-label">
              {tick.toFixed(2)}
            </text>
          </React.Fragment>
        ))}

        {/* Diagonal Reference */}
        {isRoc && (
          <line
            x1={pad}
            y1={height - pad}
            x2={width - pad}
            y2={pad}
            stroke="var(--text-muted)"
            strokeDasharray="4 4"
            strokeWidth="1.5"
          />
        )}

        {/* Curve Path */}
        <path d={pathD} fill="none" stroke="var(--accent-primary)" strokeWidth="3" strokeLinecap="round" />

        {/* Points */}
        {points.slice(0, 30).map((pt, idx) => {
          const xVal = (pt as any)[xKey] ?? 0;
          const yVal = (pt as any)[yKey] ?? 0;
          return (
            <circle
              key={idx}
              cx={pad + xVal * innerW}
              cy={pad + (1 - yVal) * innerH}
              r="3.5"
              fill="var(--accent-primary)"
            />
          );
        })}

        {/* Axis Labels */}
        <text x={width / 2} y={height - 8} textAnchor="middle" className="chart-label" style={{ fontWeight: 600 }}>
          {isRoc ? "False Positive Rate (1 - Specificity)" : "Recall (Sensitivity)"}
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
          className="chart-label"
          style={{ fontWeight: 600 }}
        >
          {isRoc ? "True Positive Rate (Sensitivity)" : "Precision (PPV)"}
        </text>
      </svg>

      {auc !== undefined && (
        <div style={{ position: "absolute", top: pad + 10, right: pad + 10 }}>
          <span className="badge badge-info">AUC = {auc.toFixed(3)}</span>
        </div>
      )}
    </div>
  );
};

// ==========================================
// CALIBRATION CURVE
// ==========================================
export const CalibrationChart: React.FC<{
  points: CurvePoint[];
  ece?: number;
  brier?: number;
  width?: number;
  height?: number;
}> = ({ points, ece, brier, width = 450, height = 300 }) => {
  const pad = 40;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;

  const sortedPoints = [...points].sort((a, b) => (a.prob_pred ?? 0) - (b.prob_pred ?? 0));
  const pathD = sortedPoints.reduce((acc, pt, i) => {
    const x = pad + (pt.prob_pred ?? 0) * innerW;
    const y = pad + (1 - (pt.prob_true ?? 0)) * innerH;
    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, "");

  return (
    <div style={{ position: "relative", width, height }}>
      <svg width={width} height={height} className="chart-svg">
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
          <React.Fragment key={tick}>
            <line x1={pad + tick * innerW} y1={pad} x2={pad + tick * innerW} y2={height - pad} className="chart-grid" />
            <line x1={pad} y1={pad + (1 - tick) * innerH} x2={width - pad} y2={pad + (1 - tick) * innerH} className="chart-grid" />
            <text x={pad + tick * innerW} y={height - pad + 18} textAnchor="middle" className="chart-label">
              {tick.toFixed(1)}
            </text>
            <text x={pad - 8} y={pad + (1 - tick) * innerH + 4} textAnchor="end" className="chart-label">
              {tick.toFixed(1)}
            </text>
          </React.Fragment>
        ))}

        {/* 45-degree Perfect Calibration Line */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={pad} stroke="var(--text-muted)" strokeDasharray="4 4" strokeWidth="1.5" />

        {/* Model Calibration Path */}
        <path d={pathD} fill="none" stroke="var(--success)" strokeWidth="3" />

        {sortedPoints.map((pt, idx) => (
          <circle
            key={idx}
            cx={pad + (pt.prob_pred ?? 0) * innerW}
            cy={pad + (1 - (pt.prob_true ?? 0)) * innerH}
            r="4.5"
            fill="var(--success)"
            stroke="var(--bg-secondary)"
            strokeWidth="2"
          />
        ))}

        <text x={width / 2} y={height - 8} textAnchor="middle" className="chart-label" style={{ fontWeight: 600 }}>
          Mean Predicted Risk Probability
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
          className="chart-label"
          style={{ fontWeight: 600 }}
        >
          Observed Pathological Fraction
        </text>
      </svg>

      <div style={{ position: "absolute", top: pad + 10, right: pad + 10, display: "flex", gap: "8px" }}>
        {ece !== undefined && <span className="badge badge-success">ECE = {ece.toFixed(4)}</span>}
        {brier !== undefined && <span className="badge badge-info">Brier = {brier.toFixed(4)}</span>}
      </div>
    </div>
  );
};

// ==========================================
// DECISION CURVE ANALYSIS (DCA)
// ==========================================
export const DcaChart: React.FC<{
  points: CurvePoint[];
  width?: number;
  height?: number;
}> = ({ points, width = 450, height = 300 }) => {
  const pad = 40;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;

  const maxNb = Math.max(0.4, ...points.map((p) => p.net_benefit ?? 0));
  const minNb = -0.1;

  const getY = (val: number) => {
    const norm = (val - minNb) / (maxNb - minNb);
    return pad + (1 - norm) * innerH;
  };

  const modelPath = points.reduce((acc, pt, i) => {
    const x = pad + (pt.threshold ?? 0) * innerW;
    const y = getY(pt.net_benefit ?? 0);
    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, "");

  const treatAllPath = points.reduce((acc, pt, i) => {
    const x = pad + (pt.threshold ?? 0) * innerW;
    const y = getY(pt.treat_all ?? 0);
    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, "");

  return (
    <div style={{ position: "relative", width, height }}>
      <svg width={width} height={height} className="chart-svg">
        <line x1={pad} y1={getY(0)} x2={width - pad} y2={getY(0)} stroke="var(--text-muted)" strokeWidth="1.5" />
        <path d={treatAllPath} fill="none" stroke="var(--warning)" strokeDasharray="4 4" strokeWidth="2" />
        <path d={modelPath} fill="none" stroke="var(--accent-primary)" strokeWidth="3" />

        <text x={width / 2} y={height - 8} textAnchor="middle" className="chart-label" style={{ fontWeight: 600 }}>
          High-Risk Threshold Probability
        </text>
        <text
          x={14}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${height / 2})`}
          className="chart-label"
          style={{ fontWeight: 600 }}
        >
          Net Benefit
        </text>
      </svg>
      <div style={{ display: "flex", gap: "12px", justifyContent: "center", marginTop: "4px", fontSize: "0.75rem" }}>
        <span style={{ color: "var(--accent-primary)", fontWeight: 600 }}>— DS-HTE Model</span>
        <span style={{ color: "var(--warning)", fontWeight: 600 }}>--- Treat All</span>
        <span style={{ color: "var(--text-muted)", fontWeight: 600 }}>— Treat None (y=0)</span>
      </div>
    </div>
  );
};

// ==========================================
// CONFUSION MATRIX COMPONENT
// ==========================================
export const ConfusionMatrixView: React.FC<{ data: ConfusionData }> = ({ data }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "100px 140px 140px",
          gap: "8px",
          alignItems: "center",
          textAlign: "center"
        }}
      >
        <div></div>
        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)" }}>Predicted Neg (0)</div>
        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)" }}>Predicted Pos (1)</div>

        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)", textAlign: "right" }}>
          Actual Neg (0)
        </div>
        <div
          style={{
            background: "rgba(16, 185, 129, 0.15)",
            border: "1px solid var(--success)",
            borderRadius: "8px",
            padding: "16px"
          }}
        >
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>True Negatives (TN)</div>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--success)" }}>{data.tn}</div>
        </div>
        <div
          style={{
            background: "rgba(239, 68, 68, 0.15)",
            border: "1px solid var(--danger)",
            borderRadius: "8px",
            padding: "16px"
          }}
        >
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>False Positives (FP)</div>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--danger)" }}>{data.fp}</div>
        </div>

        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--text-secondary)", textAlign: "right" }}>
          Actual Pos (1)
        </div>
        <div
          style={{
            background: "rgba(245, 158, 11, 0.15)",
            border: "1px solid var(--warning)",
            borderRadius: "8px",
            padding: "16px"
          }}
        >
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>False Negatives (FN)</div>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--warning)" }}>{data.fn}</div>
        </div>
        <div
          style={{
            background: "rgba(56, 189, 248, 0.15)",
            border: "1px solid var(--accent-primary)",
            borderRadius: "8px",
            padding: "16px"
          }}
        >
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>True Positives (TP)</div>
          <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-primary)" }}>{data.tp}</div>
        </div>
      </div>

      {/* Derived Clinical Metrics */}
      <div className="grid-3" style={{ marginTop: "8px" }}>
        <div className="kpi-card" style={{ padding: "12px", background: "var(--bg-tertiary)", borderRadius: "8px" }}>
          <span className="kpi-title">Sensitivity (Recall)</span>
          <span className="kpi-value" style={{ fontSize: "1.25rem", color: "var(--accent-primary)" }}>
            {(data.sensitivity * 100).toFixed(1)}%
          </span>
          <span className="kpi-sub">TP / (TP + FN)</span>
        </div>
        <div className="kpi-card" style={{ padding: "12px", background: "var(--bg-tertiary)", borderRadius: "8px" }}>
          <span className="kpi-title">Specificity</span>
          <span className="kpi-value" style={{ fontSize: "1.25rem", color: "var(--success)" }}>
            {(data.specificity * 100).toFixed(1)}%
          </span>
          <span className="kpi-sub">TN / (TN + FP)</span>
        </div>
        <div className="kpi-card" style={{ padding: "12px", background: "var(--bg-tertiary)", borderRadius: "8px" }}>
          <span className="kpi-title">PPV (Precision)</span>
          <span className="kpi-value" style={{ fontSize: "1.25rem", color: "var(--info)" }}>
            {(data.ppv * 100).toFixed(1)}%
          </span>
          <span className="kpi-sub">TP / (TP + FP)</span>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// SHAP WATERFALL ATTRIBUTION CHART
// ==========================================
export const WaterfallChart: React.FC<{
  attributions: Record<string, number>;
  baseValue?: number;
  finalProb?: number;
}> = ({ attributions, baseValue = 0.35, finalProb = 0.72 }) => {
  const items = Object.entries(attributions);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
        <span>Prior Prevalence: {(baseValue * 100).toFixed(1)}%</span>
        <span>Predicted Calibrated Risk: <strong>{(finalProb * 100).toFixed(1)}%</strong></span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {items.map(([feat, val]) => {
          const isPositive = val >= 0;
          const pctWidth = Math.min(100, Math.abs(val) * 300);
          return (
            <div key={feat} style={{ display: "grid", gridTemplateColumns: "140px 1fr 60px", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)", textAlign: "right" }}>
                {feat}
              </span>
              <div style={{ display: "flex", alignItems: "center", height: "20px", background: "var(--bg-tertiary)", borderRadius: "4px", overflow: "hidden" }}>
                <div
                  style={{
                    width: `${pctWidth}%`,
                    height: "100%",
                    background: isPositive ? "var(--danger)" : "var(--success)",
                    borderRadius: "4px",
                    transition: "width 0.3s ease"
                  }}
                />
              </div>
              <span
                style={{
                  fontSize: "0.8rem",
                  fontFamily: "var(--font-mono)",
                  fontWeight: 600,
                  color: isPositive ? "var(--danger)" : "var(--success)"
                }}
              >
                {isPositive ? `+${val.toFixed(3)}` : val.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
