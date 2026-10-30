import React, { useState, useEffect } from "react";
import {
  Sparkles,
  Send,
  Download,
  CheckCircle,
  XCircle,
  AlertTriangle,
  FileText,
  Search,
  Cpu,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  ExternalLink,
  Bot
} from "lucide-react";

interface CopilotMessage {
  id: string;
  sender: "user" | "copilot";
  text: string;
  mode?: string;
  toolCalls?: any[];
  sources?: { tool: string; source: string; label: string }[];
  isGrounded?: boolean;
  timestamp: string;
}

interface Proposal {
  proposal_id: string;
  cohort: string;
  hypothesis: string;
  status: string;
  inner_cv_auc?: number;
  inner_cv_balanced_acc?: number;
}

export const CopilotView: React.FC = () => {
  const [messages, setMessages] = useState<CopilotMessage[]>([
    {
      id: "welcome",
      sender: "copilot",
      text: "Hello! I am the **DS-HTE Clinical & ML Agent**, an intelligent assistant grounded in multi-disease prediction, medical imaging, and clinical decision support. I can assist with any questions about disease biology, patient risk assessments, model architectures, calibration, image analysis, or Python/API code!",
      mode: "explain",
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      isGrounded: true
    }
  ]);

  const [inputQuery, setInputQuery] = useState("");
  const [activeMode, setActiveMode] = useState<"explain" | "analyst" | "improver" | "reporter">("explain");
  const [selectedCohort, setSelectedCohort] = useState("PIMA");
  const [loading, setLoading] = useState(false);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [expandedTrace, setExpandedTrace] = useState<Record<string, boolean>>({});

  const suggestedQuestions = [
    "How does the DS-HTE Transformer Ensemble work?",
    "Explain PIMA diabetes clinical features & risk factors",
    "How does Youden's J threshold calibration work?",
    "Evaluate patient: glucose 165, bmi 34, age 52",
    "Why is ILPD weak?",
    "How do I upload and train an Image Lab model?"
  ];

  const fetchProposals = async () => {
    try {
      const res = await fetch("/api/copilot/proposals");
      if (res.ok) {
        const data = await res.json();
        setProposals(data.proposals || []);
      }
    } catch (e) {
      console.error("Failed to load proposals", e);
    }
  };

  useEffect(() => {
    fetchProposals();
  }, []);

  const handleSend = async (queryText?: string) => {
    const q = queryText || inputQuery;
    if (!q.trim() || loading) return;

    const userMsg: CopilotMessage = {
      id: "u_" + Date.now(),
      sender: "user",
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setLoading(true);

    try {
      const res = await fetch("/api/copilot/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          mode: activeMode,
          cohort: selectedCohort
        })
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.statusText}`);
      }

      const data = await res.json();

      const copilotMsg: CopilotMessage = {
        id: "c_" + Date.now(),
        sender: "copilot",
        text: data.answer,
        mode: data.mode,
        toolCalls: data.tool_calls,
        sources: data.sources,
        isGrounded: data.is_grounded,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };

      setMessages((prev) => [...prev, copilotMsg]);
      fetchProposals();
    } catch (err: any) {
      const errorMsg: CopilotMessage = {
        id: "err_" + Date.now(),
        sender: "copilot",
        text: `Error connecting to Copilot: ${err.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleApproveProposal = async (proposalId: string) => {
    try {
      const res = await fetch(`/api/copilot/proposals/${proposalId}/approve`, { method: "POST" });
      if (res.ok) {
        fetchProposals();
      }
    } catch (e) {
      console.error("Failed to approve", e);
    }
  };

  const handleRejectProposal = async (proposalId: string) => {
    try {
      const res = await fetch(`/api/copilot/proposals/${proposalId}/reject`, { method: "POST" });
      if (res.ok) {
        fetchProposals();
      }
    } catch (e) {
      console.error("Failed to reject", e);
    }
  };

  const downloadAnswer = (text: string) => {
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `copilot_analysis_${Date.now()}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="view-container animate-fade-in" style={{ maxWidth: "1200px", margin: "0 auto" }}>
      {/* Disclaimer Banner */}
      <div
        className="card"
        style={{
          borderLeft: "4px solid var(--accent-amber)",
          background: "rgba(245, 158, 11, 0.08)",
          marginBottom: "20px"
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <AlertTriangle color="var(--accent-amber)" size={20} />
          <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            <strong>Research Prototype Notice:</strong> The DS-HTE Copilot is an experimental decision-support agent.
            It cannot provide clinical medical diagnoses, prescribe treatments, or recommend dosages.
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "24px" }} className="responsive-copilot-grid">
        {/* Main Chat Column */}
        <div>
          {/* Controls Header */}
          <div className="card" style={{ marginBottom: "16px", padding: "16px 20px" }}>
            <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>Mode:</span>
                <div style={{ display: "flex", gap: "6px" }}>
                  {(["explain", "analyst", "improver", "reporter"] as const).map((m) => (
                    <button
                      key={m}
                      className={`btn ${activeMode === m ? "btn-primary" : "btn-secondary"}`}
                      style={{ padding: "6px 12px", fontSize: "0.78rem", textTransform: "capitalize" }}
                      onClick={() => setActiveMode(m)}
                    >
                      {m}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>Cohort:</span>
                <select
                  className="form-select"
                  style={{ padding: "6px 12px", fontSize: "0.8rem", width: "120px" }}
                  value={selectedCohort}
                  onChange={(e) => setSelectedCohort(e.target.value)}
                >
                  <option value="PIMA">PIMA</option>
                  <option value="Cleveland">Cleveland</option>
                  <option value="CKD">CKD</option>
                  <option value="ILPD">ILPD</option>
                  <option value="WBCD">WBCD</option>
                </select>
              </div>
            </div>

            {/* Suggested Chips */}
            <div style={{ marginTop: "14px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", alignSelf: "center" }}>Suggestions:</span>
              {suggestedQuestions.map((sq, idx) => (
                <button
                  key={idx}
                  className="btn btn-secondary"
                  style={{
                    padding: "4px 10px",
                    fontSize: "0.74rem",
                    borderRadius: "16px",
                    background: "var(--card-bg)"
                  }}
                  onClick={() => handleSend(sq)}
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>

          {/* Chat Window */}
          <div
            className="card"
            style={{
              minHeight: "440px",
              maxHeight: "580px",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              padding: "20px"
            }}
          >
            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px"
                }}
              >
                <div
                  style={{
                    padding: "14px 18px",
                    borderRadius: "12px",
                    background: msg.sender === "user" ? "var(--accent-blue)" : "var(--card-bg)",
                    color: msg.sender === "user" ? "#ffffff" : "var(--text-primary)",
                    border: msg.sender === "user" ? "none" : "1px solid var(--card-border)",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                    fontSize: "0.9rem",
                    lineHeight: "1.5"
                  }}
                >
                  {/* Markdown formatted content */}
                  <div style={{ whiteSpace: "pre-wrap" }}>{msg.text}</div>

                  {/* Sources Chips */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px solid rgba(255,255,255,0.15)", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                      <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Verified Sources:</span>
                      {msg.sources.map((s, i) => (
                        <span key={i} className="badge badge-info" style={{ fontSize: "0.7rem", padding: "2px 8px" }}>
                          📄 {s.label}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Show Work Tool Trace Toggle */}
                  {msg.toolCalls && msg.toolCalls.length > 0 && (
                    <div style={{ marginTop: "8px" }}>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: "3px 8px", fontSize: "0.7rem", display: "flex", alignItems: "center", gap: "4px" }}
                        onClick={() =>
                          setExpandedTrace((prev) => ({ ...prev, [msg.id]: !prev[msg.id] }))
                        }
                      >
                        <Cpu size={12} />
                        {expandedTrace[msg.id] ? "Hide Tool Calls" : "Show Work (Tool Trace)"}
                        {expandedTrace[msg.id] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      </button>

                      {expandedTrace[msg.id] && (
                        <div
                          style={{
                            marginTop: "8px",
                            padding: "8px 12px",
                            background: "rgba(0,0,0,0.25)",
                            borderRadius: "6px",
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.72rem"
                          }}
                        >
                          {msg.toolCalls.map((tc, idx) => (
                            <div key={idx} style={{ marginBottom: "4px" }}>
                              <span style={{ color: "var(--accent-teal)" }}>{tc.tool}</span>(
                              {JSON.stringify(tc.args)})
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                  {/* Actions under bubble */}
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 4px" }}>
                    <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{msg.timestamp}</span>
                    {msg.sender === "copilot" && (
                      <button
                        className="btn btn-secondary"
                        style={{ padding: "2px 8px", fontSize: "0.68rem" }}
                        onClick={() => downloadAnswer(msg.text)}
                        title="Download Markdown"
                      >
                        <Download size={12} style={{ marginRight: "4px" }} />
                        Export
                      </button>
                    )}
                  </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                <RefreshCw size={16} className="animate-spin" />
                <span>Copilot is querying verified artifacts and validating grounding...</span>
              </div>
            )}
          </div>

          {/* Chat Input Bar */}
          <div style={{ marginTop: "12px", display: "flex", gap: "10px" }}>
            <input
              type="text"
              className="form-input"
              style={{ flex: 1, padding: "12px 16px" }}
              placeholder={`Ask anything about diseases, models, patient risk, image lab, or ML concepts...`}
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
            />
            <button
              className="btn btn-primary"
              style={{ padding: "0 20px", display: "flex", alignItems: "center", gap: "6px" }}
              onClick={() => handleSend()}
              disabled={loading}
            >
              <Send size={16} />
              <span>Send</span>
            </button>
          </div>
        </div>

        {/* Sidebar: Experiment Queue & Verification Status */}
        <div>
          {/* Experiment Proposal Queue */}
          <div className="card" style={{ marginBottom: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
              <h3 style={{ fontSize: "0.95rem", fontWeight: 700 }}>Experiment Queue</h3>
              <span className="badge badge-info" style={{ fontSize: "0.7rem" }}>Budget: 40 Max</span>
            </div>

            {proposals.length === 0 ? (
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                No active experiment proposals. Propose one in <em>Improver</em> mode.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {proposals.map((p) => (
                  <div
                    key={p.proposal_id}
                    style={{
                      padding: "10px",
                      borderRadius: "8px",
                      background: "rgba(255,255,255,0.03)",
                      border: "1px solid var(--card-border)",
                      fontSize: "0.78rem"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                      <strong>{p.cohort}</strong>
                      <span className={`badge ${p.status === "completed" ? "badge-success" : "badge-amber"}`}>
                        {p.status}
                      </span>
                    </div>
                    <div style={{ color: "var(--text-secondary)", marginBottom: "6px" }}>{p.hypothesis}</div>
                    {p.inner_cv_auc && (
                      <div style={{ fontSize: "0.72rem", color: "var(--accent-teal)" }}>
                        Inner-CV AUC: {p.inner_cv_auc} | Bal Acc: {p.inner_cv_balanced_acc}
                      </div>
                    )}
                    {p.status === "pending_approval" && (
                      <div style={{ display: "flex", gap: "6px", marginTop: "8px" }}>
                        <button
                          className="btn btn-success"
                          style={{ padding: "3px 8px", fontSize: "0.7rem", flex: 1 }}
                          onClick={() => handleApproveProposal(p.proposal_id)}
                        >
                          Approve
                        </button>
                        <button
                          className="btn btn-secondary"
                          style={{ padding: "3px 8px", fontSize: "0.7rem", flex: 1 }}
                          onClick={() => handleRejectProposal(p.proposal_id)}
                        >
                          Reject
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Grounding & Integrity Card */}
          <div className="card">
            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "10px" }}>Integrity Safeguards</h3>
            <ul style={{ fontSize: "0.78rem", color: "var(--text-secondary)", paddingLeft: "18px", lineHeight: "1.6" }}>
              <li><strong>Zero Hallucination:</strong> Unmatched numbers flagged with verification notice.</li>
              <li><strong>Passive Data:</strong> Uploads treated strictly as data, disarming prompt injections.</li>
              <li><strong>Inner-CV Sandbox:</strong> Experiments run on dev splits only; outer folds sealed.</li>
              <li><strong>Budget Limit:</strong> Max 40 pre-registered configurations per cohort.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
