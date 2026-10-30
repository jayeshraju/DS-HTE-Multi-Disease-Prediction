import React, { useState } from "react";
import { Bot, Send, User, ShieldAlert, Tag, HelpCircle, CheckCircle } from "lucide-react";

interface Message {
  sender: "user" | "agent";
  text: string;
  citations?: string[];
  disclaimer?: string;
  timestamp: string;
}

export const AgentChatView: React.FC<{
  selectedJobId?: string | null;
}> = ({ selectedJobId }) => {
  const [jobId, setJobId] = useState<string>(selectedJobId || "bench_pneumoniamnist");
  const [inputQuestion, setInputQuestion] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "agent",
      text: "Hello. I am the Medical Imaging Analysis Agent. I can analyze your model's performance metrics, explain confusion matrix patterns, inspect data saturation via learning curves, and interpret Grad-CAM saliency. All my responses are grounded in computed artifacts. How can I assist?",
      citations: ["[table:metrics]", "[figure:learning_curve]"],
      disclaimer: "Research prototype, not a diagnostic device.",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  const suggestedQuestions = [
    "What would improve model accuracy?",
    "Why are misclassifications occurring?",
    "Is the training data saturated?",
    "Can you diagnose this image?"
  ];

  const handleSend = async (qToSend?: string) => {
    const query = qToSend || inputQuestion;
    if (!query.trim()) return;

    const userMsg: Message = {
      sender: "user",
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputQuestion("");
    setLoading(true);

    try {
      const res = await fetch("/api/agent/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          question: query
        })
      });
      if (res.ok) {
        const data = await res.json();
        const agentMsg: Message = {
          sender: "agent",
          text: data.answer,
          citations: data.citations,
          disclaimer: data.disclaimer,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages((prev) => [...prev, agentMsg]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 180px)", gap: "16px" }}>
      {/* Header bar */}
      <div className="glass-card" style={{ padding: "14px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "var(--accent-primary)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff" }}>
            <Bot size={18} />
          </div>
          <div>
            <h4 style={{ fontSize: "0.95rem", fontWeight: 700, margin: 0 }}>Grounded Analysis Agent</h4>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Target Model: {jobId}</span>
          </div>
        </div>

        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          <span className="badge badge-success" style={{ marginRight: "8px" }}>Artifact-Grounded</span>
          <span>Research prototype, not a diagnostic device.</span>
        </div>
      </div>

      {/* Messages stream */}
      <div
        className="glass-card"
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "20px",
          display: "flex",
          flexDirection: "column",
          gap: "16px"
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: msg.sender === "user" ? "flex-end" : "flex-start",
              maxWidth: "85%",
              alignSelf: msg.sender === "user" ? "flex-end" : "flex-start"
            }}
          >
            <div
              style={{
                background: msg.sender === "user" ? "var(--accent-primary)" : "var(--bg-tertiary)",
                color: msg.sender === "user" ? "#ffffff" : "var(--text-primary)",
                padding: "12px 16px",
                borderRadius: "10px",
                fontSize: "0.88rem",
                lineHeight: 1.5,
                border: msg.sender === "user" ? "none" : "1px solid var(--card-border)"
              }}
            >
              {msg.text}

              {/* Citations Chips */}
              {msg.citations && msg.citations.length > 0 && (
                <div style={{ marginTop: "10px", display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {msg.citations.map((cite, cIdx) => (
                    <span
                      key={cIdx}
                      style={{
                        fontSize: "0.7rem",
                        padding: "2px 8px",
                        borderRadius: "4px",
                        background: "rgba(2, 132, 199, 0.2)",
                        color: "var(--accent-primary)",
                        fontWeight: 600,
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "4px"
                      }}
                    >
                      <Tag size={10} /> {cite}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "4px" }}>
              {msg.timestamp}
            </span>
          </div>
        ))}

        {loading && (
          <div style={{ alignSelf: "flex-start", fontSize: "0.8rem", color: "var(--text-muted)", fontStyle: "italic" }}>
            Agent analyzing artifacts...
          </div>
        )}
      </div>

      {/* Suggested prompts */}
      <div style={{ display: "flex", gap: "8px", overflowX: "auto", padding: "4px 0" }}>
        {suggestedQuestions.map((q, idx) => (
          <button
            key={idx}
            className="btn btn-secondary"
            onClick={() => handleSend(q)}
            style={{ fontSize: "0.78rem", padding: "6px 12px", whiteSpace: "nowrap" }}
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input box */}
      <div style={{ display: "flex", gap: "10px" }}>
        <input
          type="text"
          className="select-input"
          value={inputQuestion}
          onChange={(e) => setInputQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask the analysis agent about metrics, confusion matrix, or data saturation..."
          style={{
            flex: 1,
            padding: "12px 16px",
            background: "var(--bg-secondary)",
            border: "1px solid var(--card-border)",
            borderRadius: "8px",
            color: "var(--text-primary)"
          }}
        />
        <button
          className="btn btn-primary"
          onClick={() => handleSend()}
          disabled={loading || !inputQuestion.trim()}
          style={{ display: "flex", alignItems: "center", gap: "6px" }}
        >
          <Send size={16} />
          <span>Ask</span>
        </button>
      </div>
    </div>
  );
};
