# System Architecture & Technical Specifications

> **System Notice**: Research prototype, not a certified clinical diagnostic device.

---

## 1. High-Level System Architecture

```mermaid
graph TD
    UI["React SPA Dashboard (Vite + TS)"]
    API["FastAPI REST & SSE Service (Python 3.13)"]
    
    subgraph "Core Orchestration & Decision Modules"
        NCV["50-Fold Nested Repeated CV Engine"]
        AFO["Dynamic Feature Optimizer (AFO)"]
        DSHTE["DS-HTE Stacking Meta-Learner (SLSQP)"]
        CONF["Split-Conformal Prediction Engine"]
    end

    subgraph "Medical Vision & Multi-Modal"
        IM_INGEST["Image Ingestion & PHI Stripper"]
        IM_MODELS["Frozen PyTorch Backbones (ResNet-18)"]
        GRADCAM["Grad-CAM Saliency Engine"]
        OOD["Mahalanobis Distance OOD Detector"]
    end

    subgraph "DS-HTE AI Copilot (Gate 1)"
        COP_ORCH["Deterministic State Machine Orchestrator"]
        COP_TOOLS["16 Typed Allow-Listed Tools"]
        COP_VERIF["Grounding Verifier (Zero Hallucination)"]
        COP_IMPR["Sandboxed Dev Improver (Inner CV)"]
    end

    subgraph "Single Source of Truth Artifact Store"
        DATA_STORE["Parquet Stores & Results Tables"]
        PROV_STORE["*.provenance.json Sidecar Vault"]
        AUDIT_LOGS["Audit Logs & Experiment Registers"]
    end

    UI <-->|HTTP REST / SSE| API
    API --> NCV
    API --> IM_INGEST
    API --> COP_ORCH
    NCV --> AFO
    AFO --> DSHTE
    DSHTE --> CONF
    IM_INGEST --> IM_MODELS
    IM_MODELS --> GRADCAM
    IM_MODELS --> OOD
    COP_ORCH --> COP_TOOLS
    COP_TOOLS --> COP_VERIF
    COP_ORCH --> COP_IMPR
    
    NCV --> DATA_STORE
    IM_MODELS --> DATA_STORE
    DATA_STORE --> PROV_STORE
```

---

## 2. End-to-End Data Flow & Ingestion

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinical Researcher
    participant UI as Dashboard Interface
    participant API as FastAPI Backend
    participant Ingest as Data Ingestion & Sanitizer
    participant Sandbox as Inner-CV Sandbox
    participant Model as DS-HTE Stacking Engine
    participant Store as Artifact Store & Provenance

    Clinician->>UI: Uploads Patient CSV / DICOM Slice
    UI->>API: POST /api/predict or /api/image/datasets
    API->>Ingest: Sanitize & Strip PHI (DICOM Tags, Path Safety)
    Ingest-->>API: Sanitized Feature Array
    API->>Model: Run Calibrated Inference
    Model->>Model: Compute Non-Conformity & Conformal Prediction Set
    Model-->>API: Return Calibrated Prob, Risk Tier & Conformal Set
    API-->>UI: Display Decision with Uncertainty Flag
    Clinician->>UI: Query AI Copilot ("Why is this sample uncertain?")
    UI->>API: POST /api/copilot/ask
    API->>Sandbox: Execute Tool on Staged Artifacts
    Sandbox-->>API: Grounded Tool Result
    API-->>UI: Verified Response with Source Chips
```

---

## 3. 50-Fold Nested Repeated Cross-Validation Protocol

```mermaid
flowchart LR
    subgraph "Dataset Stage (N Instances)"
        D[Complete Clinical Cohort] --> H[20% Isolated Holdout]
        D --> DEV[80% Development Data]
    end

    subgraph "Outer Loop (10 Repeats x 5 Folds = 50 Folds)"
        DEV --> F1[Outer Fold 1..50]
        F1 --> TR[Outer Train 80%]
        F1 --> TE[Outer Test 20%]
    end

    subgraph "Inner Validation Loop (3 Repeats x 3 Folds)"
        TR --> ITR[Inner Train]
        TR --> IVAL[Inner Validation]
        ITR --> HYP[Hyperparameter Selection & AFO]
        IVAL --> CALIB[Calibration & Youden-J Threshold]
    end

    subgraph "Evaluation (Frozen Model)"
        CALIB --> EVAL[Apply to Outer Test Once]
        EVAL --> PARQUET[(outer_fold_metrics.parquet)]
    end
```

---

## 4. AI Copilot Orchestrator Architecture

```mermaid
stateDiagram-v2
    [*] --> SecurityCheck: User Prompt
    SecurityCheck --> PromptInjectionDetected: Malicious Input
    PromptInjectionDetected --> [*]: Passive Data Advisory Notice
    SecurityCheck --> DiagnosisCheck: Safe Input
    DiagnosisCheck --> RefusalResponse: Medical Diagnosis / Treatment Requested
    RefusalResponse --> [*]: Clinical Disclaimer & Referral
    DiagnosisCheck --> PlanMode: Legitimate Query
    
    PlanMode --> ExecuteTools: Explain / Analyst / Improver / Reporter
    ExecuteTools --> SynthesizeDraft: Tool Outputs Gathered
    SynthesizeDraft --> GroundingVerification: Verify Numerical Claims
    GroundingVerification --> FlagUngrounded: Number Not In Artifact Store
    GroundingVerification --> FinalAnswer: 100% Grounded
    FlagUngrounded --> FinalAnswer: Append Audit Disclaimer
    FinalAnswer --> [*]: Return Answer + Source Chips + Tool Trace
```
