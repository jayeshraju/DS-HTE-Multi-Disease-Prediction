import React, { useState, useEffect } from "react";
import { ViewType, Sidebar, Header } from "./components/Navigation";
import { DatasetInfo, RunManifest } from "./types";
import { fetchDatasets, fetchRunMetadata } from "./services/api";

import { OverviewView } from "./views/OverviewView";
import { PerformanceView } from "./views/PerformanceView";
import { ThresholdExplorerView } from "./views/ThresholdExplorerView";
import { CalibrationView } from "./views/CalibrationView";
import { AblationView } from "./views/AblationView";
import { ExplainabilityView } from "./views/ExplainabilityView";
import { TryPatientView } from "./views/TryPatientView";
import { UploadDatasetView } from "./views/imaging/UploadDatasetView";
import { TrainingJobsView } from "./views/imaging/TrainingJobsView";
import { ImageResultsView } from "./views/imaging/ImageResultsView";
import { PredictImageView } from "./views/imaging/PredictImageView";
import { AgentChatView } from "./views/imaging/AgentChatView";
import { CopilotView } from "./views/CopilotView";

export function App() {
  const [currentView, setCurrentView] = useState<ViewType>("overview");
  const [selectedCohort, setSelectedCohort] = useState<string>("pima");
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>("bench_pneumoniamnist");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [datasets, setDatasets] = useState<Record<string, DatasetInfo>>({});
  const [manifest, setManifest] = useState<RunManifest | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    fetchDatasets()
      .then((data) => setDatasets(data))
      .catch((err) => console.error("Error loading datasets:", err));

    fetchRunMetadata()
      .then((m) => setManifest(m))
      .catch((err) => console.error("Error loading manifest:", err));
  }, []);

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  const handleSelectCohortFromOverview = (cohortKey: string) => {
    setSelectedCohort(cohortKey);
    setCurrentView("performance");
  };

  const handleStartTrainingFromUpload = (datasetId: string) => {
    setActiveDatasetId(datasetId);
    setCurrentView("image-jobs");
  };

  const handleSelectJobResults = (jobId: string) => {
    setActiveJobId(jobId);
    setCurrentView("image-results");
  };

  return (
    <div className="app-container">
      <Sidebar
        currentView={currentView}
        onSelectView={setCurrentView}
        manifest={manifest}
        theme={theme}
        onToggleTheme={handleToggleTheme}
      />

      <div className="main-content">
        <Header
          currentView={currentView}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          manifest={manifest}
        />

        <main className="page-body">
          {currentView === "overview" && (
            <OverviewView
              datasets={datasets}
              manifest={manifest}
              onSelectCohort={handleSelectCohortFromOverview}
            />
          )}

          {currentView === "performance" && (
            <PerformanceView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "threshold" && (
            <ThresholdExplorerView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "calibration" && (
            <CalibrationView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "ablation" && (
            <AblationView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "explain" && (
            <ExplainabilityView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "patient" && (
            <TryPatientView
              datasets={datasets}
              selectedCohort={selectedCohort}
              onSelectCohort={setSelectedCohort}
            />
          )}

          {currentView === "copilot" && (
            <CopilotView />
          )}

          {/* IMAGE LAB VIEWS */}
          {currentView === "image-upload" && (
            <UploadDatasetView onStartTraining={handleStartTrainingFromUpload} />
          )}

          {currentView === "image-jobs" && (
            <TrainingJobsView
              initialDatasetId={activeDatasetId}
              onSelectJobResults={handleSelectJobResults}
            />
          )}

          {currentView === "image-results" && (
            <ImageResultsView selectedJobId={activeJobId} />
          )}

          {currentView === "image-predict" && (
            <PredictImageView selectedModelId={activeJobId} />
          )}

          {currentView === "image-agent" && (
            <AgentChatView selectedJobId={activeJobId} />
          )}

          {/* Persistent Clinical Disclaimer Footer on every view */}
          <footer style={{ marginTop: "40px", padding: "16px", borderTop: "1px solid var(--card-border)", textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            <span style={{ fontWeight: 700, color: "var(--accent-teal)" }}>Research prototype, not a diagnostic device.</span>
            {" "}&bull; Developed for clinical machine learning integrity evaluation. Not certified for independent medical diagnosis or clinical treatment prescription.
          </footer>
        </main>
      </div>
    </div>
  );
}

export default App;
