import {
  RunManifest, DatasetInfo, ModelMetricsMap, ConfusionData,
  ModelComparisonItem, StatTestRecord, StabilityRecord,
  GlobalFeatureItem, PaperComparisonItem
} from "../types";

const API_BASE = "";

export async function fetchRunMetadata(): Promise<RunManifest> {
  const res = await fetch(`${API_BASE}/api/run`);
  if (!res.ok) throw new Error("Failed to fetch run metadata");
  return res.json();
}

export async function fetchDatasets(): Promise<Record<string, DatasetInfo>> {
  const res = await fetch(`${API_BASE}/api/datasets`);
  if (!res.ok) throw new Error("Failed to fetch datasets");
  return res.json();
}

export async function fetchMetrics(dataset?: string, model?: string, view: string = "cv"): Promise<Record<string, ModelMetricsMap>> {
  const params = new URLSearchParams();
  if (dataset) params.append("dataset", dataset);
  if (model) params.append("model", model);
  params.append("view", view);
  const res = await fetch(`${API_BASE}/api/metrics?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

export async function fetchCurves(dataset: string, model: string, type?: string): Promise<any> {
  const params = new URLSearchParams({ dataset, model });
  if (type) params.append("type", type);
  const res = await fetch(`${API_BASE}/api/curves?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch curves");
  return res.json();
}

export async function fetchConfusion(dataset: string, model: string = "DS-HTE", threshold: number = 0.5): Promise<ConfusionData> {
  const params = new URLSearchParams({
    dataset,
    model,
    threshold: threshold.toString()
  });
  const res = await fetch(`${API_BASE}/api/confusion?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch confusion data");
  return res.json();
}

export async function fetchComparison(dataset: string, metric: string = "accuracy", view: string = "cv"): Promise<ModelComparisonItem[]> {
  const params = new URLSearchParams({ dataset, metric, view });
  const res = await fetch(`${API_BASE}/api/compare?${params.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch comparison");
  return res.json();
}

export async function fetchStats(dataset?: string): Promise<StatTestRecord[]> {
  const url = dataset ? `${API_BASE}/api/stats?dataset=${dataset}` : `${API_BASE}/api/stats`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

export async function fetchAblation(dataset?: string): Promise<any> {
  const url = dataset ? `${API_BASE}/api/ablation?dataset=${dataset}` : `${API_BASE}/api/ablation`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch ablation");
  return res.json();
}

export async function fetchStability(dataset?: string): Promise<StabilityRecord[]> {
  const url = dataset ? `${API_BASE}/api/stability?dataset=${dataset}` : `${API_BASE}/api/stability`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch stability");
  return res.json();
}

export async function fetchGlobalExplain(dataset: string): Promise<{ cohort: string; features: GlobalFeatureItem[]; consensus_agreement: number }> {
  const res = await fetch(`${API_BASE}/api/explain/global?dataset=${dataset}`);
  if (!res.ok) throw new Error("Failed to fetch global explanation");
  return res.json();
}

export async function predictPatient(cohort: string, patientId: string, features: Record<string, number>): Promise<any> {
  const res = await fetch(`${API_BASE}/api/predict/${cohort}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId, features })
  });
  if (!res.ok) throw new Error("Inference failed");
  return res.json();
}

export async function explainPatient(cohort: string, patientId: string, features: Record<string, number>): Promise<any> {
  const res = await fetch(`${API_BASE}/api/explain/${cohort}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId, features })
  });
  if (!res.ok) throw new Error("Patient explanation failed");
  return res.json();
}

export async function fetchPaperVsReproduced(): Promise<Record<string, PaperComparisonItem>> {
  const res = await fetch(`${API_BASE}/api/paper-vs-reproduced`);
  if (!res.ok) throw new Error("Failed to fetch paper vs reproduced");
  return res.json();
}

export async function fetchAudit(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/audit`);
  if (!res.ok) throw new Error("Failed to fetch audit");
  return res.json();
}

export async function fetchSubgroups(dataset?: string): Promise<any> {
  const url = dataset ? `${API_BASE}/api/subgroups?dataset=${dataset}` : `${API_BASE}/api/subgroups`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch subgroups");
  return res.json();
}
