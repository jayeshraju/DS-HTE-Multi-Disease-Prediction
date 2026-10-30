export interface RunManifest {
  run_id: string;
  timestamp: string;
  git_commit: string;
  config_hash: string;
  profile: string;
  hardware: {
    platform: string;
    cpu_count: number;
  };
  dataset_hashes?: Record<string, string>;
}

export interface DatasetInfo {
  name: string;
  key: string;
  n_samples: number;
  n_features: number;
  n_positives?: number;
  n_negatives?: number;
  prevalence: number;
  majority_baseline_rate?: number;
  feature_names: string[];
}

export interface MetricCI {
  mean: number;
  std: number;
  ci_lower: number;
  ci_upper: number;
}

export type ModelMetricsMap = Record<string, MetricCI>;

export interface CurvePoint {
  fpr?: number;
  tpr?: number;
  recall?: number;
  precision?: number;
  prob_pred?: number;
  prob_true?: number;
  threshold?: number;
  net_benefit?: number;
  treat_all?: number;
  treat_none?: number;
}

export interface ConfusionData {
  cohort: string;
  model: string;
  threshold: number;
  default_threshold: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  total: number;
  sensitivity: number;
  specificity: number;
  precision: number;
  ppv: number;
  npv: number;
  accuracy: number;
  balanced_accuracy: number;
  f1: number;
  net_benefit: number;
}

export interface ModelComparisonItem {
  model: string;
  metric: string;
  mean: number;
  std: number;
  ci_lower: number;
  ci_upper: number;
  delta_vs_dshte: number;
}

export interface StatTestRecord {
  cohort: string;
  baseline: string;
  wilcoxon_stat?: number;
  wilcoxon_p?: number;
  nadeau_t?: number;
  nadeau_p?: number;
  mcnemar_stat?: number;
  mcnemar_p?: number;
  delong_z?: number;
  delong_p?: number;
  holm_bonferroni_p?: number;
  bh_fdr_p?: number;
  direction?: string;
}

export interface StabilityRecord {
  cohort: string;
  feature?: string;
  selection_freq?: number;
  jaccard_mean?: number;
  kuncheva_mean?: number;
}

export interface GlobalFeatureItem {
  rank: number;
  feature: string;
  mean_abs_shap: number;
  lime_weight: number;
  stability_freq: number;
}

export interface PaperComparisonItem {
  paper: {
    accuracy: number;
    f1: number;
    mcc: number;
    sensitivity: number;
  };
  cv: {
    accuracy: number;
    f1: number;
    mcc: number;
    sensitivity: number;
  };
  holdout: {
    accuracy: number;
    f1: number;
    mcc: number;
    sensitivity: number;
  };
}

export interface SubgroupErrorProfile {
  attribute: string;
  subgroup: string;
  n_samples: number;
  fnr: number;
  fnr_ci_lower: number;
  fnr_ci_upper: number;
  disparity_flag: boolean;
}
