export type ProblemType = "classification" | "regression" | "unsupervised";

export interface DatasetPreview {
  dataset_id: string;
  filename: string;
  columns: string[];
  row_count: number;
  preview: Record<string, string | number>[];
}

export interface UploadResponse extends DatasetPreview {
  suggested_target?: string;
  target_reason?: string;
  suggested_problem_type?: ProblemType;
  confidence?: number;
}

export interface DetectTargetResponse {
  dataset_id: string;
  target_column: string;
  confidence: number;
  problem_type: ProblemType;
  reason: string;
}

export interface MetricPayload {
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  rmse?: number | null;
  mae?: number | null;
  r2?: number | null;
  silhouette?: number | null;
}

export interface ArtifactPayload {
  confusion_matrix?: number[][] | null;
  residuals?: number[] | null;
  roc?: {
    fpr: number[];
    tpr: number[];
  } | null;
  plots?: Record<string, string | null> | null;
}

export interface FeatureHint {
  name: string;
  kind: "numeric" | "categorical";
  examples: string[];
  value_map?: Record<string, number> | null;
}

export interface ModelSummary {
  model_id: string;
  model_key: string;
  model_name: string;
  dataset_id: string;
  problem_type: ProblemType;
  metrics: MetricPayload;
  rank?: number | null;
  created_at: string;
  download_url?: string | null;
  download_h5_url?: string | null;
  report_url?: string | null;
  artifacts?: ArtifactPayload | null;
  columns: string[];
  feature_hints: FeatureHint[];
}

export interface ImageUploadResponse {
  dataset_id: string;
  filename: string;
  mode: "supervised" | "unsupervised";
  total_images: number;
  classes: { label: string; count: number }[];
  message: string;
}

export interface ClassSampleImage {
  class_label: string;
  image_b64: string;
}

export interface DatasetSampleImagesResponse {
  samples: ClassSampleImage[];
}

export interface ModelCatalogEntry {
  key: string;
  name: string;
  problem_types: ProblemType[];
  dataset_modes?: string[];
}

export interface TrainResponse {
  message: string;
  model: ModelSummary;
  artifacts: ArtifactPayload;
}

export interface TrainAllResponse {
  message: string;
  leaderboard?: ModelSummary[]; // Optional - not displayed in UI
  report_id?: string | null;
}

export interface PredictResponse {
  model_id: string;
  predictions: (string | number)[];
}

export interface TrainingHistoryRecord {
  history_id: string;
  user_id: string;
  dataset_id: string;
  dataset_name: string;
  dataset_mode?: string; // csv | supervised | unsupervised
  target_column: string;
  problem_type: string;
  model_key?: string | null;
  model_name?: string | null;
  model_id?: string | null; // Link to model for predictions
  intensity: string;
  metrics: {
    accuracy?: number | null;
    precision?: number | null;
    recall?: number | null;
    f1?: number | null;
    rmse?: number | null;
    r2?: number | null;
  };
  created_at: string;
  report_id?: string | null;
  pkl_path?: string | null; // Path to .pkl file
  h5_path?: string | null; // Path to .h5 file
}

export interface TrainingHistoryResponse {
  history: TrainingHistoryRecord[];
}

export interface DatasetSummary {
  dataset_id: string;
  filename: string;
  mode: string; // csv | supervised | unsupervised
  row_count?: number | null;
  total_images?: number | null;
  columns: string[];
  created_at?: string | null;
}

export interface DatasetListResponse {
  datasets: DatasetSummary[];
}

export interface ModelInfoEntry {
  key: string;
  name: string;
  description: string;
  problem_types: ProblemType[];
  dataset_modes: string[];
  advantages: string[];
  disadvantages: string[];
  use_cases: string[];
  complexity: "Low" | "Medium" | "High";
  training_speed: "Fast" | "Medium" | "Slow" | "N/A (lazy learning)";
  interpretability: "High" | "Medium" | "Low";
}

export interface ModelInfoListResponse {
  models: ModelInfoEntry[];
}

export interface ModelDetailResponse extends ModelInfoEntry {
  how_it_works: string;
  algorithm_steps: string[];
  hyperparameters: Record<string, string>;
  visualization_type: string;
}

