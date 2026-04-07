import { useEffect, useState } from "react";
import { useNavigate, Link, useParams, useLocation } from "react-router-dom";
import axios from "axios";
import toast from "react-hot-toast";
import { useAuth } from "./contexts/AuthContext";
import type {
  UploadResponse,
  DetectTargetResponse,
  ModelCatalogEntry,
  ModelSummary,
  TrainResponse,
  TrainAllResponse,
  PredictResponse,
  TrainingHistoryRecord,
  ImageUploadResponse,
  DatasetSummary,
  DatasetListResponse,
  DatasetSampleImagesResponse,
  ClassSampleImage,
} from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const metricLabel = (value?: number | null) =>
  value === undefined || value === null ? "—" : value.toFixed(3);

const fileToBase64 = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

const MODE_TIPS: Record<
  "csv" | "supervised",
  { title: string; bullets: string[]; helper: string }
> = {
  csv: {
    title: "CSV mode — tabular data",
    bullets: [
      "Upload a single .csv file (≤ 300 MB)",
      "Include one target/label column (we can auto-detect)",
      "No merged header rows; UTF-8 encoding works best",
    ],
    helper: "Need examples? Target column like price/status/label works great.",
  },
  supervised: {
    title: "Images with Labelled Dataset — classification",
    bullets: [
      "Upload a .zip with one folder per class (cats/, dogs/, etc.)",
      "Each class folder must contain at least 1 JPG/PNG",
      "Total zip size ≤ 300 MB; no stray non-image files",
    ],
    helper: "Tip: folder names are used as class labels.",
  },
};

export default function App() {
  const { user, loading: authLoading, refreshUser } = useAuth();
  const navigate = useNavigate();
  const { mode } = useParams<{ mode?: "csv" | "supervised" }>();
  const location = useLocation();
  const [uploading, setUploading] = useState(false);
  const [datasetMode, setDatasetMode] = useState<"csv" | "supervised">(mode === "supervised" ? "supervised" : "csv");
  const [dataset, setDataset] = useState<UploadResponse | null>(null);
  const [imageDataset, setImageDataset] = useState<ImageUploadResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [detectInfo, setDetectInfo] = useState<DetectTargetResponse | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalogEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [trainingIntensity, setTrainingIntensity] = useState<"less" | "medium" | "rigorous">("medium");
  const [predictModalModel, setPredictModalModel] = useState<ModelSummary | null>(null);
  const [predictionInputs, setPredictionInputs] = useState<Record<string, string>>({});
  const [predictResult, setPredictResult] = useState<string | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryRecord[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isAILoading, setIsAILoading] = useState(false);
  const [recentDatasets, setRecentDatasets] = useState<DatasetSummary[]>([]);
  const [showRecentDatasets, setShowRecentDatasets] = useState(false);
  const [sampleImages, setSampleImages] = useState<ClassSampleImage[]>([]);
  const [initialDatasetLoaded, setInitialDatasetLoaded] = useState(false);
  const currentModeTips = MODE_TIPS[datasetMode];

  // Fetch training history
  const fetchTrainingHistory = async () => {
    if (!user) return;
    try {
      const token = localStorage.getItem("token");
      if (!token) return;
      const res = await axios.get(`${API_BASE}/training-history`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      const history = res.data.history || [];
      // Ensure all records have dataset_mode for backward compatibility
      const normalizedHistory = history.map((record: TrainingHistoryRecord) => ({
        ...record,
        dataset_mode: record.dataset_mode || "csv",
      }));
      console.log("Fetched training history:", normalizedHistory.length, "records");
      setTrainingHistory(normalizedHistory);
    } catch (err) {
      console.error("Failed to fetch training history:", err);
    }
  };

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      navigate("/login");
    }
  }, [user, authLoading, navigate]);

  // Sync mode from URL
  useEffect(() => {
    if (mode && mode !== datasetMode) {
      setDatasetMode(mode);
      setDataset(null);
      setImageDataset(null);
      setTargetColumn(mode === "csv" ? "" : "image_label");
    }
  }, [mode]);

  // Fetch initial user data and history when component mounts
  useEffect(() => {
    if (user && !authLoading) {
      // Only fetch once when user is first available, not on every user update
      fetchTrainingHistory();
      
      // Restore dataset from localStorage on mount/refresh
      try {
        const saved = localStorage.getItem(`current_dataset_${datasetMode}`);
        if (saved) {
          const datasetInfo = JSON.parse(saved);
          if (datasetInfo.mode === datasetMode) {
            // For CSV, we'd need to fetch the dataset preview, but for now just restore the ID
            // The history will still show because we filter by dataset_id
            // For image datasets, we can't easily restore without re-uploading
            if (datasetMode === "csv") {
              // We'll let the user re-upload or the history will show based on mode
            }
          }
        }
      } catch (e) {
        console.error("Failed to restore dataset from localStorage:", e);
      }
    }
  }, [user?.user_id, datasetMode]); // Include datasetMode to restore when mode changes

  // If we arrive with ?datasetId=... in the URL (e.g. from report page), auto-load that dataset
  useEffect(() => {
    if (!user || authLoading || initialDatasetLoaded) return;

    const searchParams = new URLSearchParams(location.search);
    const datasetIdFromQuery = searchParams.get("datasetId");
    if (!datasetIdFromQuery) return;

    const loadFromQuery = async () => {
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          toast.error("Authentication required");
          return;
        }
        if (datasetMode === "csv") {
          const res = await axios.get<UploadResponse>(`${API_BASE}/datasets/${datasetIdFromQuery}/load`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          setDataset(res.data);
          setImageDataset(null);
          setTargetColumn(res.data.suggested_target ?? "");
          setDetectInfo(null);
          localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
            dataset_id: res.data.dataset_id,
            filename: res.data.filename,
            mode: datasetMode,
          }));
          toast.success("Dataset loaded");
        } else {
          const res = await axios.get<ImageUploadResponse>(`${API_BASE}/datasets/${datasetIdFromQuery}/load-image`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          setImageDataset(res.data);
          setDataset(null);
          setDetectInfo(null);
          setTargetColumn("image_label");
          localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
            dataset_id: res.data.dataset_id,
            filename: res.data.filename,
            mode: datasetMode,
          }));
          toast.success("Image dataset loaded");
        }
        // After loading dataset, refresh history for this mode
        setTimeout(() => {
          fetchTrainingHistory().catch(console.error);
        }, 200);
      } catch (err: any) {
        toast.error(err?.response?.data?.detail ?? "Failed to load dataset from report");
      } finally {
        setInitialDatasetLoaded(true);
      }
    };

    loadFromQuery().catch(console.error);
  }, [user, authLoading, initialDatasetLoaded, location.search, datasetMode]);

  // Refetch history when dataset changes
  useEffect(() => {
    if (user && !authLoading) {
      const currentId = dataset?.dataset_id || imageDataset?.dataset_id;
      if (currentId) {
        // Refetch history when dataset ID changes
        fetchTrainingHistory().catch(console.error);
      }
    }
  }, [dataset?.dataset_id, imageDataset?.dataset_id, user?.user_id]);


  const fetchCatalog = async () => {
    const res = await axios.get(`${API_BASE}/models/catalog`, {
      params: { dataset_mode: datasetMode },
    });
    setCatalog(res.data.models);
    if (!selectedModel && res.data.models.length) {
      setSelectedModel(res.data.models[0].key);
    }
  };

  useEffect(() => {
    fetchCatalog().catch(console.error);
  }, [datasetMode]);

  // Fetch recent datasets
  const fetchRecentDatasets = async () => {
    if (!user) return;
    try {
      const token = localStorage.getItem("token");
      if (!token) return;
      const res = await axios.get<DatasetListResponse>(`${API_BASE}/datasets`, {
        params: { mode: datasetMode },
        headers: { Authorization: `Bearer ${token}` },
      });
      setRecentDatasets(res.data.datasets);
    } catch (err) {
      console.error("Failed to fetch recent datasets:", err);
    }
  };

  // Fetch sample images for image datasets
  const fetchSampleImages = async (datasetId: string) => {
    if (!user) return;
    try {
      const token = localStorage.getItem("token");
      if (!token) return;
      const res = await axios.get<DatasetSampleImagesResponse>(`${API_BASE}/datasets/${datasetId}/sample-images`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setSampleImages(res.data.samples);
    } catch (err) {
      console.error("Failed to fetch sample images:", err);
      setSampleImages([]);
    }
  };

  useEffect(() => {
    if (user && !authLoading) {
      fetchRecentDatasets().catch(console.error);
    }
  }, [user?.user_id, datasetMode]);

  // Fetch sample images when imageDataset is loaded
  useEffect(() => {
    if (imageDataset?.dataset_id) {
      fetchSampleImages(imageDataset.dataset_id).catch(console.error);
    } else {
      setSampleImages([]);
    }
  }, [imageDataset?.dataset_id]);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.length) return;
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    setUploading(true);
    try {
      if (datasetMode === "csv") {
        const res = await axios.post<UploadResponse>(`${API_BASE}/upload`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setDataset(res.data);
        setImageDataset(null);
        setTargetColumn(res.data.suggested_target ?? "");
        setDetectInfo(null);
        // Persist dataset info to localStorage for refresh persistence
        localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
          dataset_id: res.data.dataset_id,
          filename: res.data.filename,
          mode: datasetMode,
        }));
        toast.success("CSV dataset uploaded");
        // Refetch history and recent datasets after state update
        setTimeout(() => {
          fetchTrainingHistory().catch(console.error);
          fetchRecentDatasets().catch(console.error);
        }, 100);
      } else {
        const endpoint =
          `${API_BASE}/upload/images/supervised`;
        const res = await axios.post<ImageUploadResponse>(endpoint, formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setImageDataset(res.data);
        setDataset(null);
        setDetectInfo(null);
        setTargetColumn("image_label");
        // Persist dataset info to localStorage for refresh persistence
        localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
          dataset_id: res.data.dataset_id,
          filename: res.data.filename,
          mode: datasetMode,
        }));
        toast.success(res.data.message || "Image dataset uploaded");
        // Refetch history and recent datasets after state update
        setTimeout(() => {
          fetchTrainingHistory().catch(console.error);
          fetchRecentDatasets().catch(console.error);
        }, 100);
      }
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to upload dataset");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const ensureDataset = () => {
    if (!dataset && !imageDataset) throw new Error("Upload a dataset first.");
  };

  const runDetectTarget = async () => {
    setIsAILoading(true);
    try {
      ensureDataset();
      const res = await axios.post<DetectTargetResponse>(`${API_BASE}/detect-target`, {
        dataset_id: dataset!.dataset_id,
      });
      setDetectInfo(res.data);
      setTargetColumn(res.data.target_column);
      toast.success(`Detected target: ${res.data.target_column}`);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Detection failed");
    } finally {
      setIsAILoading(false);
    }
  };

  const runTrain = async (endpoint: "train" | "train-all", body: Record<string, unknown>) => {
    setIsTraining(true);
    try {
      ensureDataset();
    const url = `${API_BASE}/${endpoint}`;
    const res = await axios.post<TrainResponse | TrainAllResponse>(url, body);
      if (endpoint === "train") {
        toast.success("Model trained");
      } else {
        const payload = res.data as TrainAllResponse;
        toast.success("All models trained");
        if (payload.report_id) {
          toast.success("PDF report generated and ready for download");
        }
      }
      // Refresh user info and training history
      refreshUser().catch(console.error);
      // Refetch history after training to show the new training record
      // Use setTimeout to ensure backend has saved the history
      setTimeout(() => {
        fetchTrainingHistory().catch(console.error);
      }, 1000); // Increased timeout to ensure backend has saved the history
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Training failed");
    } finally {
      setIsTraining(false);
    }
  };

  const currentDatasetId = () => dataset?.dataset_id || imageDataset?.dataset_id;

  const handleTrainSingle = async () => {
    if (!canTrain) {
      toast.error("No free runs remaining. Please subscribe to continue.");
      return;
    }
    const dsId = currentDatasetId();
    if (!dsId) {
      toast.error("Upload a dataset first.");
      return;
    }
    const modelForMode =
      datasetMode === "supervised" ? "image_cnn" : selectedModel;
    if (!modelForMode) {
      toast.error("Select a model first.");
      return;
    }
    const targetForMode =
      datasetMode === "supervised" ? "image_label" : targetColumn;
    await runTrain("train", {
      dataset_id: dsId,
      target_column: targetForMode,
      model_key: modelForMode,
      intensity: trainingIntensity,
    });
  };

  const handleTrainAll = async () => {
    if (!canTrain) {
      toast.error("No free runs remaining. Please subscribe to continue.");
      return;
    }
    const dsId = currentDatasetId();
    if (!dsId) {
      toast.error("Upload a dataset first.");
      return;
    }
    const targetForMode =
      datasetMode === "supervised" ? "image_label" : targetColumn;
    await runTrain("train-all", {
      dataset_id: dsId,
      target_column: targetForMode,
      intensity: trainingIntensity,
    });
  };

  const runAutoSelect = async () => {
    if (!canTrain) {
      toast.error("No free runs remaining. Please subscribe to continue.");
      return;
    }
    setIsAILoading(true);
    try {
      ensureDataset();
      const res = await axios.post(`${API_BASE}/auto-select`, {
        dataset_id: currentDatasetId(),
        target_column: targetColumn || undefined,
      });
      toast.success(res.data.recommended_model);
      // Refresh user info in the background (non-blocking)
      refreshUser().catch(console.error);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "AI selection failed");
    } finally {
      setIsAILoading(false);
    }
  };

  const runAutoEverything = async () => {
    if (!canTrain) {
      toast.error("No free runs remaining. Please subscribe to continue.");
      return;
    }
    setIsTraining(true);
    try {
      ensureDataset();
      const res = await axios.post(`${API_BASE}/auto-everything`, {
        dataset_id: currentDatasetId(),
      });
      setTargetColumn(res.data.target_column);
      toast.success("AI completed the full workflow");
      // Refresh user info and training history
      refreshUser().catch(console.error);
      // Refetch history after training to show the new training record
      // Use setTimeout to ensure backend has saved the history
      setTimeout(() => {
        fetchTrainingHistory().catch(console.error);
      }, 1000); // Increased timeout to ensure backend has saved the history
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Auto-everything failed");
    } finally {
      setIsTraining(false);
    }
  };

  const openPredictModal = (model: ModelSummary) => {
    const defaults: Record<string, string> = {};
    model.columns.forEach((column) => {
      defaults[column] = "";
    });
    setPredictModalModel(model);
    setPredictionInputs(defaults);
    setPredictResult(null);
  };

  const closePredictModal = () => {
    setPredictModalModel(null);
    setPredictionInputs({});
    setPredictResult(null);
  };

  const runPredict = async () => {
    if (!predictModalModel) return;
    try {
      const record =
        predictModalModel.columns.length === 0
          ? { image_b64: predictionInputs["image_b64"] }
          : predictModalModel.columns.reduce<Record<string, string | number | null>>(
              (acc, column) => {
                const value = predictionInputs[column] ?? "";
                if (value.trim() === "") {
                  acc[column] = null;
                  return acc;
                }
                const hint = predictModalModel.feature_hints.find((item) => item.name === column);
                if (hint?.kind === "categorical" && hint.value_map) {
                  const numericValue = Number(value);
                  if (!Number.isNaN(numericValue)) {
                    const match = Object.entries(hint.value_map).find(
                      ([, code]) => code === numericValue,
                    );
                    if (match) {
                      acc[column] = match[0];
                      return acc;
                    }
                  }
                  acc[column] = value;
                  return acc;
                }
                const numericValue = Number(value);
                acc[column] = Number.isNaN(numericValue) ? value : numericValue;
                return acc;
              },
              {},
            );
      const res = await axios.post<PredictResponse>(`${API_BASE}/predict`, {
        model_id: predictModalModel.model_id,
        records: [record],
      });
      setPredictResult(res.data.predictions.map((p) => String(p)).join(", "));
      toast.success("Prediction ready");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Prediction failed.");
    }
  };
  const canTrain = user ? (user.is_subscribed || user.free_runs_remaining > 0) : false;

  if (authLoading) {
    return <div className="app-container"><p>Loading...</p></div>;
  }

  if (!user) {
    return null; // Will redirect to login
  }

  return (
    <div className="app-container" style={{ position: "relative" }}>
      {(isTraining || uploading || isAILoading) && (
        <div className="training-overlay">
          <div className="training-message">
            <div className="training-spinner"></div>
            <p>
              {uploading 
                ? "Uploading dataset please wait..." 
                : isAILoading 
                ? "AI is processing please wait..." 
                : "Training the model please wait..."}
            </p>
          </div>
        </div>
      )}
      <header>
        <div>
          <p className="eyebrow">AI-as-a-Service</p>
          <h1>Code-Free Machine Learning Studio</h1>
          <p className="subtitle">
            Mode selected: {datasetMode === "csv" ? "CSV (Tabular)" : "Images with Labelled Dataset"}.
            Upload, train, and get your PDF & predictions—tailored for this mode.
          </p>
          <div className="hero-steps">
            <span className="pill">1. Upload</span>
            <span className="pill">2. Train</span>
            <span className="pill">4. PDF & Predict</span>
          </div>
        </div>
        <div className="header-actions">
          <div className="mode-grid">
            <div className="mode-card active locked">
              <div className="pill subtle">Mode locked</div>
              <h4>{datasetMode === "csv" ? "CSV (Tabular)" : "Images with Labelled Dataset"}</h4>
            </div>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            {recentDatasets.length > 0 && (
              <button
                type="button"
                onClick={() => setShowRecentDatasets(!showRecentDatasets)}
                style={{
                  background: showRecentDatasets ? "#4f46e5" : "#e2e8f0",
                  color: showRecentDatasets ? "white" : "#0f172a",
                  padding: "1rem 2rem",
                  borderRadius: "12px",
                  fontWeight: "600",
                  cursor: "pointer",
                  border: "none",
                  transition: "all 0.2s",
                }}
              >
                {showRecentDatasets ? "Hide Recent" : `Recent Datasets (${recentDatasets.length})`}
              </button>
            )}
            <label className="upload-cta">
              <input
                type="file"
                accept={datasetMode === "csv" ? ".csv" : ".zip"}
                onChange={handleUpload}
                disabled={uploading}
              />
              {datasetMode === "csv" ? "Upload CSV" : "Upload Images (zip)"}
            </label>
          </div>
        </div>
      </header>

      {showRecentDatasets && recentDatasets.length > 0 && (
        <section className="panel" style={{ marginBottom: "1.5rem" }}>
          <div className="panel-header">
            <h2>Recent Datasets</h2>
            <p className="muted">Select a previously uploaded dataset to use</p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", gap: "1rem" }}>
            {recentDatasets.map((ds) => (
              <div
                key={ds.dataset_id}
                onClick={async () => {
                  try {
                    const token = localStorage.getItem("token");
                    if (!token) {
                      toast.error("Authentication required");
                      return;
                    }
                    if (ds.mode === "csv") {
                      const res = await axios.get<UploadResponse>(`${API_BASE}/datasets/${ds.dataset_id}/load`, {
                        headers: { Authorization: `Bearer ${token}` },
                      });
                      setDataset(res.data);
                      setImageDataset(null);
                      setTargetColumn(res.data.suggested_target ?? "");
                      setDetectInfo(null);
                      localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
                        dataset_id: res.data.dataset_id,
                        filename: res.data.filename,
                        mode: datasetMode,
                      }));
                      toast.success("Dataset loaded");
                      setShowRecentDatasets(false);
                      setTimeout(() => {
                        fetchTrainingHistory().catch(console.error);
                      }, 100);
                    } else {
                      const res = await axios.get<ImageUploadResponse>(`${API_BASE}/datasets/${ds.dataset_id}/load-image`, {
                        headers: { Authorization: `Bearer ${token}` },
                      });
                      setImageDataset(res.data);
                      setDataset(null);
                      setDetectInfo(null);
                      setTargetColumn("image_label");
                      localStorage.setItem(`current_dataset_${datasetMode}`, JSON.stringify({
                        dataset_id: res.data.dataset_id,
                        filename: res.data.filename,
                        mode: datasetMode,
                      }));
                      toast.success("Image dataset loaded");
                      setShowRecentDatasets(false);
                      setTimeout(() => {
                        fetchTrainingHistory().catch(console.error);
                      }, 100);
                    }
                  } catch (err: any) {
                    toast.error(err?.response?.data?.detail ?? "Failed to load dataset");
                  }
                }}
                style={{
                  padding: "1rem",
                  border: "1px solid #e2e8f0",
                  borderRadius: "12px",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  background: "white",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "#f8fafc";
                  e.currentTarget.style.borderColor = "#cbd5e1";
                  e.currentTarget.style.transform = "translateY(-2px)";
                  e.currentTarget.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "white";
                  e.currentTarget.style.borderColor = "#e2e8f0";
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div style={{ fontWeight: "600", marginBottom: "0.5rem", color: "#0f172a" }}>
                  {ds.filename}
                </div>
                <div style={{ fontSize: "0.875rem", color: "#64748b" }}>
                  {ds.mode === "csv" ? (
                    <>
                      {ds.row_count?.toLocaleString()} rows · {ds.columns.length} columns
                    </>
                  ) : (
                    <>
                      {ds.total_images?.toLocaleString()} images
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="panel highlight compact">
        <div className="panel-header">
          <div>
            <h2>{currentModeTips.title}</h2>
            <p className="muted">{currentModeTips.helper}</p>
          </div>
          <div className="badge">Quick guide</div>
        </div>
        <div className="instruction-list compact">
          {currentModeTips.bullets.map((item) => (
            <div key={item} className="instruction-item slim">
              <span className="pill tiny">Tip</span>
              <p>{item}</p>
            </div>
          ))}
        </div>
      </section>

      {dataset && (
        <section className="panel">
          <div className="panel-header">
            <h2>Dataset Preview</h2>
            <div className="badge">
              {dataset.row_count} rows · {dataset.columns.length} columns
            </div>
          </div>
          <div className="dataset-meta">
            <div>
              <span>File</span>
              <strong>{dataset.filename}</strong>
            </div>
            <div>
              <span>Suggested Target</span>
              <strong>{dataset.suggested_target ?? "—"}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{dataset.confidence ? metricLabel(dataset.confidence) : "—"}</strong>
            </div>
          </div>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  {dataset.preview.length > 0 &&
                    Object.keys(dataset.preview[0]).map((col) => <th key={col}>{col}</th>)}
                </tr>
              </thead>
              <tbody>
                {dataset.preview.map((row, idx) => (
                  <tr key={idx}>
                    {Object.values(row).map((value, cellIdx) => (
                      <td key={cellIdx}>{String(value ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {imageDataset && (
        <>
          <section className="panel">
            <div className="panel-header">
              <h2>Image Dataset</h2>
              <div className="badge">
                {imageDataset.total_images} images · {imageDataset.classes.length} classes
              </div>
            </div>
            <div className="dataset-meta">
              <div>
                <span>File</span>
                <strong>{imageDataset.filename}</strong>
              </div>
              <div>
                <span>Mode</span>
                <strong>Images with Labelled Dataset</strong>
              </div>
              <div>
                <span>Classes</span>
                <strong>
                  {imageDataset.classes.map((c) => `${c.label} (${c.count})`).join(", ")}
                </strong>
              </div>
            </div>
            <div className="callout">
              Images with labelled dataset: each folder = class. Train will use a CNN.
            </div>
          </section>

          {sampleImages.length > 0 && (
            <section className="panel">
              <div className="panel-header">
                <h2>Dataset Preview</h2>
                <div className="badge">
                  Sample from each class
                </div>
              </div>
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", 
                gap: "1.5rem",
                marginTop: "1rem"
              }}>
                {sampleImages.map((sample) => (
                  <div 
                    key={sample.class_label}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "0.5rem",
                      padding: "1rem",
                      border: "1px solid #e2e8f0",
                      borderRadius: "12px",
                      background: "white",
                      transition: "all 0.2s",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = "#cbd5e1";
                      e.currentTarget.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = "#e2e8f0";
                      e.currentTarget.style.boxShadow = "none";
                    }}
                  >
                    <div style={{ 
                      width: "100%", 
                      aspectRatio: "1", 
                      borderRadius: "8px",
                      overflow: "hidden",
                      background: "#f8fafc",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center"
                    }}>
                      <img 
                        src={`data:image/jpeg;base64,${sample.image_b64}`}
                        alt={sample.class_label}
                        style={{
                          width: "100%",
                          height: "100%",
                          objectFit: "cover"
                        }}
                      />
                    </div>
                    <div style={{ 
                      textAlign: "center",
                      fontWeight: "600",
                      color: "#0f172a",
                      fontSize: "0.875rem"
                    }}>
                      {sample.class_label}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </>
      )}

      <section className="grid">
        {datasetMode === "csv" && (
          <article className="panel">
            <h3>AI Assistance</h3>
            <p className="muted">Let AI pick the target column, best model, or automate everything.</p>
            <div className="button-stack">
              <button onClick={runDetectTarget} disabled={!dataset || datasetMode !== "csv" || isAILoading || isTraining}>
                {!dataset ? "Upload CSV first" : isAILoading ? "AI processing..." : "Ask AI to detect target"}
              </button>
              <button onClick={runAutoSelect} disabled={!currentDatasetId() || isAILoading || isTraining}>
                {!currentDatasetId() ? "Upload dataset first" : isAILoading ? "AI processing..." : "Ask AI to choose the best model"}
              </button>
              <button className="primary" onClick={runAutoEverything} disabled={!currentDatasetId() || isTraining || isAILoading}>
                {!currentDatasetId() ? "Upload dataset first" : isTraining ? "Training..." : "Ask AI to do everything"}
              </button>
            </div>
            {detectInfo && (
              <div className="callout">
                <strong>Detected Target:</strong> {detectInfo.target_column} ({detectInfo.problem_type})<br />
                <em>{detectInfo.reason}</em>
              </div>
            )}
          </article>
        )}

        <article className="panel">
          <h3>Training Controls</h3>
          <div className="form-grid">
            {datasetMode === "csv" && (
              <label>
                Target Column
                <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                  <option value="">Select target</option>
                  {dataset?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <div style={{ gridColumn: "1 / -1", marginBottom: "0.5rem" }}>
              <small style={{ color: "#666" }}>
                {datasetMode === "csv" && "✓ Classical ML models only (KNN, Decision Tree, Random Forest, SVM, Logistic Regression, Naive Bayes, XGBoost)"}
                {datasetMode === "supervised" && "✓ Deep Learning CNN models only (CNN, Transfer Learning, Vision Transformers)"}
              </small>
            </div>
            {datasetMode === "csv" && (
              <label>
                Model Choice
                <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                  {catalog.map((entry) => (
                    <option key={entry.key} value={entry.key}>
                      {entry.name}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {datasetMode === "supervised" && (
              <div style={{ gridColumn: "1 / -1", padding: "0.75rem", background: "#f5f5f5", borderRadius: "4px" }}>
                <small>
                  <strong>Auto-selected model:</strong> Convolutional Neural Network (CNN)
                </small>
              </div>
            )}
            <label>
              Training Intensity
              <select value={trainingIntensity} onChange={(e) => setTrainingIntensity(e.target.value as "less" | "medium" | "rigorous")}>
                <option value="less">Less Training (Fast / Basic)</option>
                <option value="medium">Medium Training (Normal / Balanced)</option>
                <option value="rigorous">Rigorous Training (Advanced / Deep)</option>
              </select>
            </label>
          </div>
          {!canTrain && (
            <div className="callout warning">
              <strong>No free runs remaining.</strong> Subscribe to continue training models.
            </div>
          )}
          <div className="button-row">
            <button onClick={handleTrainSingle} disabled={!currentDatasetId() || (!targetColumn && datasetMode === "csv") || !canTrain}>
              {!currentDatasetId()
                ? "Upload dataset first"
                : datasetMode === "csv" && !targetColumn
                ? "Select target column"
                : !canTrain
                ? "Subscribe to continue"
                : datasetMode === "supervised"
                ? "Train CNN Model"
                : "Train Selected Model"}
            </button>
            {datasetMode === "csv" && (
              <button className="primary" onClick={handleTrainAll} disabled={!currentDatasetId() || (!targetColumn && datasetMode === "csv") || !canTrain}>
                {!currentDatasetId()
                  ? "Upload dataset first"
                  : datasetMode === "csv" && !targetColumn
                  ? "Select target column"
                  : !canTrain
                  ? "Subscribe to continue"
                  : "Train All Models"}
              </button>
            )}
          </div>
        </article>
      </section>

      {/* Training History - Moved to bottom */}
      {(() => {
        // Filter history by mode only (show all history for current mode)
        const filteredHistory = trainingHistory
          .filter((record) => {
            // For old records without dataset_mode, default to "csv" for backward compatibility
            const recordMode = record.dataset_mode || "csv";
            // Match mode
            return datasetMode === "csv" ? recordMode === "csv" : recordMode === datasetMode;
          })
          // Sort by created_at descending (newest first)
          .sort((a, b) => {
            const dateA = new Date(a.created_at).getTime();
            const dateB = new Date(b.created_at).getTime();
            return dateB - dateA; // Descending order (newest first)
          });
        
        return (
          <section className="panel">
            <div className="panel-header">
              <h2>Training History ({datasetMode} mode)</h2>
              <p className="muted">All training runs for {datasetMode} datasets, sorted by newest first</p>
            </div>
            {filteredHistory.length > 0 ? (
              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Training Info</th>
                      <th>Accuracy</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredHistory.map((record) => {
                      // Format: dataset-model-type-date/time(IST)
                      const intensityMap: Record<string, string> = {
                        "less": "Easy",
                        "medium": "Medium",
                        "rigorous": "Rigorous"
                      };
                      const intensityDisplay = intensityMap[record.intensity] || record.intensity.charAt(0).toUpperCase() + record.intensity.slice(1);
                      const modelDisplay = record.model_name || "All Models";
                      const trainingType = `${modelDisplay} - ${intensityDisplay}`;
                      
                      // Convert UTC to IST (UTC+5:30)
                      const utcDate = new Date(record.created_at);
                      const istDate = new Date(utcDate.getTime() + (5.5 * 60 * 60 * 1000));
                      const istDateTime = istDate.toLocaleString("en-IN", {
                        timeZone: "Asia/Kolkata",
                        year: "numeric",
                        month: "2-digit",
                        day: "2-digit",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                        hour12: true
                      });
                      
                      const trainingInfo = `${record.dataset_name} - ${trainingType} - ${istDateTime} (IST)`;
                      
                      return (
                        <tr key={record.history_id}>
                          <td>
                            <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                              <strong>{trainingInfo}</strong>
                              <span className="muted" style={{ fontSize: "0.75rem" }}>
                                {record.target_column} · {record.problem_type}
                              </span>
                            </div>
                          </td>
                          <td>
                            {record.metrics?.accuracy
                              ? (record.metrics.accuracy * 100).toFixed(1) + "%"
                              : record.metrics?.r2
                              ? record.metrics.r2.toFixed(3)
                              : record.metrics?.silhouette
                              ? record.metrics.silhouette.toFixed(3)
                              : "—"}
                          </td>
                          <td style={{ verticalAlign: "top", padding: "0.75rem 1rem" }}>
                            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", alignItems: "flex-start" }}>
                              {/* First line: Predict and View Report */}
                              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center", width: "100%" }}>
                                {record.model_id && (
                                  <button
                                    type="button"
                                    className="link-button"
                                    style={{ width: "140px", height: "36px", textAlign: "center", padding: "0.4rem 0.8rem", boxSizing: "border-box" }}
                                    onClick={async (event) => {
                                      event.stopPropagation();
                                      try {
                                        // Fetch model details from models list
                                        const token = localStorage.getItem("token");
                                        if (!token) {
                                          toast.error("Authentication required");
                                          return;
                                        }
                                        const res = await axios.get(`${API_BASE}/models`, {
                                          headers: { Authorization: `Bearer ${token}` },
                                        });
                                        const model = res.data.models.find((m: ModelSummary) => m.model_id === record.model_id);
                                        if (model) {
                                          openPredictModal(model);
                                        } else {
                                          toast.error("Model not found");
                                        }
                                      } catch (err: any) {
                                        toast.error(err?.response?.data?.detail ?? "Failed to load model");
                                      }
                                    }}
                                  >
                                    Predict
                                  </button>
                                )}
                                {record.report_id && (
                                  <Link 
                                    to={`/report/${record.report_id}`} 
                                    className="link-button"
                                    style={{ width: "140px", height: "36px", textAlign: "center", display: "inline-flex", alignItems: "center", justifyContent: "center", padding: "0.4rem 0.8rem", boxSizing: "border-box", textDecoration: "none", border: "1px solid #cbd5f5", borderRadius: "8px", fontFamily: "inherit", fontSize: "0.875rem", fontWeight: "500", color: "#4f46e5", background: "none", cursor: "pointer", whiteSpace: "nowrap", lineHeight: "1.5" }}
                                  >
                                    View Report
                                  </Link>
                                )}
                              </div>
                              {/* Second line: Download buttons */}
                              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center", width: "100%" }}>
                                {record.pkl_path && record.model_id && (
                                  <button
                                    type="button"
                                    className="link-button"
                                    style={{ width: "140px", height: "36px", textAlign: "center", padding: "0.4rem 0.8rem", boxSizing: "border-box" }}
                                    onClick={async (e) => {
                                      e.stopPropagation();
                                      try {
                                        const token = localStorage.getItem("token");
                                        if (!token) {
                                          toast.error("Authentication required");
                                          return;
                                        }
                                        const response = await axios.get(`${API_BASE}/models/download/${record.model_id}`, {
                                          headers: { Authorization: `Bearer ${token}` },
                                          responseType: "blob",
                                        });
                                        const url = window.URL.createObjectURL(new Blob([response.data]));
                                        const link = document.createElement("a");
                                        link.href = url;
                                        link.setAttribute("download", `${record.model_name || "model"}.pkl`);
                                        document.body.appendChild(link);
                                        link.click();
                                        link.remove();
                                        window.URL.revokeObjectURL(url);
                                      } catch (err: any) {
                                        toast.error(err?.response?.data?.detail ?? "Failed to download .pkl file");
                                      }
                                    }}
                                  >
                                    Download .pkl
                                  </button>
                                )}
                                {record.h5_path && record.model_id && (
                                  <button
                                    type="button"
                                    className="link-button"
                                    style={{ width: "140px", height: "36px", textAlign: "center", padding: "0.4rem 0.8rem", boxSizing: "border-box" }}
                                    onClick={async (e) => {
                                      e.stopPropagation();
                                      try {
                                        const token = localStorage.getItem("token");
                                        if (!token) {
                                          toast.error("Authentication required");
                                          return;
                                        }
                                        const response = await axios.get(`${API_BASE}/models/download/${record.model_id}/h5`, {
                                          headers: { Authorization: `Bearer ${token}` },
                                          responseType: "blob",
                                        });
                                        const url = window.URL.createObjectURL(new Blob([response.data]));
                                        const link = document.createElement("a");
                                        link.href = url;
                                        link.setAttribute("download", `${record.model_name || "model"}.h5`);
                                        document.body.appendChild(link);
                                        link.click();
                                        link.remove();
                                        window.URL.revokeObjectURL(url);
                                      } catch (err: any) {
                                        toast.error(err?.response?.data?.detail ?? "Failed to download .h5 file");
                                      }
                                    }}
                                  >
                                    Download .h5
                                  </button>
                                )}
                              </div>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="muted" style={{ padding: "1rem", textAlign: "center" }}>
                No history yet. Train a model to see results here.
              </p>
            )}
          </section>
        );
      })()}

      {predictModalModel && (
        <div className="modal-backdrop" onClick={closePredictModal}>
          <div className="modal-panel" onClick={(event) => event.stopPropagation()}>
            <h3>Predict with {predictModalModel.model_name}</h3>
            <p className="muted">Enter feature values and run this trained model instantly.</p>
            {predictModalModel.columns.length === 0 ? (
              <div className="prediction-grid">
                <label>
                  Upload Image
                  <input
                    type="file"
                    accept="image/*"
                    onChange={async (event) => {
                      const file = event.target.files?.[0];
                      if (!file) return;
                      const b64 = await fileToBase64(file);
                      setPredictionInputs({ image_b64: b64 });
                    }}
                  />
                  <span className="hint">We accept JPG/PNG for image models.</span>
                </label>
              </div>
            ) : (
              <div className="prediction-grid">
                {predictModalModel.columns.map((column) => (
                  <label key={column}>
                    {column}
                    <input
                      type="text"
                      value={predictionInputs[column] ?? ""}
                      onChange={(event) =>
                        setPredictionInputs((prev) => ({
                          ...prev,
                          [column]: event.target.value,
                        }))
                      }
                      placeholder="e.g. value"
                    />
                    {(() => {
                      const hint = predictModalModel.feature_hints.find((item) => item.name === column);
                      if (!hint) return null;
                      if (hint.kind === "categorical" && hint.value_map) {
                        const mapping = Object.entries(hint.value_map)
                          .map(([label, code]) => `${label}=${code}`)
                          .join(", ");
                        return <span className="hint">Mappings: {mapping}</span>;
                      }
                      if (hint.examples.length > 0) {
                        return <span className="hint">Examples: {hint.examples.join(", ")}</span>;
                      }
                      return null;
                    })()}
                  </label>
                ))}
              </div>
            )}
            {predictResult && (
              <div className="callout">
                <strong>Prediction</strong>
                <p>{predictResult}</p>
              </div>
            )}
            <div className="modal-actions">
              <button type="button" className="link-button" onClick={closePredictModal}>
                Cancel
              </button>
              <button type="button" className="primary" onClick={runPredict}>
                Predict
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


