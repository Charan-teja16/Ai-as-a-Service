import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { useAuth } from "../contexts/AuthContext";
import type { ModelInfoEntry, ModelInfoListResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const complexityColors = {
  Low: "#10b981",
  Medium: "#f59e0b",
  High: "#ef4444",
};

const speedColors = {
  Fast: "#10b981",
  Medium: "#f59e0b",
  Slow: "#ef4444",
  "N/A (lazy learning)": "#6b7280",
};

const interpretabilityColors = {
  High: "#10b981",
  Medium: "#f59e0b",
  Low: "#ef4444",
};

export default function ModelInfo() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfoEntry[]>([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [filter, setFilter] = useState<{
    problemType?: string;
    datasetMode?: string;
  }>({});

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    if (user) {
      fetchModels();
    }
  }, [user, filter]);

  const fetchModels = async () => {
    try {
      setLoadingModels(true);
      const token = localStorage.getItem("token");
      if (!token) return;

      const params = new URLSearchParams();
      if (filter.problemType) params.append("problem_type", filter.problemType);
      if (filter.datasetMode) params.append("dataset_mode", filter.datasetMode);

      const res = await axios.get<ModelInfoListResponse>(
        `${API_BASE}/models/info?${params.toString()}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      setModels(res.data.models);
    } catch (err) {
      console.error("Failed to fetch models:", err);
    } finally {
      setLoadingModels(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <div>
          <p className="eyebrow">Model Information</p>
          <h1>Explore Machine Learning Models</h1>
          <p className="subtitle">
            Learn how different algorithms work, their strengths, weaknesses, and when to use them.
          </p>
        </div>
      </header>

      <section className="panel highlight">
        <div className="panel-header">
          <div>
            <h2>Filter Models</h2>
            <p className="muted">Filter by problem type or dataset mode</p>
          </div>
        </div>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "2rem" }}>
          <select
            value={filter.problemType || ""}
            onChange={(e) =>
              setFilter({ ...filter, problemType: e.target.value || undefined })
            }
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              fontSize: "0.875rem",
            }}
          >
            <option value="">All Problem Types</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
            <option value="unsupervised">Unsupervised</option>
          </select>
          <select
            value={filter.datasetMode || ""}
            onChange={(e) =>
              setFilter({ ...filter, datasetMode: e.target.value || undefined })
            }
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              fontSize: "0.875rem",
            }}
          >
            <option value="">All Dataset Modes</option>
            <option value="csv">CSV</option>
            <option value="supervised">Supervised Images</option>
            <option value="unsupervised">Unsupervised Images</option>
          </select>
          {(filter.problemType || filter.datasetMode) && (
            <button
              onClick={() => setFilter({})}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "8px",
                border: "1px solid #e5e7eb",
                background: "#fff",
                cursor: "pointer",
                fontSize: "0.875rem",
              }}
            >
              Clear Filters
            </button>
          )}
        </div>
      </section>

      {loadingModels ? (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p>Loading models...</p>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))",
            gap: "1.5rem",
            marginTop: "2rem",
          }}
        >
          {models.map((model) => (
            <div
              key={model.key}
              onClick={() => navigate(`/models/${model.key}`)}
              style={{
                background: "#fff",
                border: "1px solid #e5e7eb",
                borderRadius: "12px",
                padding: "1.5rem",
                cursor: "pointer",
                transition: "all 0.2s",
                boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-4px)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 1px 3px rgba(0,0,0,0.1)";
              }}
            >
              <div style={{ marginBottom: "1rem" }}>
                <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1.25rem", fontWeight: "600" }}>
                  {model.name}
                </h3>
                <p
                  style={{
                    margin: 0,
                    color: "#6b7280",
                    fontSize: "0.875rem",
                    lineHeight: "1.5",
                  }}
                >
                  {model.description}
                </p>
              </div>

              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
                {model.problem_types.map((type) => (
                  <span
                    key={type}
                    style={{
                      padding: "0.25rem 0.75rem",
                      borderRadius: "12px",
                      background: "#f3f4f6",
                      color: "#374151",
                      fontSize: "0.75rem",
                      fontWeight: "500",
                    }}
                  >
                    {type}
                  </span>
                ))}
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: "0.75rem",
                  marginBottom: "1rem",
                  paddingTop: "1rem",
                  borderTop: "1px solid #e5e7eb",
                }}
              >
                <div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "#6b7280",
                      marginBottom: "0.25rem",
                    }}
                  >
                    Complexity
                  </div>
                  <div
                    style={{
                      fontSize: "0.875rem",
                      fontWeight: "600",
                      color: complexityColors[model.complexity],
                    }}
                  >
                    {model.complexity}
                  </div>
                </div>
                <div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "#6b7280",
                      marginBottom: "0.25rem",
                    }}
                  >
                    Speed
                  </div>
                  <div
                    style={{
                      fontSize: "0.875rem",
                      fontWeight: "600",
                      color: speedColors[model.training_speed],
                    }}
                  >
                    {model.training_speed}
                  </div>
                </div>
                <div>
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "#6b7280",
                      marginBottom: "0.25rem",
                    }}
                  >
                    Interpretability
                  </div>
                  <div
                    style={{
                      fontSize: "0.875rem",
                      fontWeight: "600",
                      color: interpretabilityColors[model.interpretability],
                    }}
                  >
                    {model.interpretability}
                  </div>
                </div>
              </div>

              <div
                style={{
                  color: "#3b82f6",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  marginTop: "1rem",
                }}
              >
                Learn more →
              </div>
            </div>
          ))}
        </div>
      )}

      {!loadingModels && models.length === 0 && (
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p style={{ color: "#6b7280" }}>No models found matching your filters.</p>
        </div>
      )}
    </div>
  );
}


