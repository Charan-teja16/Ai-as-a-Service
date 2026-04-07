import { useEffect, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import type { TrainingHistoryRecord } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function Profile() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [history, setHistory] = useState<TrainingHistoryRecord[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoadingHistory(true);
        const token = localStorage.getItem("token");
        if (!token) return;
        const res = await axios.get(`${API_BASE}/training-history`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setHistory(res.data.history || []);
      } catch (err) {
        console.error("Failed to load history", err);
      } finally {
        setLoadingHistory(false);
      }
    };
    if (user) fetchHistory();
  }, [user]);

  if (!user) return null;

  return (
    <div className="app-container">
      <header>
        <div>
          <p className="eyebrow">AI-as-a-Service</p>
          <h1>Your Profile</h1>
          <p className="subtitle">Account, subscription, and recent activity.</p>
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>Account</h2>
          <div className="badge">{user.is_subscribed ? "Subscribed" : `Free runs: ${user.free_runs_remaining}`}</div>
        </div>
        <div className="dataset-meta">
          <div>
            <span>Username</span>
            <strong>{user.username}</strong>
          </div>
          <div>
            <span>Email</span>
            <strong>{user.email}</strong>
          </div>
          <div>
            <span>Total runs</span>
            <strong>{user.total_runs}</strong>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Recent Training</h2>
          <p className="muted">Most recent jobs with quick metrics.</p>
        </div>
        {loadingHistory ? (
          <p className="muted">Loading history...</p>
        ) : history.length === 0 ? (
          <p className="muted">No training runs yet.</p>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Dataset</th>
                  <th>Type</th>
                  <th>Model</th>
                  <th>Metric</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => {
                  const metric =
                    h.metrics?.accuracy !== undefined && h.metrics?.accuracy !== null
                      ? `${(h.metrics.accuracy * 100).toFixed(1)}% acc`
                      : h.metrics?.r2 !== undefined && h.metrics?.r2 !== null
                      ? `R² ${h.metrics.r2.toFixed(3)}`
                      : (h as any)?.metrics?.silhouette !== undefined && (h as any)?.metrics?.silhouette !== null
                      ? `Sil ${(h as any).metrics.silhouette.toFixed(3)}`
                      : "—";
                  const datasetType = h.dataset_mode 
                    ? (h.dataset_mode === "csv" ? "CSV" : "Images with Labelled Dataset")
                    : h.problem_type;
                  return (
                    <tr key={h.history_id}>
                      <td>{new Date(h.created_at).toLocaleString()}</td>
                      <td>{h.dataset_name}</td>
                      <td>{datasetType}</td>
                      <td>{h.model_name || "All Models"}</td>
                      <td>{metric}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

