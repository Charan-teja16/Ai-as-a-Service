import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import toast from "react-hot-toast";
import { useAuth } from "../contexts/AuthContext";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

interface ReportData {
  report_id: string;
  target_column: string;
  problem_type: string;
  dataset_id?: string | null;
  dataset_mode?: string | null;
  leaderboard: Array<{
    model_name: string;
    model_key: string;
    rank: number;
    metrics: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1?: number;
      rmse?: number;
      r2?: number;
    };
  }>;
  created_at: string;
}

export default function ReportPreview() {
  const { reportId } = useParams<{ reportId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!reportId) {
      navigate("/");
      return;
    }
    const fetchReportData = async () => {
      setLoading(true);
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          navigate("/login");
          return;
        }
        const response = await axios.get(`${API_BASE}/report/${reportId}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setReportData(response.data);
        // fetch actual PDF for exact preview
        const pdfRes = await fetch(`${API_BASE}/report/download/${reportId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (pdfRes.ok) {
          const blob = await pdfRes.blob();
          const url = URL.createObjectURL(blob);
          setPdfUrl(url);
        }
      } catch (error: any) {
        console.error("Report fetch error:", error);
        toast.error(error?.response?.data?.detail || "Failed to load report");
        if (error?.response?.status === 403 || error?.response?.status === 404) {
          navigate("/");
        }
      } finally {
        setLoading(false);
      }
    };
    fetchReportData();
  }, [reportId, navigate]);

  const handleDownload = () => {
    if (!reportId) return;
    const token = localStorage.getItem("token");
    if (!token) {
      toast.error("Please login to download reports");
      return;
    }
    // Create a temporary link to download with auth header
    const url = `${API_BASE}/report/download/${reportId}`;
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    // For authenticated downloads, we need to use fetch with headers
    fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((res) => {
        if (res.ok) {
          return res.blob();
        }
        throw new Error("Download failed");
      })
      .then((blob) => {
        const blobUrl = window.URL.createObjectURL(blob);
        link.href = blobUrl;
        link.download = `report-${reportId}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(blobUrl);
        toast.success("Report downloaded!");
      })
      .catch((err) => {
        console.error("Download error:", err);
        toast.error("Failed to download report");
      });
  };

  const handleSendEmail = async () => {
    if (!reportId || !user) return;
    setSending(true);
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        toast.error("Please login to send reports");
        return;
      }
      const response = await axios.post(
        `${API_BASE}/reports/${reportId}/send-email`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      toast.success(response.data.message || "Report sent to your email!");
    } catch (error: any) {
      toast.error(error?.response?.data?.detail || "Failed to send email");
    } finally {
      setSending(false);
    }
  };

  if (loading) {
    return (
      <div className="app-container">
        <p>Loading report...</p>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header>
        <div>
          <h1>Training Report Preview</h1>
          <p className="subtitle">Detailed analysis of your model training results</p>
        </div>
        <button
          className="link-button"
          onClick={() => {
            // If report knows which workspace + dataset it came from, restore that
            if (reportData?.dataset_mode && reportData?.dataset_id) {
              const mode =
                reportData.dataset_mode === "supervised" ? "supervised" : "csv";
              navigate(`/workspace/${mode}?datasetId=${reportData.dataset_id}`);
            } else {
              // Fallback to mode select
              navigate("/");
            }
          }}
        >
          Back to Dashboard
        </button>
      </header>

      <section className="panel">
        <div className="panel-header">
          <h2>Report Details</h2>
          <div className="badge">Report ID: {reportId}</div>
        </div>

        {reportData && (
          <div className="report-preview">
            <div className="report-section">
              <h3>Executive Summary</h3>
              <p>
                This report summarizes the performance of machine learning models trained on the{" "}
                <strong>{reportData.target_column}</strong> dataset for a{" "}
                <strong>{reportData.problem_type}</strong> problem.
              </p>
              <p className="muted" style={{ marginTop: "0.5rem" }}>
                Generated on: {new Date(reportData.created_at).toLocaleString()}
              </p>
            </div>
            
            {reportData.leaderboard && reportData.leaderboard.length > 0 && (
              <div className="report-section">
                <h3>Top Models</h3>
                <table className="preview-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Model</th>
                      <th>Accuracy</th>
                      <th>F1 / R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reportData.leaderboard.slice(0, 5).map((model) => (
                      <tr key={model.model_key}>
                        <td>{model.rank}</td>
                        <td>{model.model_name}</td>
                        <td>{model.metrics.accuracy?.toFixed(3) ?? "—"}</td>
                        <td>
                          {reportData.problem_type === "classification"
                            ? model.metrics.f1?.toFixed(3) ?? "—"
                            : model.metrics.r2?.toFixed(3) ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="report-section">
              <h3>Report Contents</h3>
              <ul>
                <li>Executive Summary</li>
                <li>Top Models Overview with Performance Metrics</li>
                <li>Detailed Model Reports</li>
                <li>Performance Metrics (Accuracy, Precision, Recall, F1, RMSE, R²)</li>
                <li>Confusion Matrix (for classification)</li>
                <li>Feature Information and Mappings</li>
                <li>Visualizations (ROC curves, Residuals, Feature Importance)</li>
                <li>Model Recommendations</li>
              </ul>
            </div>

            <div className="report-section">
              <h3>PDF Preview (exact file)</h3>
              {pdfUrl ? (
                <div className="pdf-preview-box">
                  <object data={pdfUrl} type="application/pdf" width="100%" height="640px">
                    <p>PDF preview not supported in this browser. Please download instead.</p>
                  </object>
                </div>
              ) : (
                <p className="muted">Loading PDF preview...</p>
              )}
            </div>
          </div>
        )}

        <div className="report-actions">
          <button className="primary" onClick={handleDownload}>
            Download PDF Report
          </button>
          <button className="primary" onClick={handleSendEmail} disabled={sending}>
            {sending ? "Sending..." : "Send Report to Email"}
          </button>
        </div>
      </section>
    </div>
  );
}

