import { useEffect, useState, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import axios from "axios";
import { useAuth } from "../contexts/AuthContext";
import type { ModelDetailResponse } from "../types";

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

export default function ModelDetail() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const { modelKey } = useParams<{ modelKey: string }>();
  const [model, setModel] = useState<ModelDetailResponse | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    if (user && modelKey) {
      fetchModel();
    }
  }, [user, modelKey]);

  useEffect(() => {
    if (!model) return;
    // For some models we use static images instead of canvas drawing
    if (
      model.key === "decision_tree" ||
      model.key === "gradient_boosting" ||
      model.key === "random_forest" ||
      model.key === "xgboost" ||
      model.key === "svm" ||
      model.key === "image_cnn" ||
      model.key === "image_kmeans" ||
      model.key === "knn" ||
      model.key === "linear_regression" ||
      model.key === "logistic_regression" ||
      model.key === "naive_bayes"
    ) {
      return;
    }
    if (canvasRef.current) {
      drawVisualization();
    }
  }, [model]);

  const fetchModel = async () => {
    try {
      setLoadingModel(true);
      const token = localStorage.getItem("token");
      if (!token) return;

      const res = await axios.get<ModelDetailResponse>(
        `${API_BASE}/models/info/${modelKey}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      setModel(res.data);
    } catch (err) {
      console.error("Failed to fetch model:", err);
    } finally {
      setLoadingModel(false);
    }
  };

  const drawVisualization = () => {
    if (!model || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Light background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Top header: three clearly separated parts (Before | What happens | After)
    const headerY = 16;
    const headerHeight = 26;
    const pillWidth = 190;
    const gap = (width - pillWidth * 3) / 4;
    const leftX = gap;
    const midX = 2 * gap + pillWidth;
    const rightX = 3 * gap + 2 * pillWidth;

    const drawHeaderPill = (x: number, label: string) => {
      const radius = 12;
      const y = headerY;
      const w = pillWidth;
      const h = headerHeight;
      ctx.beginPath();
      ctx.moveTo(x + radius, y);
      ctx.lineTo(x + w - radius, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
      ctx.lineTo(x + w, y + h - radius);
      ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
      ctx.lineTo(x + radius, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
      ctx.lineTo(x, y + radius);
      ctx.quadraticCurveTo(x, y, x + radius, y);
      ctx.closePath();
      ctx.fillStyle = "#f3f4f6";
      ctx.fill();
      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#374151";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x + w / 2, y + h / 2 + 4);
      ctx.textAlign = "left";
    };

    drawHeaderPill(leftX, "Before: raw data");
    drawHeaderPill(midX, "What the model does");
    drawHeaderPill(rightX, "After: model output");

    // Example sentence tailored per model so students understand the story
    let exampleLabel = "";
    switch (model.key) {
      case "linear_regression":
        exampleLabel = "Example: House size (sq ft) and price → model learns a straight line to predict price.";
        break;
      case "logistic_regression":
        exampleLabel = "Example: Email features → model outputs probability of spam vs not spam.";
        break;
      case "svm":
        exampleLabel = "Example: Two groups of points → SVM finds the widest margin boundary between them.";
        break;
      case "decision_tree":
        exampleLabel = "Example: Age & income → the tree asks yes/no questions to reach a decision.";
        break;
      case "random_forest":
        exampleLabel = "Example: Many different trees vote on the final prediction (ensemble of trees).";
        break;
      case "gradient_boosting":
        exampleLabel = "Example: Small trees are added one by one, each fixing previous mistakes (boosting).";
        break;
      case "xgboost":
        exampleLabel = "Example: Optimized boosting trees for tabular data (fast, regularized gradient boosting).";
        break;
      case "knn":
        exampleLabel = "Example: New point looks at its closest neighbours and copies the majority label.";
        break;
      case "naive_bayes":
        exampleLabel = "Example: Word frequencies per class → model turns counts into probabilities.";
        break;
      case "image_cnn":
        exampleLabel = "Example: Raw image → filters detect edges and shapes → final image class.";
        break;
      case "image_kmeans":
        exampleLabel = "Example: Unlabeled images → K-Means groups visually similar ones together.";
        break;
      default:
        exampleLabel = "";
    }
    if (exampleLabel) {
      ctx.fillStyle = "#111827";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(exampleLabel, width / 2, headerY + headerHeight + 20);
      ctx.textAlign = "left";
    }

    switch (model.visualization_type) {
      case "regression_line":
        drawRegressionLine(ctx, width, height);
        break;
      case "decision_boundary":
        drawDecisionBoundary(ctx, width, height);
        break;
      case "tree":
        drawDecisionTree(ctx, width, height);
        break;
      case "feature_importance":
        drawFeatureImportance(ctx, width, height);
        break;
      case "knn":
        drawKNN(ctx, width, height);
        break;
      case "cnn_architecture":
        drawCNNArchitecture(ctx, width, height);
        break;
      case "clusters":
        drawClusters(ctx, width, height);
        break;
      case "probability_heatmap":
        drawProbabilityHeatmap(ctx, width, height);
        break;
      default:
        drawDefaultVisualization(ctx, width, height);
    }
  };

  const drawRegressionLine = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;
    const plotHeight = height - 2 * padding;

    // ----- LEFT: raw scattered data (no line) -----
    const leftX0 = padding;
    const leftY0 = padding;

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, leftY0, panelWidth, plotHeight);

    const rawPoints: { x: number; y: number }[] = [];
    for (let i = 0; i < 16; i++) {
      const x = leftX0 + Math.random() * panelWidth;
      const y = leftY0 + Math.random() * plotHeight;
      rawPoints.push({ x, y });
    }

    ctx.fillStyle = "#9ca3af";
    rawPoints.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Just raw X & Y values (no pattern yet)", leftX0, leftY0 + plotHeight + 16);

    // ----- RIGHT: same data organized + best fit line -----
    const rightX0 = midX + padding / 2;
    const rightY0 = padding;

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(rightX0, rightY0, panelWidth, plotHeight);

    // Generate ordered points that roughly follow a line
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i < 14; i++) {
      const t = i / 13;
      const x = rightX0 + t * panelWidth;
      const baseY = rightY0 + plotHeight * (1 - t * 0.8);
      const y = baseY + (Math.random() - 0.5) * (plotHeight * 0.1);
      points.push({ x, y });
    }

    // Draw points
    ctx.fillStyle = "#3b82f6";
    points.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw best fit line
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(rightX0, rightY0 + plotHeight * 0.9);
    ctx.lineTo(rightX0 + panelWidth, rightY0 + plotHeight * 0.1);
    ctx.stroke();

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Model learns a straight line that best fits points", rightX0, rightY0 + plotHeight + 16);
  };

  const drawDecisionBoundary = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;
    const plotHeight = height - 2 * padding;

    // LEFT: mixed unlabeled points
    const leftX0 = padding;
    const leftY0 = padding;
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, leftY0, panelWidth, plotHeight);

    ctx.fillStyle = "#9ca3af";
    for (let i = 0; i < 24; i++) {
      const x = leftX0 + Math.random() * panelWidth;
      const y = leftY0 + Math.random() * plotHeight;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("All points look similar – no clear boundary yet", leftX0, leftY0 + plotHeight + 16);

    // RIGHT: colored classes + learned boundary
    const rightX0 = midX + padding / 2;
    const rightY0 = padding;
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(rightX0, rightY0, panelWidth, plotHeight);

    const plotWidth = panelWidth;

    // decision boundary curve
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = 0; x <= plotWidth; x++) {
      const normalizedX = x / plotWidth;
      const y =
        rightY0 +
        plotHeight * (0.3 + 0.4 * Math.sin(normalizedX * Math.PI * 2));
      if (x === 0) {
        ctx.moveTo(rightX0 + x, y);
      } else {
        ctx.lineTo(rightX0 + x, y);
      }
    }
    ctx.stroke();

    // Class 0 (blue) above boundary
    ctx.fillStyle = "#3b82f6";
    for (let i = 0; i < 20; i++) {
      const x = rightX0 + Math.random() * plotWidth;
      const y = rightY0 + Math.random() * plotHeight * 0.4;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Class 1 (green) below boundary
    ctx.fillStyle = "#10b981";
    for (let i = 0; i < 20; i++) {
      const x = rightX0 + Math.random() * plotWidth;
      const y = rightY0 + plotHeight * 0.6 + Math.random() * plotHeight * 0.4;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Support vectors near boundary
    ctx.fillStyle = "#fbbf24";
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    const sv1 = {
      x: rightX0 + plotWidth * 0.35,
      y: rightY0 + plotHeight * 0.4,
    };
    const sv2 = {
      x: rightX0 + plotWidth * 0.6,
      y: rightY0 + plotHeight * 0.55,
    };
    [sv1, sv2].forEach((sv) => {
      ctx.beginPath();
      ctx.arc(sv.x, sv.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("SVM finds the boundary that best separates colors", rightX0, rightY0 + plotHeight + 16);
  };

  const drawDecisionTree = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const nodeRadius = 25;
    const levelHeight = 100;
    const startY = 50;

    // Draw root node
    const rootX = width / 2;
    const rootY = startY;
    drawTreeNode(ctx, rootX, rootY, nodeRadius, "Feature A\n< 5", "#3b82f6");

    // Level 1 nodes
    const left1X = width / 4;
    const right1X = (3 * width) / 4;
    const y1 = startY + levelHeight;
    drawTreeNode(ctx, left1X, y1, nodeRadius, "Feature B\n< 3", "#10b981");
    drawTreeNode(ctx, right1X, y1, nodeRadius, "Class 1", "#ef4444");

    // Level 2 nodes
    const y2 = startY + 2 * levelHeight;
    drawTreeNode(ctx, left1X / 2, y2, nodeRadius, "Class 0", "#ef4444");
    drawTreeNode(ctx, (left1X + right1X) / 2, y2, nodeRadius, "Class 1", "#ef4444");

    // Draw connections
    ctx.strokeStyle = "#6b7280";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rootX, rootY + nodeRadius);
    ctx.lineTo(left1X, y1 - nodeRadius);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(rootX, rootY + nodeRadius);
    ctx.lineTo(right1X, y1 - nodeRadius);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left1X, y1 + nodeRadius);
    ctx.lineTo(left1X / 2, y2 - nodeRadius);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(left1X, y1 + nodeRadius);
    ctx.lineTo((left1X + right1X) / 2, y2 - nodeRadius);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px sans-serif";
    ctx.fillText("Yes", (rootX + left1X) / 2 - 15, (rootY + y1) / 2);
    ctx.fillText("No", (rootX + right1X) / 2 - 10, (rootY + y1) / 2);
    ctx.fillText("Yes", (left1X + left1X / 2) / 2 - 15, (y1 + y2) / 2);
    ctx.fillText("No", (left1X + (left1X + right1X) / 2) / 2 - 10, (y1 + y2) / 2);
  };

  const drawTreeNode = (
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    radius: number,
    text: string,
    color: string
  ) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#fff";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    const lines = text.split("\n");
    lines.forEach((line, i) => {
      ctx.fillText(line, x, y - 5 + i * 12);
    });
    ctx.textAlign = "left";
  };

  const drawFeatureImportance = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;
    const plotHeight = height - 2 * padding;
    const barCount = 5;

    const features = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"];
    const importances = [0.85, 0.72, 0.45, 0.28, 0.15];

    // LEFT: all features equal (no idea which matters)
    const leftX0 = padding;
    const barWidthLeft = panelWidth / (barCount * 1.8);
    const equalHeight = plotHeight * 0.6;

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, padding, panelWidth, plotHeight);

    for (let i = 0; i < barCount; i++) {
      const x = leftX0 + (panelWidth / barCount) * i + barWidthLeft / 2;
      const y = padding + plotHeight - equalHeight;

      ctx.fillStyle = "#9ca3af";
      ctx.fillRect(x - barWidthLeft / 2, y, barWidthLeft, equalHeight);

      ctx.save();
      ctx.translate(x, padding + plotHeight + 14);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(features[i], 0, 0);
      ctx.restore();
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Before: all features treated the same", leftX0, padding + plotHeight + 32);

    // RIGHT: learned feature importance bars
    const rightX0 = midX + padding / 2;
    const barWidthRight = panelWidth / (barCount * 1.8);

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(rightX0, padding, panelWidth, plotHeight);

    for (let i = 0; i < barCount; i++) {
      const x = rightX0 + (panelWidth / barCount) * i + barWidthRight / 2;
      const barHeight = importances[i] * plotHeight;
      const y = padding + plotHeight - barHeight;

      ctx.fillStyle = i < 2 ? "#10b981" : i < 3 ? "#f59e0b" : "#ef4444";
      ctx.fillRect(x - barWidthRight / 2, y, barWidthRight, barHeight);

      ctx.fillStyle = "#374151";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(importances[i].toFixed(2), x, y - 5);

      ctx.save();
      ctx.translate(x, padding + plotHeight + 14);
      ctx.rotate(-Math.PI / 4);
      ctx.fillStyle = "#6b7280";
      ctx.font = "10px sans-serif";
      ctx.fillText(features[i], 0, 0);
      ctx.restore();
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("After: model ranks which features matter most", rightX0, padding + plotHeight + 32);
  };

  const drawKNN = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 40;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Draw class 0 points (blue)
    ctx.fillStyle = "#3b82f6";
    for (let i = 0; i < 20; i++) {
      const x = padding + Math.random() * plotWidth * 0.5;
      const y = padding + Math.random() * plotHeight;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw class 1 points (green)
    ctx.fillStyle = "#10b981";
    for (let i = 0; i < 20; i++) {
      const x = padding + plotWidth * 0.5 + Math.random() * plotWidth * 0.5;
      const y = padding + Math.random() * plotHeight;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw query point
    const queryX = padding + plotWidth * 0.45;
    const queryY = padding + plotHeight * 0.5;
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.arc(queryX, queryY, 6, 0, Math.PI * 2);
    ctx.fill();

    // Draw K nearest neighbors circle
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(queryX, queryY, 80, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Highlight K nearest neighbors
    ctx.fillStyle = "#fbbf24";
    const neighbors = [
      { x: queryX - 30, y: queryY - 20 },
      { x: queryX + 20, y: queryY - 30 },
      { x: queryX - 20, y: queryY + 25 },
      { x: queryX + 35, y: queryY + 15 },
      { x: queryX - 10, y: queryY - 40 },
    ];
    neighbors.forEach((n) => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  const drawCNNArchitecture = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;

    // LEFT: raw image pixels grid
    const leftX0 = padding;
    const leftY0 = padding + 20;
    const gridSize = 10;
    const cell = Math.min(panelWidth / gridSize, (height - leftY0 - padding) / gridSize);

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, leftY0, panelWidth, cell * gridSize);

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const intensity = Math.floor(150 + Math.random() * 100);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        ctx.fillRect(leftX0 + j * cell, leftY0 + i * cell, cell - 1, cell - 1);
      }
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Raw image pixels", leftX0, leftY0 - 6);

    // RIGHT: CNN layers pipeline
    const layerWidth = 80;
    const layerHeight = 60;
    const spacing = 26;
    const startY = height / 2 - layerHeight / 2;
    const rightStartX = midX + padding / 2;

    const layers = [
      { name: "Conv2D\nfilters", color: "#10b981" },
      { name: "MaxPool\n2x2", color: "#f59e0b" },
      { name: "Conv2D\nmore\nfilters", color: "#10b981" },
      { name: "Flatten", color: "#0ea5e9" },
      { name: "Dense\nlayer", color: "#8b5cf6" },
      { name: "Output\nclasses", color: "#ef4444" },
    ];

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("CNN learns filters & patterns", rightStartX, padding + 8);

    layers.forEach((layer, i) => {
      const x = rightStartX + i * (layerWidth + spacing);
      const y = startY;

      ctx.fillStyle = layer.color;
      ctx.fillRect(x, y, layerWidth, layerHeight);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, layerWidth, layerHeight);

      ctx.fillStyle = "#ffffff";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      const lines = layer.name.split("\n");
      lines.forEach((line, j) => {
        ctx.fillText(line, x + layerWidth / 2, y + layerHeight / 2 - 5 + j * 12);
      });
      ctx.textAlign = "left";

      if (i < layers.length - 1) {
        ctx.strokeStyle = "#6b7280";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + layerWidth, y + layerHeight / 2);
        ctx.lineTo(x + layerWidth + spacing, y + layerHeight / 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x + layerWidth + spacing - 5, y + layerHeight / 2 - 5);
        ctx.lineTo(x + layerWidth + spacing, y + layerHeight / 2);
        ctx.lineTo(x + layerWidth + spacing - 5, y + layerHeight / 2 + 5);
        ctx.stroke();
      }
    });
  };

  const drawClusters = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;
    const plotHeight = height - 2 * padding;

    const colors = ["#3b82f6", "#10b981", "#f59e0b"];

    // LEFT: raw unlabeled points
    const leftX0 = padding;
    const leftY0 = padding;
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, leftY0, panelWidth, plotHeight);

    ctx.fillStyle = "#9ca3af";
    for (let i = 0; i < 45; i++) {
      const x = leftX0 + Math.random() * panelWidth;
      const y = leftY0 + Math.random() * plotHeight;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Before: all images look just like dots", leftX0, leftY0 + plotHeight + 16);

    // RIGHT: K-Means finds clusters
    const rightX0 = midX + padding / 2;
    const rightY0 = padding;
    const plotWidth = panelWidth;

    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(rightX0, rightY0, plotWidth, plotHeight);

    const centers = [
      { x: rightX0 + plotWidth * 0.25, y: rightY0 + plotHeight * 0.3 },
      { x: rightX0 + plotWidth * 0.75, y: rightY0 + plotHeight * 0.7 },
      { x: rightX0 + plotWidth * 0.5, y: rightY0 + plotHeight * 0.5 },
    ];

    centers.forEach((center, i) => {
      ctx.fillStyle = colors[i];
      for (let j = 0; j < 18; j++) {
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.random() * 55;
        const x = center.x + Math.cos(angle) * distance;
        const y = center.y + Math.sin(angle) * distance;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // cluster center
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = colors[i];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(center.x, center.y, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("After: K-Means groups similar points into clusters", rightX0, rightY0 + plotHeight + 16);
  };

  const drawProbabilityHeatmap = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    const padding = 50;
    const midX = width / 2;
    const panelWidth = (width - 3 * padding) / 2;
    const cellSize = 22;
    const rows = 5;
    const cols = 5;

    // LEFT: simple word counts grid (no probabilities)
    const leftX0 = padding;
    const leftY0 = padding;
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX0, leftY0, panelWidth, rows * cellSize + 20);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const count = Math.floor(Math.random() * 9);
        const x = leftX0 + j * cellSize;
        const y = leftY0 + i * cellSize;
        ctx.fillStyle = "#f3f4f6";
        ctx.fillRect(x, y, cellSize - 2, cellSize - 2);
        ctx.fillStyle = "#374151";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(count), x + cellSize / 2, y + cellSize / 2 + 3);
        ctx.textAlign = "left";
      }
    }
    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Counts of words for each class", leftX0, leftY0 + rows * cellSize + 32);

    // RIGHT: Naive Bayes converts to probabilities heatmap
    const rightX0 = midX + padding / 2;
    const rightY0 = padding;
    ctx.strokeStyle = "#d1d5db";
    ctx.lineWidth = 1;
    ctx.strokeRect(rightX0, rightY0, panelWidth, rows * cellSize + 20);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const prob = Math.random();
        const x = rightX0 + j * cellSize;
        const y = rightY0 + i * cellSize;

        const intensity = Math.floor(prob * 255);
        ctx.fillStyle = `rgb(${255 - intensity}, ${intensity}, 120)`;
        ctx.fillRect(x, y, cellSize - 2, cellSize - 2);

        ctx.fillStyle = prob > 0.5 ? "#ffffff" : "#000000";
        ctx.font = "10px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(prob.toFixed(2), x + cellSize / 2, y + cellSize / 2 + 3);
        ctx.textAlign = "left";
      }
    }

    ctx.fillStyle = "#374151";
    ctx.font = "11px sans-serif";
    ctx.fillText("Naive Bayes turns counts into P(word | class)", rightX0, rightY0 + rows * cellSize + 32);
  };

  const drawDefaultVisualization = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.fillStyle = "#f3f4f6";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#6b7280";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Visualization not available", width / 2, height / 2);
    ctx.textAlign = "left";
  };

  if (loadingModel) {
    return (
      <div className="app-container">
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p>Loading model information...</p>
        </div>
      </div>
    );
  }

  if (!model) {
    return (
      <div className="app-container">
        <div style={{ textAlign: "center", padding: "3rem" }}>
          <p>Model not found.</p>
          <button onClick={() => navigate("/models")} style={{ marginTop: "1rem", padding: "0.5rem 1rem" }}>
            Back to Models
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header>
        <div>
          <button
            onClick={() => navigate("/models")}
            style={{
              background: "none",
              border: "none",
              color: "#3b82f6",
              cursor: "pointer",
              fontSize: "0.875rem",
              marginBottom: "1rem",
              padding: 0,
            }}
          >
            ← Back to Models
          </button>
          <p className="eyebrow">Model Details</p>
          <h1>{model.name}</h1>
          <p className="subtitle">{model.description}</p>
        </div>
      </header>

      <section className="panel highlight">
        <div className="panel-header">
          <div>
            <h2>Overview</h2>
          </div>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: "1.5rem",
            marginBottom: "2rem",
          }}
        >
          <div style={{ padding: "1rem", background: "#f9fafb", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              Complexity
            </div>
            <div
              style={{
                fontSize: "1.25rem",
                fontWeight: "600",
                color: complexityColors[model.complexity],
              }}
            >
              {model.complexity}
            </div>
          </div>
          <div style={{ padding: "1rem", background: "#f9fafb", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              Training Speed
            </div>
            <div
              style={{
                fontSize: "1.25rem",
                fontWeight: "600",
                color: speedColors[model.training_speed],
              }}
            >
              {model.training_speed}
            </div>
          </div>
          <div style={{ padding: "1rem", background: "#f9fafb", borderRadius: "8px" }}>
            <div style={{ fontSize: "0.75rem", color: "#6b7280", marginBottom: "0.5rem" }}>
              Interpretability
            </div>
            <div
              style={{
                fontSize: "1.25rem",
                fontWeight: "600",
                color: interpretabilityColors[model.interpretability],
              }}
            >
              {model.interpretability}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "2rem" }}>
          {model.problem_types.map((type) => (
            <span
              key={type}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "8px",
                background: "#e0e7ff",
                color: "#3730a3",
                fontSize: "0.875rem",
                fontWeight: "500",
              }}
            >
              {type}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>How It Works</h2>
            <p className="muted">{model.how_it_works}</p>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Algorithm Steps</h2>
          </div>
        </div>
        <ol style={{ paddingLeft: "1.5rem", lineHeight: "1.8" }}>
          {model.algorithm_steps.map((step, i) => (
            <li key={i} style={{ marginBottom: "0.75rem", color: "#374151" }}>
              {step}
            </li>
          ))}
        </ol>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Visualization</h2>
            <p className="muted">Interactive visualization of how this model works</p>
          </div>
        </div>
        {model.key === "decision_tree" ||
        model.key === "gradient_boosting" ||
        model.key === "random_forest" ||
        model.key === "xgboost" ||
        model.key === "svm" ||
        model.key === "image_cnn" ||
        model.key === "image_kmeans" ||
        model.key === "knn" ||
        model.key === "linear_regression" ||
        model.key === "logistic_regression" ||
        model.key === "naive_bayes" ? (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              padding: "2rem",
            }}
          >
            {model.key === "decision_tree" && (
              <img
                src="/decision_tree_example.png"
                alt="Decision tree with root, internal, and leaf nodes"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "gradient_boosting" && (
              <img
                src="/gradient_boosting_example.gif"
                alt="Gradient boosting: multiple bootstrapped datasets and trees"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "image_cnn" && (
              <img
                src="/cnn_example.png"
                alt="Convolutional neural network: from image to feature maps to output classes"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "image_kmeans" && (
              <img
                src="/kmeans_example.png"
                alt="K-Means clustering: raw mixed data grouped into separate outputs"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "knn" && (
              <img
                src="/knn_example.png"
                alt="k-NN: different values of k and nearest neighbours around a query point"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "linear_regression" && (
              <img
                src="/linear_regression_example.png"
                alt="Linear regression: data points and best fit regression line"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "logistic_regression" && (
              <img
                src="/logistic_regression_example.png"
                alt="Logistic regression: S-shaped boundary separating two classes"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "naive_bayes" && (
              <img
                src="/naive_bayes_example.png"
                alt="Naive Bayes: data separated into groups using Bayes rule"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "random_forest" && (
              <img
                src="/random_forest_example.png"
                alt="Random Forest: many decision trees voting for the final result"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "svm" && (
              <img
                src="/svm_example.png"
                alt="Support Vector Machine: hyperplanes separating different classes"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
            {model.key === "xgboost" && (
              <img
                src="/xgboost_example.png"
                alt="XGBoost: many boosted trees combining their results"
                style={{
                  maxWidth: "100%",
                  height: "auto",
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                }}
              />
            )}
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              padding: "2rem",
              overflowX: "auto",
            }}
          >
            <canvas
              ref={canvasRef}
              width={1000}
              height={450}
              style={{
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
                background: "#fff",
                maxWidth: "100%",
                height: "auto",
              }}
            />
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Advantages</h2>
          </div>
        </div>
        <ul style={{ paddingLeft: "1.5rem", lineHeight: "1.8" }}>
          {model.advantages.map((advantage, i) => (
            <li key={i} style={{ marginBottom: "0.75rem", color: "#374151" }}>
              {advantage}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Disadvantages</h2>
          </div>
        </div>
        <ul style={{ paddingLeft: "1.5rem", lineHeight: "1.8" }}>
          {model.disadvantages.map((disadvantage, i) => (
            <li key={i} style={{ marginBottom: "0.75rem", color: "#374151" }}>
              {disadvantage}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Use Cases</h2>
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          {model.use_cases.map((useCase, i) => (
            <span
              key={i}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "8px",
                background: "#f0fdf4",
                color: "#166534",
                fontSize: "0.875rem",
              }}
            >
              {useCase}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Hyperparameters</h2>
            <p className="muted">Key parameters you can tune to improve performance</p>
          </div>
        </div>
        <div style={{ display: "grid", gap: "1rem" }}>
          {Object.entries(model.hyperparameters).map(([key, value]) => (
            <div
              key={key}
              style={{
                padding: "1rem",
                background: "#f9fafb",
                borderRadius: "8px",
                border: "1px solid #e5e7eb",
              }}
            >
              <div style={{ fontWeight: "600", marginBottom: "0.25rem", color: "#374151" }}>
                {key}
              </div>
              <div style={{ color: "#6b7280", fontSize: "0.875rem" }}>{value}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

