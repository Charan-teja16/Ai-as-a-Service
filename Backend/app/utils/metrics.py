from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
)


@dataclass
class EvaluationBundle:
    metrics: Dict[str, float]
    confusion: Optional[np.ndarray]
    roc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    residuals: Optional[np.ndarray]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def evaluate(
    problem_type: str,
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Optional[np.ndarray] = None,
) -> EvaluationBundle:
    # Ensure inputs are numpy arrays with proper types
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    # Convert to integers for classification (ensure no float labels)
    if problem_type == "classification":
        # Ensure both arrays are integers for proper comparison
        if y_true_arr.dtype.kind == 'f':  # float
            y_true_arr = y_true_arr.astype(np.int64)
        if y_pred_arr.dtype.kind == 'f':  # float
            y_pred_arr = y_pred_arr.astype(np.int64)
    
    if problem_type == "classification":
        # Calculate accuracy with proper type handling
        acc = accuracy_score(y_true_arr, y_pred_arr)
        metrics = {
            "accuracy": float(acc),  # Explicitly convert to Python float
            "precision": float(
                precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
            "rmse": None,
            "mae": None,
            "r2": None,
            "silhouette": None,
        }
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        roc_payload = None
        if y_proba is not None and len(np.unique(y_true_arr)) == 2:
            fpr, tpr, thresholds = roc_curve(y_true_arr, y_proba[:, 1])
            roc_payload = (fpr, tpr, thresholds)
        residuals = None
    elif problem_type == "regression":
        rmse = float(mean_squared_error(y_true_arr, y_pred_arr, squared=False))
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "r2": float(r2_score(y_true_arr, y_pred_arr)),
            "silhouette": None,
        }
        cm = None
        roc_payload = None
        residuals = y_true_arr - y_pred_arr
    else:  # unsupervised
        # Expect y_true_arr to optionally be feature embeddings for silhouette computation
        try:
            X_embed = y_true_arr
            if X_embed.ndim == 1:
                X_embed = X_embed.reshape(-1, 1)
            silhouette = float(silhouette_score(X_embed, y_pred_arr))
        except Exception:
            silhouette = None
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "rmse": None,
            "mae": None,
            "r2": None,
            "silhouette": silhouette,
        }
        cm = None
        roc_payload = None
        residuals = None

    return EvaluationBundle(
        metrics=metrics,
        confusion=cm,
        roc=roc_payload,
        residuals=residuals,
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_proba=y_proba,
    )

