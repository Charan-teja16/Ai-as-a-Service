from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .. import config

config.ensure_directories()


def _path(name: str) -> Path:
    path = config.PLOTS_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion(confusion_matrix, model_id: str) -> Optional[str]:
    if confusion_matrix is None:
        return None
    try:
        plt.figure(figsize=(4, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        path = _path(f"{model_id}_confusion")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        # If anything goes wrong while plotting, fail gracefully
        plt.close()
        return None


def plot_roc_curve(roc_payload, model_id: str) -> Optional[str]:
    if roc_payload is None:
        return None
    try:
        fpr, tpr, _ = roc_payload
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        path = _path(f"{model_id}_roc")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        plt.close()
        return None


def plot_residuals(residuals, model_id: str) -> Optional[str]:
    if residuals is None:
        return None
    try:
        plt.figure(figsize=(5, 3))
        plt.plot(residuals, marker="o", linestyle="-")
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Residuals")
        plt.xlabel("Sample")
        plt.ylabel("Error")
        path = _path(f"{model_id}_residuals")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        plt.close()
        return None


def plot_feature_importance(model, feature_names, model_id: str) -> Optional[str]:
    importance = None
    if hasattr(model, "model"):
        estimator = model.model
    else:
        estimator = model
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.ravel(estimator.coef_)
    if importance is None:
        return None
    try:
        plt.figure(figsize=(6, 4))
        idx = np.argsort(importance)[-20:]
        # Guard against any mismatch between indices and feature names
        feature_array = np.array(feature_names)
        safe_idx = idx[idx < feature_array.shape[0]]
        names = feature_array[safe_idx]
        values = importance[safe_idx]
        sns.barplot(x=values, y=names)
        plt.title("Feature Importance")
        path = _path(f"{model_id}_feature_importance")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        plt.close()
        return None


def plot_tree_structure(model, feature_names, model_id: str) -> Optional[str]:
    """
    Draw a compact decision-tree style diagram when the estimator exposes a
    tree structure (e.g. DecisionTree, RandomForest, GradientBoosting).
    For non-tree models this safely returns None.
    """
    try:
        from sklearn import tree as sktree  # type: ignore
    except Exception:
        # If sklearn plotting utilities are not available, skip gracefully
        return None

    # Unwrap underlying estimator if we are using a wrapper
    estimator = model.model if hasattr(model, "model") else model

    # Only plot if estimator has a tree_ attribute (classic tree-based models)
    if not hasattr(estimator, "tree_"):
        return None

    try:
        plt.figure(figsize=(10, 6))
        # Draw the full tree (no max_depth) so the user can see every split
        sktree.plot_tree(
            estimator,
            feature_names=list(feature_names) if feature_names is not None else None,
            filled=True,
            fontsize=6,
        )
        plt.title("Decision Tree Structure")
        path = _path(f"{model_id}_tree")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        plt.close()
        return None


def compute_shap_chart(model, X_sample, feature_names, model_id: str) -> Optional[str]:
    try:
        import shap  # type: ignore
    except ImportError:
        return None
    try:
        estimator = model.model if hasattr(model, "model") else model
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X_sample[:200])
        path = _path(f"{model_id}_shap")
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:  # pragma: no cover - shap is optional
        return None


def build_plot_bundle(
    model,
    eval_bundle,
    feature_names,
    artifact_id: str,
    X_reference: Optional[np.ndarray],
) -> Dict[str, Optional[str]]:
    return {
        "confusion_plot": plot_confusion(eval_bundle.confusion, artifact_id),
        "roc_plot": plot_roc_curve(eval_bundle.roc, artifact_id),
        "residual_plot": plot_residuals(eval_bundle.residuals, artifact_id),
        "feature_importance_plot": plot_feature_importance(model, feature_names, artifact_id),
        "shap_plot": compute_shap_chart(model, X_reference, feature_names, artifact_id)
        if X_reference is not None
        else None,
        # Extra: full decision tree diagram for tree-based models (if available)
        "tree_plot": plot_tree_structure(model, feature_names, artifact_id),
    }

