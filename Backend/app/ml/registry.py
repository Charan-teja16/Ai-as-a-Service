from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .decision_tree_model import DecisionTreeModel
from .gradient_boosting_model import GradientBoostingModel
from .knn_model import KNearestNeighborsModel
from .linear_regression_model import LinearRegressionModel
from .logistic_regression_model import LogisticRegressionModel
from .naive_bayes_model import NaiveBayesModel
from .random_forest_model import RandomForestModel
from .svm_model import SupportVectorModel
from .xgboost_model import XGBoostModel


class ModelRegistry:
    """Central place to discover and instantiate estimator classes."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict] = {
            "linear_regression": {
                "name": "Linear Regression",
                "problem_types": ["regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": LinearRegressionModel,
            },
            "logistic_regression": {
                "name": "Logistic Regression",
                "problem_types": ["classification"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": LogisticRegressionModel,
            },
            "naive_bayes": {
                "name": "Naive Bayes",
                "problem_types": ["classification"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": NaiveBayesModel,
            },
            "decision_tree": {
                "name": "Decision Tree",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": DecisionTreeModel,
            },
            "random_forest": {
                "name": "Random Forest",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": RandomForestModel,
            },
            "svm": {
                "name": "Support Vector Machine",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": SupportVectorModel,
            },
            "knn": {
                "name": "K-Nearest Neighbors",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": KNearestNeighborsModel,
            },
            "gradient_boosting": {
                "name": "Gradient Boosting",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": GradientBoostingModel,
            },
            "xgboost": {
                "name": "XGBoost",
                "problem_types": ["classification", "regression"],
                "dataset_modes": ["csv"],  # Only for CSV
                "factory": XGBoostModel,
            },
            # Image-specific model keys are handled separately in the training service.
            "image_cnn": {
                "name": "Convolutional Neural Network (CNN)",
                "problem_types": ["classification"],
                "dataset_modes": ["supervised"],  # Only for labeled images
                "factory": LogisticRegressionModel,  # placeholder, not used
            },
            "image_kmeans": {
                "name": "K-Means Clustering",
                "problem_types": ["unsupervised"],
                "dataset_modes": ["unsupervised"],  # Only for unlabeled images
                "factory": LogisticRegressionModel,  # placeholder, not used
            },
        }

    def list_models(self, problem_type: str | None = None, dataset_mode: str | None = None) -> List[Dict]:
        filtered = []
        for key, meta in sorted(self._registry.items(), key=lambda item: item[0]):
            # Filter by problem type if specified
            if problem_type and problem_type not in meta.get("problem_types", []):
                continue
            # Filter by dataset mode if specified
            if dataset_mode and dataset_mode not in meta.get("dataset_modes", []):
                continue
            filtered.append({**meta, "key": key})
        return filtered

    def resolve(self, model_key: str, problem_type: str | None = None, intensity: str = "medium", model_params: Dict | None = None):
        if model_key not in self._registry:
            raise ValueError(f"Model {model_key} is not registered.")
        meta = self._registry[model_key]
        if problem_type and problem_type not in meta["problem_types"]:
            raise ValueError(
                f"Model {model_key} does not support {problem_type} problems."
            )
        factory = meta["factory"]
        instance = None
        
        # Try to pass model_params during instantiation if model supports it
        init_kwargs = {}
        if model_params:
            init_kwargs = model_params.copy()
        
        # Try creating instance with parameters, falling back gracefully
        if problem_type and problem_type in ("classification", "regression"):
            try:
                # First try with mode parameter (for models that support it)
                instance = factory(mode=problem_type, **init_kwargs)
            except TypeError:
                try:
                    # Then try without mode but with kwargs
                    instance = factory(**init_kwargs)
                except TypeError:
                    # Finally try with just mode, no kwargs
                    try:
                        instance = factory(mode=problem_type)
                    except TypeError:
                        # Last resort: no parameters
                        instance = factory()
        else:
            try:
                instance = factory(**init_kwargs)
            except TypeError:
                instance = factory()
        
        # Apply parameters via set_params if they weren't applied during init
        # This is important for parameters that can be changed after initialization
        if model_params and hasattr(instance, "set_params"):
            try:
                instance.set_params(**model_params)
            except Exception:
                # If set_params fails on wrapper, try on underlying sklearn model
                if hasattr(instance, "model") and hasattr(instance.model, "set_params"):
                    try:
                        instance.model.set_params(**model_params)
                    except Exception:
                        # Some parameters can't be changed after init, that's OK
                        pass
        
        return instance

    def get_name(self, model_key: str) -> str:
        return self._registry[model_key]["name"]


REGISTRY = ModelRegistry()

