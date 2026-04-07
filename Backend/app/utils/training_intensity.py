"""Training intensity configuration for all ML models."""
from typing import Dict, Literal

TrainingLevel = Literal["less", "medium", "rigorous"]


def get_training_params(level: TrainingLevel) -> Dict:
    """Get general training parameters based on intensity level."""
    if level == "less":
        return {
            "n_estimators": 50,
            "max_depth": 3,
            "cv_folds": 3,
            "test_size": 0.3,  # Larger test set for faster training
        }
    elif level == "medium":
        return {
            "n_estimators": 200,
            "max_depth": 10,
            "cv_folds": 5,
            "test_size": 0.2,
        }
    else:  # rigorous
        return {
            "n_estimators": 500,
            "max_depth": 20,
            "cv_folds": 10,
            "test_size": 0.2,
        }


def get_image_params(level: TrainingLevel) -> Dict:
    """Get image training parameters based on intensity level.
    
    LESS: fewer epochs, higher learning rate, larger batch size, minimal augmentation, frozen layers
    MEDIUM: moderate epochs, standard learning rate, medium batch size, basic augmentation, partial fine-tuning
    RIGOROUS: highest epochs, low learning rate with scheduler, smallest batch size, strong augmentation, full fine-tuning
    """
    if level == "less":
        return {
            "epochs": 10,  # Fewer epochs for faster training
            "batch_size": 32,  # Larger batch size for faster training
            "image_size": (96, 96),  # Smaller images for faster processing
            "learning_rate": 0.002,  # Higher learning rate for faster convergence
            "use_augmentation": True,  # Basic augmentation
            "augmentation_strength": "basic",
            "freeze_backbone": False,
            "use_callbacks": True,  # Early stopping to prevent overfitting
            "use_scheduler": False,  # No scheduler for simplicity
            "early_stopping_patience": 5,  # Less patience - stop earlier
        }
    elif level == "medium":
        return {
            "epochs": 20,  # Moderate epochs
            "batch_size": 16,  # Medium batch size
            "image_size": (160, 160),  # Medium image size
            "learning_rate": 0.0008,  # Moderate learning rate
            "use_augmentation": True,  # Strong augmentation
            "augmentation_strength": "strong",
            "freeze_backbone": False,
            "freeze_layers": 0,
            "use_callbacks": True,
            "use_scheduler": True,  # Learning rate scheduler
            "early_stopping_patience": 8,  # Moderate patience
        }
    else:  # rigorous
        return {
            "epochs": 30,  # Maximum epochs (capped at 30)
            "batch_size": 8,  # Smaller batch size for better gradient updates
            "image_size": (224, 224),  # Largest images for maximum detail
            "learning_rate": 0.0003,  # Lower learning rate for fine-tuning
            "use_augmentation": True,  # Strong augmentation
            "augmentation_strength": "strong",
            "freeze_backbone": False,
            "freeze_layers": 0,
            "use_callbacks": True,
            "use_scheduler": True,  # Learning rate scheduler
            "early_stopping_patience": 12,  # More patience for best results
            "checkpoint_monitor": "val_accuracy",
        }


def get_model_params(model_key: str, level: TrainingLevel, problem_type: str = "classification") -> Dict:
    """Get model-specific parameters based on intensity level.
    
    Returns parameters that can be passed to model constructors or set_params.
    """
    base_params = get_training_params(level)
    
    params = {}
    
    if model_key == "random_forest":
        params = {
            "n_estimators": base_params["n_estimators"],
            "max_depth": base_params["max_depth"],
            "min_samples_split": 5 if level == "less" else (3 if level == "medium" else 2),
            "min_samples_leaf": 2 if level == "less" else (1 if level == "medium" else 1),
            "max_features": "sqrt" if level == "less" else ("log2" if level == "medium" else None),
        }
    elif model_key == "gradient_boosting":
        params = {
            "n_estimators": base_params["n_estimators"],
            "max_depth": base_params["max_depth"],
            "learning_rate": 0.1 if level == "less" else (0.05 if level == "medium" else 0.01),
            "subsample": 0.8 if level == "less" else (0.9 if level == "medium" else 1.0),
            "min_samples_split": 5 if level == "less" else (3 if level == "medium" else 2),
        }
    elif model_key == "decision_tree":
        params = {
            "max_depth": base_params["max_depth"],
            "min_samples_split": 10 if level == "less" else (5 if level == "medium" else 2),
            "min_samples_leaf": 5 if level == "less" else (2 if level == "medium" else 1),
            "max_features": "sqrt" if level == "less" else ("log2" if level == "medium" else None),
        }
    elif model_key == "xgboost":
        params = {
            "n_estimators": base_params["n_estimators"],
            "max_depth": base_params["max_depth"],
            "learning_rate": 0.1 if level == "less" else (0.05 if level == "medium" else 0.01),
            "subsample": 0.8 if level == "less" else (0.9 if level == "medium" else 1.0),
            "colsample_bytree": 0.8 if level == "less" else (0.9 if level == "medium" else 1.0),
            "min_child_weight": 3 if level == "less" else (1 if level == "medium" else 1),
            "gamma": 0 if level == "less" else (0.1 if level == "medium" else 0.2),
        }
    elif model_key == "svm":
        params = {
            "C": 1.0 if level == "less" else (10.0 if level == "medium" else 100.0),
            "gamma": "scale" if level == "less" else ("auto" if level == "medium" else "auto"),
            "kernel": "rbf",
            "max_iter": 500 if level == "less" else (1000 if level == "medium" else 2000),
        }
    elif model_key == "knn":
        params = {
            "n_neighbors": 5 if level == "less" else (7 if level == "medium" else 9),
            "weights": "uniform" if level == "less" else ("distance" if level == "medium" else "distance"),
            "algorithm": "auto",
            "leaf_size": 30 if level == "less" else (20 if level == "medium" else 10),
        }
    elif model_key == "logistic_regression":
        params = {
            "max_iter": 500 if level == "less" else (1000 if level == "medium" else 2000),
            "C": 1.0 if level == "less" else (10.0 if level == "medium" else 100.0),
            "solver": "lbfgs" if level == "less" else ("lbfgs" if level == "medium" else "liblinear"),
            "penalty": "l2",
        }
    elif model_key == "naive_bayes":
        params = {
            "var_smoothing": 1e-9 if level == "less" else (1e-8 if level == "medium" else 1e-10),
        }
    elif model_key == "linear_regression":
        params = {
            "fit_intercept": True,
            "normalize": False,
        }
    
    return params
