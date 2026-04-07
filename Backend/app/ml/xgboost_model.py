import numpy as np
from xgboost import XGBClassifier, XGBRegressor


class XGBoostModel:
    """Optional XGBoost model using the same DSA-style API."""

    name = "XGBoost"
    key = "xgboost"
    problem_type = "both"

    def __init__(self, mode: str = "classification", **kwargs) -> None:
        self.mode = mode
        params = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        # Override with any provided parameters
        params.update(kwargs)
        self.model = (
            XGBClassifier(
                eval_metric="logloss",
                **params,
            )
            if mode == "classification"
            else XGBRegressor(
                objective="reg:squarederror",
                **params,
            )
        )
    
    def set_params(self, **params):
        """Update model parameters."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X: np.ndarray):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(np.asarray(X))
        raise AttributeError("Model does not support probability outputs.")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.model.score(np.asarray(X), np.asarray(y)))


if __name__ == "__main__":
    X_demo = np.random.rand(40, 5)
    y_demo = np.random.randint(0, 2, 40)
    model = XGBoostModel().fit(X_demo, y_demo)
    print("XGBoost score:", model.score(X_demo, y_demo))

