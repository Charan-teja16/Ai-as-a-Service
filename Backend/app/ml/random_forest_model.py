import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestModel:
    """Random forest ensemble with DSA-style API."""

    name = "Random Forest"
    key = "random_forest"
    problem_type = "both"

    def __init__(self, mode: str = "classification", **kwargs) -> None:
        self.mode = mode
        params = dict(n_estimators=200, random_state=42)
        params.update(kwargs)
        self.model = (
            RandomForestClassifier(**params)
            if mode == "classification"
            else RandomForestRegressor(**params)
        )
    
    def set_params(self, **params):
        """Update model parameters."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Fit the model with proper input validation and conversion."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        # Convert y to integers if it's numeric strings or ensure it's numeric
        if y_arr.dtype == object or y_arr.dtype.kind in ['U', 'S']:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr)
        else:
            y_arr = y_arr.astype(np.int64) if y_arr.dtype.kind in ['i', 'u'] else y_arr
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with proper output type conversion."""
        X_arr = np.asarray(X, dtype=np.float64)
        predictions = self.model.predict(X_arr)
        # Ensure predictions are integers for classification
        if self.mode == "classification":
            return predictions.astype(np.int64)
        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate score with proper type handling."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        # Convert y to integers if needed
        if y_arr.dtype == object or y_arr.dtype.kind in ['U', 'S']:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_arr = le.fit_transform(y_arr)
        else:
            y_arr = y_arr.astype(np.int64) if y_arr.dtype.kind in ['i', 'u'] else y_arr
        score_value = self.model.score(X_arr, y_arr)
        # Ensure we return a proper float (not numpy float)
        return float(score_value)


if __name__ == "__main__":
    X_demo = np.random.rand(20, 3)
    y_demo = np.random.randint(0, 2, 20)
    clf = RandomForestModel().fit(X_demo, y_demo)
    print("Classification score:", clf.score(X_demo, y_demo))
    reg = RandomForestModel(mode="regression").fit(X_demo, np.random.rand(20))
    print("Regression score:", reg.score(X_demo, np.random.rand(20)))

