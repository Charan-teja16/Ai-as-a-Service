import numpy as np
from sklearn.naive_bayes import GaussianNB


class NaiveBayesModel:
    """Gaussian Naive Bayes classifier with DSA-style API."""

    name = "Naive Bayes"
    key = "naive_bayes"
    problem_type = "classification"

    def __init__(self, **kwargs) -> None:
        self.model = GaussianNB(**kwargs)
    
    def set_params(self, **params):
        """Update model parameters."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesModel":
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
        # Ensure predictions are integers
        return predictions.astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X_arr)

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
    X_demo = np.array([[0], [1], [1], [0]])
    y_demo = np.array([0, 1, 1, 0])
    model = NaiveBayesModel().fit(X_demo, y_demo)
    print("Predictions:", model.predict(X_demo))
    print("Score:", model.score(X_demo, y_demo))

