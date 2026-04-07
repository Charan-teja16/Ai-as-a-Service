import numpy as np
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    """DSA-style wrapper around sklearn LinearRegression."""

    name = "Linear Regression"
    key = "linear_regression"
    problem_type = "regression"

    def __init__(self, **kwargs) -> None:
        self.model = LinearRegression(**kwargs)
    
    def set_params(self, **params):
        """Update model parameters."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        """Fit the model with proper input validation and conversion."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with proper output type conversion."""
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict(X_arr)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate score with proper type handling."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        score_value = self.model.score(X_arr, y_arr)
        # Ensure we return a proper float (not numpy float)
        return float(score_value)


if __name__ == "__main__":
    X_demo = np.array([[1], [2], [3], [4]])
    y_demo = np.array([2, 3, 4, 5])
    model = LinearRegressionModel().fit(X_demo, y_demo)
    print("Predictions:", model.predict(np.array([[5], [6]])))
    print("Score:", model.score(X_demo, y_demo))

