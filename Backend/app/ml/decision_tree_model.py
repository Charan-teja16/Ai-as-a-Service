import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DecisionTreeModel:
    """Decision Tree that adapts to classification or regression."""

    name = "Decision Tree"
    key = "decision_tree"
    problem_type = "both"

    def __init__(self, mode: str = "classification", **kwargs) -> None:
        self.mode = mode
        params = dict(max_depth=8, random_state=42)
        params.update(kwargs)
        self.model = (
            DecisionTreeClassifier(**params)
            if mode == "classification"
            else DecisionTreeRegressor(**params)
        )
    
    def set_params(self, **params):
        """Update model parameters."""
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeModel":
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
    X_demo = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_demo = np.array([0, 1, 1, 0])
    clf = DecisionTreeModel(mode="classification").fit(X_demo, y_demo)
    print("Classification preds:", clf.predict(X_demo))
    X_reg = np.array([[1], [2], [3], [4]])
    y_reg = np.array([2, 4, 6, 8])
    reg = DecisionTreeModel(mode="regression").fit(X_reg, y_reg)
    print("Regression preds:", reg.predict(X_reg))

