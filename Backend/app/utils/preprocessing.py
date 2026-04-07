from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class PreprocessResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: Pipeline
    feature_names: List[str]
    label_encoder: Optional[LabelEncoder]
    y_train_original: np.ndarray
    y_test_original: np.ndarray


def preprocess(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessResult:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    label_encoder: Optional[LabelEncoder] = None
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
        else:
            y_encoded = y
    else:
        if not pd.api.types.is_numeric_dtype(y):
            y_numeric = pd.to_numeric(y, errors="coerce")
            if y_numeric.isnull().any():
                raise ValueError("Regression target column must be numeric.")
            y_encoded = y_numeric
        else:
            y_encoded = y

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))],
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    pipeline.fit(X)
    X_processed = pipeline.transform(X)

    if hasattr(pipeline.named_steps["preprocessor"], "get_feature_names_out"):
        feature_names = (
            pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        )
    else:
        feature_names = list(range(X_processed.shape[1]))

    stratify = (
        y_encoded if problem_type == "classification" and y_encoded.nunique() > 1 else None
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_orig,
        y_test_orig,
    ) = train_test_split(
        X_processed,
        y_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return PreprocessResult(
        X_train=np.asarray(X_train),
        X_test=np.asarray(X_test),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        pipeline=pipeline,
        feature_names=feature_names,
        label_encoder=label_encoder,
        y_train_original=np.asarray(y_train_orig),
        y_test_original=np.asarray(y_test_orig),
    )

