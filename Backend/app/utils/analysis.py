from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


TARGET_KEYWORDS = [
    "target",
    "label",
    "class",
    "outcome",
    "result",
    "price",
    "status",
    "response",
    # Extra hints for common ML datasets
    "species",
    "diagnosis",
    "survived",
    "churn",
]


def detect_target_column(df: pd.DataFrame) -> Tuple[str, float, str, str]:
    """
    Return (target_column, confidence, reason, problem_type).

    Heuristic scoring based on column name semantics and statistical profile.
    """
    best_column = df.columns[-1]
    best_score = -1.0
    best_reason = "Defaulted to last column."
    best_problem_type = "regression"
    row_count = len(df)

    for column in df.columns:
        score = 0.0
        col = df[column]
        name_lower = column.lower()
        matches = [kw for kw in TARGET_KEYWORDS if kw in name_lower]
        if matches:
            score += 0.4 + 0.05 * len(matches)
        # How many distinct values compared to total rows
        unique_ratio = col.nunique() / max(1, row_count)

        # Treat "categorical-style" columns (including typical targets like species)
        # as much stronger target candidates than fully numeric continuous ones.
        is_categorical = col.dtype == object or unique_ratio <= 0.1
        if is_categorical:
            # Stronger base score for categorical targets
            score += 0.4
            problem_type = "classification"
            # Extra boost if it's the last column (very common pattern)
            if column == df.columns[-1]:
                score += 0.2
        else:
            # Continuous / many unique values → more likely regression
            score += 0.25
            problem_type = "regression"

        # Small bias towards the last column overall
        if column == df.columns[-1]:
            score += 0.05

        if abs(unique_ratio - 0.5) < 0.1:
            score -= 0.05
        std = float(np.std(col)) if np.issubdtype(col.dtype, np.number) else 0
        if std == 0 and col.dtype != object:
            score -= 0.3

        # Build explanation string
        reason_parts = []
        if matches:
            reason_parts.append(f"Keywords {matches} boosted confidence.")
        if is_categorical:
            reason_parts.append("Column looks categorical (good target for classification).")
        else:
            reason_parts.append("Column looks continuous (good target for regression).")
        reason_parts.append(f"Unique ratio={unique_ratio:.2f}.")
        reason_parts.append(f"Problem type guessed as {problem_type}.")
        reason = " ".join(reason_parts)

        if score > best_score:
            best_column = column
            best_score = score
            best_reason = reason
            best_problem_type = problem_type

    confidence = max(0.2, min(0.95, best_score))
    return best_column, confidence, best_reason, best_problem_type


def infer_problem_type(series: pd.Series) -> str:
    """Infer problem type from target series distribution."""
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"

