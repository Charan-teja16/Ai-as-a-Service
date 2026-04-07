from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol

import numpy as np


ProblemType = Literal["classification", "regression"]


class BaseModel(Protocol):
    """DSA-style estimator interface."""

    name: str
    key: str
    problem_type: ProblemType | Literal["both"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        ...


@dataclass
class DemoResult:
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

