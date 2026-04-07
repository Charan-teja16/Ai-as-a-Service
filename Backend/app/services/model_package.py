from dataclasses import dataclass
from typing import Any, List, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class ModelPackage:
    estimator: Any
    pipeline: Pipeline
    feature_names: List[str]
    target_column: str
    problem_type: str
    columns: List[str]
    feature_hints: List[dict]
    label_encoder: Optional[LabelEncoder] = None

