import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from .. import config
from ..schemas import FeatureHint, MetricPayload, ModelSummary, ProblemType
from ..utils.state import JSONStore


@dataclass
class StoredModel:
    model_id: str
    model_key: str
    model_name: str
    dataset_id: str
    user_id: str
    problem_type: ProblemType
    metrics: Dict
    path: str
    h5_path: str
    created_at: str
    report_id: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    feature_hints: List[Dict] = field(default_factory=list)


class ModelManager:
    def __init__(self) -> None:
        config.ensure_directories()
        self._store = JSONStore(config.STORAGE_DIR / "models_index.json")

    def save_model(
        self,
        model_obj: object,
        model_key: str,
        model_name: str,
        dataset_id: str,
        user_id: str,
        problem_type: ProblemType,
        metrics: Dict,
        report_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
        feature_hints: Optional[List[Dict]] = None,
        h5_override: Optional[Path] = None,
    ) -> StoredModel:
        model_id = str(uuid.uuid4())
        pickle_bytes = pickle.dumps(model_obj)
        path = config.MODELS_DIR / f"{model_id}.pkl"
        with open(path, "wb") as fh:
            fh.write(pickle_bytes)
        if h5_override:
            h5_path = Path(h5_override)
        else:
            h5_path = config.MODELS_DIR / f"{model_id}.h5"
            with h5py.File(h5_path, "w") as h5_file:
                h5_file.create_dataset(
                    "model_pickle",
                    data=np.frombuffer(pickle_bytes, dtype="uint8"),
                )
        stored = StoredModel(
            model_id=model_id,
            model_key=model_key,
            model_name=model_name,
            dataset_id=dataset_id,
            user_id=user_id,
            problem_type=problem_type,
            metrics=metrics,
            path=str(path),
            h5_path=str(h5_path),
            created_at=datetime.utcnow().isoformat(),
            report_id=report_id,
            columns=columns or [],
            feature_hints=feature_hints or [],
        )
        self._store.set(model_id, stored.__dict__)
        return stored

    def get(self, model_id: str) -> StoredModel:
        payload: Dict = self._store.get(model_id)
        if not payload:
            raise ValueError(f"Model {model_id} not found")
        return StoredModel(**payload)

    def attach_report(self, model_id: str, report_id: str) -> None:
        stored = self.get(model_id)
        stored.report_id = report_id
        self._store.set(model_id, stored.__dict__)

    def list(self, user_id: Optional[str] = None) -> List[ModelSummary]:
        summaries: List[ModelSummary] = []
        for payload in self._store.list():
            stored = StoredModel(**payload)
            if user_id and stored.user_id != user_id:
                continue
            summaries.append(
                ModelSummary(
                    model_id=stored.model_id,
                    model_key=stored.model_key,
                    model_name=stored.model_name,
                    dataset_id=stored.dataset_id,
                    problem_type=stored.problem_type,
                    metrics=MetricPayload(**stored.metrics),
                    rank=None,
                    created_at=datetime.fromisoformat(stored.created_at),
                    download_url=f"/models/download/{stored.model_id}",
                    download_h5_url=f"/models/download/{stored.model_id}/h5",
                    report_url=f"/report/download/{stored.report_id}"
                    if stored.report_id
                    else None,
                    columns=stored.columns or [],
                    feature_hints=[FeatureHint(**hint) for hint in stored.feature_hints],
                )
            )
        return summaries

    def load_pickle(self, model_id: str) -> object:
        stored = self.get(model_id)
        with open(stored.path, "rb") as fh:
            return pickle.load(fh)

