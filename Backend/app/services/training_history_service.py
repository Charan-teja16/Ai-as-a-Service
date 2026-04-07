"""Training history service for tracking user training activities."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .. import config
from ..utils.state import JSONStore


@dataclass
class TrainingHistoryRecord:
    history_id: str
    user_id: str
    dataset_id: str
    dataset_name: str
    target_column: str
    problem_type: str
    dataset_mode: str = "csv"  # csv | supervised | unsupervised (default for backward compatibility)
    model_key: Optional[str] = None  # None for train-all
    model_name: Optional[str] = None
    model_id: Optional[str] = None  # Link to model for predictions
    intensity: str = "medium"
    metrics: dict = field(default_factory=dict)
    created_at: str = ""
    report_id: Optional[str] = None
    pkl_path: Optional[str] = None  # Path to .pkl file
    h5_path: Optional[str] = None  # Path to .h5 file


class TrainingHistoryService:
    def __init__(self):
        config.ensure_directories()
        self._store = JSONStore(config.STORAGE_DIR / "training_history_index.json")

    def add_history(
        self,
        user_id: str,
        dataset_id: str,
        dataset_name: str,
        dataset_mode: str,
        target_column: str,
        problem_type: str,
        model_key: Optional[str] = None,
        model_name: Optional[str] = None,
        model_id: Optional[str] = None,
        intensity: str = "medium",
        metrics: Optional[dict] = None,
        report_id: Optional[str] = None,
        pkl_path: Optional[str] = None,
        h5_path: Optional[str] = None,
    ) -> TrainingHistoryRecord:
        """Add a training history record."""
        import uuid
        history_id = str(uuid.uuid4())
        record = TrainingHistoryRecord(
            history_id=history_id,
            user_id=user_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_mode=dataset_mode,
            target_column=target_column,
            problem_type=problem_type,
            model_key=model_key,
            model_name=model_name,
            model_id=model_id,
            intensity=intensity,
            metrics=metrics or {},
            created_at=datetime.utcnow().isoformat(),
            report_id=report_id,
            pkl_path=pkl_path,
            h5_path=h5_path,
        )
        self._store.set(history_id, record.__dict__)
        return record

    def get_user_history(self, user_id: str, limit: int = 20) -> List[TrainingHistoryRecord]:
        """Get training history for a user, most recent first."""
        history = []
        for payload in self._store.list():
            if payload.get("user_id") == user_id:
                # Handle backward compatibility: if dataset_mode is missing, default to "csv"
                if "dataset_mode" not in payload:
                    payload["dataset_mode"] = "csv"
                try:
                    history.append(TrainingHistoryRecord(**payload))
                except TypeError as e:
                    # Skip records that can't be loaded (missing required fields)
                    import logging
                    logging.warning(f"Skipping invalid history record {payload.get('history_id')}: {e}")
                    continue
        # Sort by created_at descending
        history.sort(key=lambda x: x.created_at, reverse=True)
        return history[:limit]



