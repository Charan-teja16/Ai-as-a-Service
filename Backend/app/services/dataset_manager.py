import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import tempfile
import uuid
import zipfile

import pandas as pd
from PIL import Image

from .. import config
from ..utils.state import JSONStore


MAX_DATASET_MB = 300
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetRecord:
    dataset_id: str
    filename: str
    path: str
    user_id: str
    mode: str = "csv"  # csv | supervised | unsupervised
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    target_column: Optional[str] = None
    total_images: int = 0
    classes: List[Dict] = field(default_factory=list)  # [{"label": str, "count": int}]


class DatasetManager:
    """Persist uploaded datasets and expose convenience helpers."""

    def __init__(self) -> None:
        config.ensure_directories()
        self._store = JSONStore(config.STORAGE_DIR / "datasets_index.json")

    # CSV ---------------------------------------------------------------------
    def register(self, file: bytes, filename: str, user_id: str) -> DatasetRecord:
        self._enforce_size(file)
        dataset_id = str(uuid.uuid4())
        dataset_path = config.DATASETS_DIR / f"{user_id}_{dataset_id}.csv"
        dataset_path.write_bytes(file)
        df = pd.read_csv(dataset_path)
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            path=str(dataset_path),
            columns=list(df.columns),
            row_count=len(df),
            user_id=user_id,
            mode="csv",
        )
        self._store.set(dataset_id, record.__dict__)
        return record

    # Images ------------------------------------------------------------------
    def register_supervised_images(self, file: bytes, filename: str, user_id: str) -> DatasetRecord:
        root, total_images, classes = self._extract_and_validate_images(file, filename, supervised=True)
        dataset_id = str(uuid.uuid4())
        target_dir = config.DATASETS_DIR / f"{user_id}_{dataset_id}"
        shutil.move(str(root), target_dir)
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            path=str(target_dir),
            user_id=user_id,
            mode="supervised",
            total_images=total_images,
            classes=[{"label": c[0], "count": c[1]} for c in classes],
        )
        self._store.set(dataset_id, record.__dict__)
        return record

    def register_unsupervised_images(self, file: bytes, filename: str, user_id: str) -> DatasetRecord:
        root, total_images, classes = self._extract_and_validate_images(file, filename, supervised=False)
        dataset_id = str(uuid.uuid4())
        target_dir = config.DATASETS_DIR / f"{user_id}_{dataset_id}"
        shutil.move(str(root), target_dir)
        record = DatasetRecord(
            dataset_id=dataset_id,
            filename=filename,
            path=str(target_dir),
            user_id=user_id,
            mode="unsupervised",
            total_images=total_images,
            classes=[{"label": c[0], "count": c[1]} for c in classes],
        )
        self._store.set(dataset_id, record.__dict__)
        return record

    # Shared helpers ----------------------------------------------------------
    def list_user_datasets(self, user_id: str) -> list[DatasetRecord]:
        """List all datasets for a user."""
        datasets = []
        for payload in self._store.list():
            if payload.get("user_id") == user_id:
                datasets.append(DatasetRecord(**payload))
        return datasets

    def get(self, dataset_id: str) -> DatasetRecord:
        payload: Dict = self._store.get(dataset_id)
        if not payload:
            raise ValueError(f"Dataset {dataset_id} not found")
        return DatasetRecord(**payload)

    def update_target(self, dataset_id: str, target_column: str) -> None:
        record = self.get(dataset_id)
        record.target_column = target_column
        self._store.set(dataset_id, record.__dict__)

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        record = self.get(dataset_id)
        if record.mode != "csv":
            raise ValueError("Dataset is not a CSV dataset.")
        return pd.read_csv(Path(record.path))

    def get_image_paths(self, dataset_id: str) -> Tuple[Path, DatasetRecord]:
        record = self.get(dataset_id)
        if record.mode not in ("supervised", "unsupervised"):
            raise ValueError("Dataset is not an image dataset.")
        return Path(record.path), record

    # Internal utilities ------------------------------------------------------
    def _enforce_size(self, file: bytes) -> None:
        size_mb = len(file) / (1024 * 1024)
        if size_mb > MAX_DATASET_MB:
            raise ValueError(f"Dataset too large ({size_mb:.1f} MB). Limit is {MAX_DATASET_MB} MB.")

    def _extract_and_validate_images(self, file: bytes, filename: str, supervised: bool) -> Tuple[Path, int, List[Tuple[str, int]]]:
        self._enforce_size(file)
        temp_dir = Path(tempfile.mkdtemp())
        zip_path = temp_dir / filename
        zip_path.write_bytes(file)
        if not zipfile.is_zipfile(zip_path):
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("Image uploads must be a zip file containing folders of images.")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        # Locate root folder (ignore __MACOSX etc.)
        candidates = [p for p in temp_dir.iterdir() if p.is_dir() and not p.name.startswith("__MACOSX")]
        if not candidates:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("No folders found in zip. Please provide folder-based images.")
        root = candidates[0]
        if len(candidates) > 1 and supervised is False:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError("Unsupervised mode requires exactly one folder containing images.")
        if supervised:
            class_dirs = [p for p in root.iterdir() if p.is_dir()]
            if len(class_dirs) < 2:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError("Supervised mode requires at least 2 class folders, each with images.")
            class_counts: List[Tuple[str, int]] = []
            total_images = 0
            for class_dir in class_dirs:
                images = self._list_images(class_dir)
                if not images:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise ValueError(f"Class folder '{class_dir.name}' contains no valid images.")
                class_counts.append((class_dir.name, len(images)))
                total_images += len(images)
            return root, total_images, class_counts
        else:
            # Unsupervised: allow images directly inside root
            images = self._list_images(root)
            if not images:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError("Unsupervised mode requires a folder with at least one image.")
            class_counts = [("all", len(images))]
            return root, len(images), class_counts

    def _list_images(self, folder: Path) -> List[Path]:
        images: List[Path] = []
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTS:
                # Basic integrity check
                try:
                    with Image.open(path) as img:
                        img.verify()
                    images.append(path)
                except Exception:
                    continue
        return images

