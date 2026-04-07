from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
REPORTS_DIR = STORAGE_DIR / "reports"
PLOTS_DIR = STORAGE_DIR / "plots"


def ensure_directories() -> None:
    """Create all required storage directories."""
    for directory in (STORAGE_DIR, DATASETS_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

