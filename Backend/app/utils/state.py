import json
from pathlib import Path
from typing import Any, Dict, Iterable


class JSONStore:
    """Tiny JSON-backed key-value store for lightweight persistence."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def _write(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, default=str))

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._write()

    def get(self, key: str, default: Any = None) -> Any:
        # Reload from file to ensure we have the latest data
        self._load()
        return self._data.get(key, default)

    def delete(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
            self._write()

    def list(self) -> Iterable[Any]:
        # Reload from file to ensure we have the latest data
        self._load()
        return self._data.values()

    def as_dict(self) -> Dict[str, Any]:
        return self._data.copy()

