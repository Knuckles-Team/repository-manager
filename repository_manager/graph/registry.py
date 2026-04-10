import json
import hashlib
from pathlib import Path
from typing import Dict, Optional


class GraphRegistry:
    """Manages the graph staleness state by checking file SHA256 hashes."""

    def __init__(self, workspace_path: str):
        self.registry_path = Path(workspace_path) / ".repo_graph" / "registry.json"
        self.state: Dict[str, str] = {}
        self._load()

    def _load(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    self.state = json.load(f)
            except json.JSONDecodeError:
                self.state = {}

    def save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def compute_hash(self, file_path: Path) -> Optional[str]:
        if not file_path.exists() or file_path.is_dir():
            return None
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except OSError:
            return None

    def is_stale(self, file_path: Path) -> bool:
        current_hash = self.compute_hash(file_path)
        if not current_hash:
            return False  # if not a file, not tracking

        file_key = str(file_path.absolute())
        if self.state.get(file_key) != current_hash:
            return True
        return False

    def mark_updated(self, file_path: Path):
        current_hash = self.compute_hash(file_path)
        if current_hash:
            self.state[str(file_path.absolute())] = current_hash
