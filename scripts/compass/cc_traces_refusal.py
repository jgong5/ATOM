"""Observe existing oracle refusal markers in one fresh execution directory."""

import hashlib
import json
import tempfile
from pathlib import Path


class RefusalWatch:
    """Directory ownership and freshness, not an inferred writer PID."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.after_ns = self.directory.stat().st_mtime_ns
        self.preexisting = {p.name for p in self.directory.iterdir()}

    @classmethod
    def create(cls, cell, side, repeat):
        parent = Path(cell) / "refusals"
        parent.mkdir(parents=True, exist_ok=True)
        return cls(tempfile.mkdtemp(prefix=f"{side}.r{repeat}.", dir=parent))

    def description(self):
        return {"directory": str(self.directory), "after_mtime_ns": self.after_ns,
                "scope": "fresh directory assigned to this execution; marker writer PID is not recorded"}

    def read(self):
        for path in sorted(self.directory.glob("*refusal_*.json")):
            if path.name in self.preexisting or path.is_symlink() or not path.is_file():
                continue
            try:
                before = path.stat()
                if before.st_mtime_ns < self.after_ns:
                    continue
                data = path.read_bytes()
                after = path.stat()
                if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                    continue
                payload = json.loads(data)
            except (OSError, ValueError):
                continue  # Existing writers are not atomic; try a complete marker later.
            if not isinstance(payload, dict) or not isinstance(payload.get("shape"), dict):
                continue
            if (path.name.startswith("region_refusal_")
                    and payload.get("refused_by") == "region model"
                    and isinstance(payload.get("why"), str) and payload["why"]):
                kind, exception = "region_model", payload["why"]
            elif (path.name.startswith("refusal_")
                  and isinstance(payload.get("coverage"), dict)
                  and "body_graph" in payload and "head_graph" in payload):
                kind, exception = "library_coverage", None
            else:
                continue
            return {"kind": kind, "path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data), "mtime_ns": after.st_mtime_ns,
                    "exception_text": exception,
                    "exception_source": "exact marker why field" if exception is not None
                                        else "marker preserves coverage/graphs but no exception text"}
        return None
