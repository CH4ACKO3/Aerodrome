"""Explicit, versioned factories and verified local assets; no configuration eval."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType


class Registry:
    def __init__(self):
        self._factories = {}

    def register(self, model_id, version, factory):
        key = (model_id, version)
        if not model_id or not version or key in self._factories or not callable(factory):
            raise ValueError(f"invalid or duplicate registry entry: {key}")
        self._factories[key] = factory

    def create(self, model_id, version, **configuration):
        key = (model_id, version)
        if key not in self._factories:
            raise ValueError(f"unregistered model/version: {key}")
        return self._factories[key](**configuration)


@dataclass(frozen=True)
class Asset:
    id: str
    version: str
    path: Path
    sha256: str
    source: str
    license: str
    metadata: object = None

    def __post_init__(self):
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("asset requires a lowercase SHA-256 digest")
        if not all((self.id, self.version, self.source, self.license)):
            raise ValueError("asset requires identity, version, source and license")

    def read_verified(self):
        """Return the exact checked bytes; parsers/loaders operate outside JIT."""
        data = self.path.read_bytes()
        if sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"asset checksum mismatch: {self.id}@{self.version}")
        return data
