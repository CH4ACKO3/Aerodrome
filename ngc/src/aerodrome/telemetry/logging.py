"""Select fields inside JIT; transfer/write one completed chunk on the host."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import re
import jax
import numpy as np


@dataclass(frozen=True)
class Channel:
    name: str
    select: Callable
    unit: str = "1"
    description: str = ""


class Recorder:
    def __init__(self, channels):
        self.channels = tuple(channels)
        names = [c.name for c in self.channels]
        if not names or len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in names):
            raise ValueError("channels require unique safe names")

    def __call__(self, *args):
        """Pure projection; usable as BatchedWorld's record callback."""
        return {c.name: c.select(*args) for c in self.channels}

    def schema(self):
        return {c.name: dict(unit=c.unit, description=c.description) for c in self.channels}


class ChunkWriter:
    """Single-writer append-only NPZ chunks with per-chunk JSON schema.

    No device callbacks, pickle, background threads, or unbounded in-memory history.
    The JSON sidecar is the commit marker; consumers ignore NPZ without a sidecar.
    """
    def __init__(self, directory, recorder, *, axes=("time", "world"), metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.schema = recorder.schema()
        self.axes, self.metadata = tuple(axes), dict(metadata or {})
        self._layout = None

    def write(self, index, values):
        if type(index) is not int or index < 0:
            raise ValueError("chunk index must be a nonnegative integer")
        if set(values) != set(self.schema):
            raise ValueError("logged values must match channels")
        arrays = {k: np.asarray(v) for k, v in jax.device_get(values).items()}
        prefix = None
        for array in arrays.values():
            if array.dtype.kind not in "biufc" or array.ndim < len(self.axes):
                raise ValueError("channels must be numerical arrays with all declared axes")
            shape = array.shape[:len(self.axes)]
            if prefix is not None and shape != prefix:
                raise ValueError("channel leading axes must match")
            prefix = shape
        layout = {k: (str(v.dtype), v.shape[1:]) for k, v in arrays.items()}
        if self._layout is not None and layout != self._layout:
            raise ValueError("only the leading chunk length may change")
        target = self.directory / f"chunk-{index:06d}.npz"
        sidecar = target.with_suffix(".json")
        manifest = dict(version=1, axes=self.axes, metadata=self.metadata,
                        channels={k: dict(self.schema[k], shape=v.shape, dtype=str(v.dtype)) for k, v in arrays.items()})
        serialized = json.dumps(manifest, indent=2, allow_nan=False)
        if sidecar.exists():
            raise FileExistsError(sidecar)
        # Exclusive creation also prevents accidentally replacing existing chunks.
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with sidecar.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        self._layout = layout
        return target
