"""Bounded streaming feature storage, used by memories and statistical monitoring."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Reservoir:
    capacity: int
    dimension: int
    seed: int = 0
    seen: int = 0
    _rows: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.capacity < 1 or self.dimension < 1:
            raise ValueError("capacity and dimension must be positive")
        self._rng = np.random.default_rng(self.seed)

    def update(self, array: np.ndarray) -> None:
        x = np.asarray(array, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("Expected finite [batch, dimension] array")
        for row in x:
            self.seen += 1
            if len(self._rows) < self.capacity:
                self._rows.append(row.copy())
            else:
                j = int(self._rng.integers(self.seen))
                if j < self.capacity:
                    self._rows[j] = row.copy()

    @property
    def samples(self) -> np.ndarray:
        if not self._rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack(self._rows)

    @property
    def bytes(self) -> int:
        return self.samples.nbytes

    def snapshot(self) -> dict:
        return {"capacity": self.capacity, "dimension": self.dimension,
                "seed": self.seed, "seen": self.seen, "rows": self.samples.copy(),
                "rng_state": self._rng.bit_generator.state}

    @classmethod
    def restore(cls, snapshot: dict) -> "Reservoir":
        obj = cls(snapshot["capacity"], snapshot["dimension"], snapshot["seed"])
        obj.seen = snapshot["seen"]
        obj._rows = [r.copy() for r in snapshot["rows"]]
        obj._rng.bit_generator.state = snapshot["rng_state"]
        return obj
