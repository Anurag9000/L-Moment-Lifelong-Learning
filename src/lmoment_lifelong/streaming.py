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

    def merge(self, other: "Reservoir") -> None:
        """Exact-count-weighted merge of two uniform reservoir samples.

        Conditional on each reservoir being a uniform sample without replacement
        of its complete stream, hypergeometric sample allocation produces a
        uniform bounded reservoir for the union. Appending the stored sample
        from the second reservoir as though it were the full stream is biased.
        """
        if self is other or not isinstance(other, Reservoir):
            raise ValueError("Expected a different Reservoir")
        if self.capacity != other.capacity or self.dimension != other.dimension:
            raise ValueError("Reservoir shape/capacity mismatch")
        total = self.seen + other.seen
        if total == 0:
            return
        retained = min(self.capacity, total)
        from_self = int(self._rng.hypergeometric(self.seen, other.seen, retained))
        from_other = retained - from_self
        if from_self > len(self._rows) or from_other > len(other._rows):
            raise RuntimeError("Reservoir samples inconsistent with seen counts")
        idx_a = self._rng.choice(len(self._rows), size=from_self, replace=False)
        idx_b = self._rng.choice(len(other._rows), size=from_other, replace=False)
        chosen = [self._rows[int(i)].copy() for i in idx_a] + [
            other._rows[int(i)].copy() for i in idx_b]
        self._rng.shuffle(chosen)
        self._rows = chosen
        self.seen = total
