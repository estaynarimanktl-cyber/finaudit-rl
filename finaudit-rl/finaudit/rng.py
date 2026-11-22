"""A small SHA-256 counter-mode RNG.

Python's ``random`` module is stable in practice, but this generator makes the
reproducibility contract independent of interpreter internals.  Every draw is a
function of (seed, counter); ``fork`` creates independent labelled streams.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence, TypeVar

T = TypeVar("T")
MASK64 = (1 << 64) - 1


class Rng:
    def __init__(self, seed: int | str, *, counter: int = 0, label: str = "root"):
        self.seed = str(seed)
        self.label = label
        self.counter = counter

    def _u64(self) -> int:
        payload = f"{self.seed}|{self.label}|{self.counter}".encode("utf-8")
        self.counter += 1
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def random(self) -> float:
        return self._u64() / float(1 << 64)

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("empty range")
        span = b - a + 1
        limit = (1 << 64) - ((1 << 64) % span)
        while True:
            value = self._u64()
            if value < limit:
                return a + value % span

    def choice(self, seq: Sequence[T]) -> T:
        if not seq:
            raise ValueError("choice from empty sequence")
        return seq[self.randint(0, len(seq) - 1)]

    def weighted(self, items: Sequence[T], weights: Sequence[float]) -> T:
        if len(items) != len(weights) or not items:
            raise ValueError("items/weights mismatch")
        total = sum(weights)
        point = self.random() * total
        upto = 0.0
        for item, weight in zip(items, weights):
            upto += weight
            if point < upto:
                return item
        return items[-1]

    def shuffle(self, seq: list[T]) -> None:
        for i in range(len(seq) - 1, 0, -1):
            j = self.randint(0, i)
            seq[i], seq[j] = seq[j], seq[i]

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        if k < 0 or k > len(seq):
            raise ValueError("sample larger than population")
        # Floyd's algorithm: works without materializing large ranges.
        n = len(seq)
        selected: set[int] = set()
        for j in range(n - k, n):
            t = self.randint(0, j)
            selected.add(j if t in selected else t)
        return [seq[i] for i in sorted(selected)]

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def decimal_between(self, a: str | Decimal, b: str | Decimal, places: int = 8) -> Decimal:
        lo, hi = Decimal(str(a)), Decimal(str(b))
        value = lo + (hi - lo) * Decimal(str(self.random()))
        return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)

    def hex(self, nbytes: int = 8) -> str:
        raw = bytearray()
        while len(raw) < nbytes:
            raw.extend(self._u64().to_bytes(8, "big"))
        return raw[:nbytes].hex()

    def fork(self, label: str) -> "Rng":
        digest = hashlib.sha256(f"{self.seed}|{self.label}|fork:{label}".encode("utf-8")).digest()
        return Rng(str(int.from_bytes(digest[:16], "big")), label=f"{self.label}/{label}")
