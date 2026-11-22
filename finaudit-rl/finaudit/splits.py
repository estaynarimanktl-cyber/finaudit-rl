"""Named train/dev episode specifications.

The specifications are deliberately small and explicit so the benchmark is
reproducible without a config file.  Each entry is one seed and one generator
configuration.  ``n_issues`` is the number of injected issues; interacting
episodes use a three-issue chain (allocation + FX + cutoff) or a two-issue
chain (duplicate + FX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inject import ALL_TYPES, TYPE_ALLOCATION, TYPE_DUPLICATE, TYPE_FX, TYPE_PERIOD


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    split: str
    seed: int
    config: dict

    def public(self) -> dict:
        return {
            "name": self.name,
            "split": self.split,
            "seed": self.seed,
            "config": dict(self.config),
        }


def _cfg(
    n_issues: int,
    *,
    interacting: bool = False,
    error_types: list[str] | None = None,
    n_invoices: int = 55,
    n_payments: int = 45,
    difficulty: str = "medium",
    clean: bool = False,
) -> dict:
    return {
        "difficulty": difficulty,
        "n_issues": int(n_issues),
        "error_types": list(error_types or ALL_TYPES),
        "interacting": bool(interacting),
        "clean": bool(clean),
        "n_invoices": int(n_invoices),
        "n_payments": int(n_payments),
        "margin_invoices": 2,
        "max_steps": 100,
        "time_limit_seconds": 60,
    }


TRAIN_SPECS: list[EpisodeSpec] = [
    EpisodeSpec("train-01", "train", 1101, _cfg(0, clean=True, n_invoices=48, n_payments=38, difficulty="easy")),
    EpisodeSpec("train-02", "train", 1102, _cfg(1, n_invoices=52, n_payments=40, difficulty="easy")),
    EpisodeSpec("train-03", "train", 1103, _cfg(2, n_invoices=55, n_payments=42)),
    EpisodeSpec("train-04", "train", 1104, _cfg(3, n_invoices=58, n_payments=45)),
    EpisodeSpec("train-05", "train", 1105, _cfg(4, n_invoices=60, n_payments=48)),
    EpisodeSpec("train-06", "train", 1106, _cfg(5, n_invoices=62, n_payments=50)),
    EpisodeSpec("train-07", "train", 1107, _cfg(6, n_invoices=64, n_payments=52, difficulty="hard")),
    EpisodeSpec("train-08", "train", 1108, _cfg(8, n_invoices=66, n_payments=54, difficulty="hard")),
    EpisodeSpec(
        "train-09",
        "train",
        1109,
        _cfg(
            3,
            interacting=True,
            error_types=[TYPE_ALLOCATION, TYPE_FX, TYPE_PERIOD, TYPE_DUPLICATE],
            n_invoices=60,
            n_payments=48,
            difficulty="hard",
        ),
    ),
    EpisodeSpec(
        "train-10",
        "train",
        1110,
        _cfg(
            2,
            interacting=True,
            error_types=[TYPE_DUPLICATE, TYPE_FX],
            n_invoices=58,
            n_payments=46,
            difficulty="hard",
        ),
    ),
]

DEV_SPECS: list[EpisodeSpec] = [
    EpisodeSpec("dev-01", "dev", 2201, _cfg(0, clean=True, n_invoices=50, n_payments=40, difficulty="easy")),
    EpisodeSpec("dev-02", "dev", 2202, _cfg(1, n_invoices=53, n_payments=41, difficulty="easy")),
    EpisodeSpec("dev-03", "dev", 2203, _cfg(2, n_invoices=56, n_payments=44)),
    EpisodeSpec("dev-04", "dev", 2204, _cfg(3, n_invoices=59, n_payments=46)),
    EpisodeSpec("dev-05", "dev", 2205, _cfg(4, n_invoices=61, n_payments=49)),
    EpisodeSpec("dev-06", "dev", 2206, _cfg(5, n_invoices=63, n_payments=51)),
    EpisodeSpec("dev-07", "dev", 2207, _cfg(6, n_invoices=65, n_payments=53, difficulty="hard")),
    EpisodeSpec("dev-08", "dev", 2208, _cfg(8, n_invoices=68, n_payments=56, difficulty="hard")),
    EpisodeSpec(
        "dev-09",
        "dev",
        2209,
        _cfg(
            3,
            interacting=True,
            error_types=[TYPE_ALLOCATION, TYPE_FX, TYPE_PERIOD, TYPE_DUPLICATE],
            n_invoices=62,
            n_payments=50,
            difficulty="hard",
        ),
    ),
    EpisodeSpec(
        "dev-10",
        "dev",
        2210,
        _cfg(
            3,
            interacting=True,
            error_types=[TYPE_DUPLICATE, TYPE_FX],
            n_invoices=60,
            n_payments=48,
            difficulty="hard",
        ),
    ),
]


ALL_SPECS = TRAIN_SPECS + DEV_SPECS


def specs_for(split: str) -> list[EpisodeSpec]:
    if split == "train":
        return list(TRAIN_SPECS)
    if split == "dev":
        return list(DEV_SPECS)
    if split in ("all", ""):
        return list(ALL_SPECS)
    raise ValueError(f"unknown split {split!r}")
