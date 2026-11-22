"""Small helpers shared by the test-suite and verification scripts.

This module is not imported by the environment at runtime; it only makes the
public data easier to inspect in tests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import Episode
from .validate import validate_episode


def episode_fingerprint(episode: Episode) -> str:
    blob = json.dumps(
        {"open": episode.open_dicts(), "truth": episode.truth},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def open_fingerprint(episode: Episode) -> str:
    blob = json.dumps(episode.open_dicts(), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assert_valid(episode: Episode) -> None:
    errors = validate_episode(episode)
    if errors:
        raise AssertionError("; ".join(errors))


def issue_by_type(episode: Episode, issue_type: str) -> dict:
    for issue in episode.truth.get("issues", []):
        if issue["type"] == issue_type:
            return issue
    raise KeyError(issue_type)


def all_ids(episode: Episode) -> set[str]:
    return episode.all_record_ids()
