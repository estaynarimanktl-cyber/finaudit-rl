"""Trajectory recording and policy evaluation helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .env import AuditEnv
from .policies import Policy, make_policy


def _obs_summary(obs: dict | None) -> dict:
    if not obs:
        return {}
    out: dict[str, Any] = {
        "step": obs.get("step"),
        "budget_remaining": obs.get("budget_remaining"),
        "error": obs.get("error"),
    }
    action = obs.get("action")
    if action:
        out["action"] = action
    if action == "list_records":
        out.update(
            {
                "table": obs.get("table"),
                "returned": obs.get("returned"),
                "total": obs.get("total"),
                "has_next": obs.get("next_cursor") is not None,
            }
        )
    elif action == "get_record":
        out["table"] = obs.get("table")
        record = obs.get("record") or {}
        out["id"] = record.get("row_id") or record.get("doc_id") or record.get("fx_id") or record.get("policy_id")
    elif action == "get_document":
        out["doc_type"] = obs.get("doc_type")
        document = obs.get("document") or {}
        out["id"] = document.get("doc_id")
    elif action == "calculator":
        out["expression"] = obs.get("expression")
        out["result_value"] = obs.get("result_value")
    elif action == "submit":
        out["submitted"] = obs.get("submitted")
    return out


def _action_summary(action: Any) -> dict:
    if not isinstance(action, dict):
        return {"type": type(action).__name__}
    out: dict[str, Any] = {"type": action.get("type")}
    for key in ("table", "id", "doc_id", "expression", "cursor", "limit"):
        if key in action:
            out[key] = action[key]
    if action.get("type") == "submit":
        payload = action.get("findings") or action.get("payload") or {}
        if isinstance(payload, dict):
            out["n_findings"] = len(payload.get("findings") or [])
    return out


def _digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_policy(
    env: AuditEnv,
    policy: Policy,
    *,
    include_observations: bool = False,
    max_steps: int | None = None,
) -> dict:
    """Run ``policy`` in ``env`` and return a compact trajectory record."""
    obs, info = env.reset()
    policy.reset(obs, info)
    steps: list[dict] = []
    observed_reward = 0.0
    if max_steps is not None:
        limit = max_steps
    elif hasattr(env, "max_steps"):
        limit = int(env.max_steps) + 1
    else:
        limit = 101
    terminated = truncated = False
    while not (terminated or truncated):
        if len(steps) >= limit:
            break
        action = policy.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        observed_reward += float(reward)
        step_record: dict[str, Any] = {
            "i": len(steps) + 1,
            "action": _action_summary(action),
            "reward": float(reward),
            "done": bool(terminated or truncated),
            "obs": _obs_summary(next_obs),
        }
        if include_observations:
            step_record["observation"] = next_obs
        else:
            step_record["obs_digest"] = _digest(next_obs)
        steps.append(step_record)
        obs = next_obs
    grade_fn = getattr(env, "grade", None)
    grade = grade_fn() if callable(grade_fn) else None
    graded_reward = float(grade.reward) if grade is not None else None
    result = {
        "policy": policy.name,
        "episode_id": obs.get("episode_id") if obs else None,
        "observed_reward": observed_reward,
        "graded_reward": graded_reward,
        "steps": len(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "grade": grade.to_dict() if grade is not None else None,
        "trajectory": steps,
    }
    return result


def run_named_policy(
    episode: Any,
    name: str,
    *,
    seed: int = 0,
    mode: str = "eval",
    include_observations: bool = False,
    max_steps: int | None = None,
) -> dict:
    env = AuditEnv(episode=episode, mode=mode)
    policy = make_policy(name, seed=seed, episode=episode)
    return run_policy(
        env,
        policy,
        include_observations=include_observations,
        max_steps=max_steps,
    )


def trajectory_jsonl(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records) + "\n"
