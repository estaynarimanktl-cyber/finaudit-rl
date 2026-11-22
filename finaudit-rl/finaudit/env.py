"""Gymnasium-style audit environment with a narrow tool interface."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import time
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

from .generator import generate_episode
from .matcher import (
    MatchResult,
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    empty_submission,
    score_submission,
    terminal_reward,
)
from .models import Episode
from .money import format_rate

TABLES = ("ledger", "invoices", "payments", "fx", "policy")

TASK_TEXT = """You are an AI financial auditor. One episode is one fictional company and one reporting period.
The ledger is a balanced double-entry extract in the reporting (base) currency. Invoices and payments may be in foreign currencies; convert them using fx.csv and policy.json. Missing references are null.

Find the hidden errors. Supported types:
- duplicate_account: a journal entry posted twice.
- payment_allocation: a payment is missing, partial, or applied to the wrong invoice.
- fx_conversion: an entry uses the wrong FX date/rate.
- period_cutoff: a journal entry is posted in a different reporting period than its source document.

You have at most 100 tool steps including submit. Exploration rewards are zero. The terminal reward is clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1). P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN); empty truth and empty findings give F1=1, empty truth with findings gives F1=0.

Actions (JSON objects):
1. {"type":"list_records","table":"ledger|invoices|payments|fx|policy","cursor":null,"limit":50}
2. {"type":"get_record","table":"ledger|invoices|payments|fx|policy","id":"..."}
3. {"type":"get_document","doc_id":"..."}
4. {"type":"calculator","expression":"125000 * 0.0375"}
5. {"type":"submit","findings":{"episode_id":"...","findings":[...]}}

A finding has type, affected_ids, claimed_amount_minor (integer), currency, evidence_ids (max 10), explanation (max 1000 chars). Affected IDs are the records whose own values are wrong, not every record needed as evidence. Use ledger row_id, invoice/payment doc_id, fx:<date>:<src>:<dst>, or 'policy'. Business invoice_id/payment_id aliases are accepted. Evidence must contain at least one complete allowed set; every cited evidence and affected ID must exist. Matching is one-to-one; duplicates are false positives.

The environment never exposes truth.json, provenance.json, the seed, the filesystem, SQL, shell or the internet. Only requested records are returned. Unknown IDs/actions consume a step and return a limited error."""

ACTION_SCHEMA = {
    "list_records": {"table": list(TABLES), "cursor": "opaque string or null", "limit": "1..50, default 50"},
    "get_record": {"table": list(TABLES), "id": "canonical record id"},
    "get_document": {"doc_id": "invoice or payment doc_id (business id accepted)"},
    "calculator": {"expression": "arithmetic with decimal numbers, + - * / and parentheses"},
    "submit": {"findings": "{episode_id, findings:[...]}"},
}

SUBMISSION_SCHEMA = {
    "episode_id": "must equal manifest.episode_id",
    "findings": {
        "type": "duplicate_account|payment_allocation|fx_conversion|period_cutoff",
        "affected_ids": "list of canonical record ids",
        "claimed_amount_minor": "non-negative integer in base currency minor units",
        "currency": "base currency code",
        "evidence_ids": f"list of at most {MAX_EVIDENCE_IDS} existing record ids",
        "explanation": f"string of at most {MAX_EXPLANATION_CHARS} characters",
    },
    "limits": {"max_findings": MAX_FINDINGS},
}


class AuditEnv:
    """The environment used by the evaluator.

    A policy should be given ``PolicyEnv`` (below), not this object, so that it
    cannot reach ``episode.truth``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        episode: Episode | None = None,
        episode_dir: str | None = None,
        seed: int | None = None,
        config: dict | None = None,
        mode: str = "train",
        max_steps: int | None = None,
        time_limit_s: float | None = None,
    ):
        self._fixed_episode = episode
        self._episode_dir = episode_dir
        self._initial_seed = seed
        self._initial_config = dict(config or {})
        self._mode = mode
        self._max_steps_override = max_steps
        self._time_limit_override = time_limit_s
        self.episode: Episode | None = None
        self._reset_state()

    # ---- state ------------------------------------------------------------
    def _reset_state(self) -> None:
        self.step_count = 0
        self.inspection_steps = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.submitted = False
        self.submission: dict | None = None
        self.last_result: MatchResult | None = None
        self.last_observation: dict | None = None
        self.last_info: dict | None = None
        self._started_at = time.monotonic()
        self._table_cache: dict[str, list[dict]] = {}

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def max_steps(self) -> int:
        if self._max_steps_override is not None:
            return max(1, min(100, int(self._max_steps_override)))
        if self.episode is not None:
            return min(100, int(self.episode.manifest.get("limits", {}).get("max_steps", 100)))
        return 100

    @property
    def time_limit_s(self) -> float:
        if self._time_limit_override is not None:
            return float(self._time_limit_override)
        if self.episode is not None:
            return float(self.episode.manifest.get("limits", {}).get("time_limit_seconds", 60))
        return 60.0

    @property
    def expose_reward(self) -> bool:
        return self._mode != "eval"

    # ---- reset ------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        config: dict | None = None,
        *,
        episode: Episode | None = None,
        episode_dir: str | None = None,
    ) -> tuple[dict, dict]:
        self._reset_state()
        if episode is not None:
            self.episode = episode
        elif episode_dir is not None or self._episode_dir is not None:
            self.episode = Episode.load(episode_dir or self._episode_dir)
        elif self._fixed_episode is not None and seed is None and config is None:
            self.episode = self._fixed_episode
        else:
            use_seed = seed if seed is not None else self._initial_seed
            if use_seed is None:
                use_seed = 0
            use_config = {**self._initial_config, **(config or {})}
            self.episode = generate_episode(int(use_seed), use_config)
        if self.episode is None:
            raise RuntimeError("no episode available")
        self._build_table_cache()
        obs = {
            "episode_id": self.episode.episode_id,
            "step": 0,
            "budget_remaining": self.max_steps,
            "task": TASK_TEXT,
            "manifest": self.episode.manifest,
            "policy": self.episode.policy,
            "action_schema": ACTION_SCHEMA,
            "submission_schema": SUBMISSION_SCHEMA,
            "error": None,
            "result": None,
        }
        info = {
            "episode_id": self.episode.episode_id,
            "max_steps": self.max_steps,
            "mode": self._mode,
        }
        self.last_observation = obs
        self.last_info = info
        return obs, info

    def _build_table_cache(self) -> None:
        assert self.episode is not None
        self._table_cache = {
            "ledger": [r.to_dict() for r in self.episode.ledger],
            "invoices": [i.to_dict() for i in self.episode.invoices],
            "payments": [p.to_dict() for p in self.episode.payments],
            "fx": [
                {
                    "fx_id": r.fx_id,
                    "date": r.date.isoformat(),
                    "source_currency": r.source_currency,
                    "target_currency": r.target_currency,
                    "rate": format_rate(r.rate),
                }
                for r in self.episode.fx
            ],
            "policy": [self.episode.policy],
        }

    # ---- observations -----------------------------------------------------
    def _base_obs(self) -> dict:
        return {
            "episode_id": self.episode.episode_id if self.episode else None,
            "step": self.step_count,
            "budget_remaining": max(0, self.max_steps - self.step_count),
            "error": None,
            "result": None,
        }

    def _finish_observation(self, obs: dict, *, reward: float) -> tuple[dict, float, bool, bool, dict]:
        obs["done"] = self.done
        obs["terminated"] = self.terminated
        obs["truncated"] = self.truncated
        if self._mode != "eval" and self.last_result is not None:
            obs["result"] = self.last_result.to_dict()
        info: dict[str, Any] = {
            "step": self.step_count,
            "submitted": self.submitted,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        if self._mode != "eval":
            if self.last_result is not None:
                info["result"] = self.last_result.to_dict()
            info["budget_remaining"] = max(0, self.max_steps - self.step_count)
        else:
            # Evaluation mode withholds reward and diagnostics from the policy.
            info = {"step": self.step_count, "submitted": self.submitted}
        self.last_observation = obs
        self.last_info = info
        return obs, reward, self.terminated, self.truncated, info

    # ---- cursor -----------------------------------------------------------
    def _cursor_secret(self) -> str:
        return f"{self.episode.episode_id}|v1"

    def _encode_cursor(self, table: str, offset: int) -> str:
        payload = f"{table}:{offset}"
        sig = hashlib.sha256(f"{self._cursor_secret()}|{payload}".encode("utf-8")).hexdigest()[:12]
        return base64.urlsafe_b64encode(f"{payload}:{sig}".encode("utf-8")).decode("ascii")

    def _decode_cursor(self, cursor: str, table: str) -> int:
        try:
            raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
            t, offset_s, sig = raw.rsplit(":", 2)
            offset = int(offset_s)
        except Exception as exc:
            raise ValueError("invalid cursor") from exc
        payload = f"{t}:{offset}"
        expected = hashlib.sha256(f"{self._cursor_secret()}|{payload}".encode("utf-8")).hexdigest()[:12]
        if t != table or sig != expected or offset < 0:
            raise ValueError("invalid or stale cursor")
        return offset

    # ---- actions ----------------------------------------------------------
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if self.episode is None:
            raise RuntimeError("call reset() first")
        if self.done:
            obs = self._base_obs()
            obs["error"] = "episode already finished"
            return self._finish_observation(obs, reward=0.0)

        # Wall-clock limit. Deterministic replay uses the step limit; the time
        # limit is a safety valve for official runs.
        if time.monotonic() - self._started_at > self.time_limit_s:
            self.step_count += 1
            self.inspection_steps = self.step_count
            self.truncated = True
            self.done = True
            self.submitted = False
            self.submission = empty_submission(self.episode, inspection_steps=self.inspection_steps)
            self.last_result = score_submission(
                self.episode, self.submission, inspection_steps=self.inspection_steps
            )
            obs = self._base_obs()
            obs["error"] = "time limit reached"
            reward = self.last_result.reward if self.expose_reward else 0.0
            return self._finish_observation(obs, reward=reward)

        self.step_count += 1
        try:
            obs, reward = self._dispatch(action)
        except Exception as exc:
            obs = self._base_obs()
            obs["error"] = f"invalid action: {type(exc).__name__}"
            reward = 0.0

        # Submit terminates inside _dispatch; otherwise enforce step budget.
        if not self.done and self.step_count >= self.max_steps:
            self.inspection_steps = self.step_count
            self.truncated = True
            self.done = True
            self.submitted = False
            self.submission = empty_submission(self.episode, inspection_steps=self.inspection_steps)
            self.last_result = score_submission(
                self.episode, self.submission, inspection_steps=self.inspection_steps
            )
            obs["error"] = "step limit reached"
            reward = self.last_result.reward if self.expose_reward else 0.0
        return self._finish_observation(obs, reward=reward)

    def _dispatch(self, action: Any) -> tuple[dict, float]:
        if not isinstance(action, dict):
            raise ValueError("action must be an object")
        atype = action.get("type")
        if not isinstance(atype, str):
            raise ValueError("action.type must be a string")
        if atype == "list_records":
            return self._list_records(action)
        if atype == "get_record":
            return self._get_record(action)
        if atype == "get_document":
            return self._get_document(action)
        if atype == "calculator":
            return self._calculator(action)
        if atype == "submit":
            return self._submit(action)
        raise ValueError("unknown action type")

    def _list_records(self, action: dict) -> tuple[dict, float]:
        table = action.get("table")
        if table not in TABLES:
            raise ValueError("unknown table")
        limit = action.get("limit", 50)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        limit = min(50, limit)
        cursor = action.get("cursor")
        if cursor in (None, ""):
            offset = 0
        elif isinstance(cursor, str):
            offset = self._decode_cursor(cursor, table)
        else:
            raise ValueError("cursor must be a string or null")
        rows = self._table_cache[table]
        page = rows[offset : offset + limit]
        next_cursor = self._encode_cursor(table, offset + limit) if offset + limit < len(rows) else None
        obs = self._base_obs()
        obs.update(
            {
                "action": "list_records",
                "table": table,
                "records": page,
                "returned": len(page),
                "total": len(rows),
                "next_cursor": next_cursor,
                "limit": limit,
            }
        )
        return obs, 0.0

    def _resolve_table_id(self, table: str, raw: str) -> dict | None:
        assert self.episode is not None
        if table == "ledger":
            for row in self.episode.ledger:
                if row.row_id == raw:
                    return row.to_dict()
            return None
        if table == "invoices":
            for inv in self.episode.invoices:
                if raw in (inv.doc_id, inv.invoice_id):
                    return inv.to_dict()
            return None
        if table == "payments":
            for pay in self.episode.payments:
                if raw in (pay.doc_id, pay.payment_id):
                    return pay.to_dict()
            return None
        if table == "fx":
            for rate in self.episode.fx:
                if raw == rate.fx_id:
                    return {
                        "fx_id": rate.fx_id,
                        "date": rate.date.isoformat(),
                        "source_currency": rate.source_currency,
                        "target_currency": rate.target_currency,
                        "rate": format_rate(rate.rate),
                    }
            return None
        if table == "policy":
            if raw in ("policy", "POLICY"):
                return self.episode.policy
            return None
        return None

    def _get_record(self, action: dict) -> tuple[dict, float]:
        table = action.get("table")
        if table not in TABLES:
            raise ValueError("unknown table")
        raw = action.get("id")
        if not isinstance(raw, str) or not raw:
            raise ValueError("id must be a non-empty string")
        record = self._resolve_table_id(table, raw)
        if record is None:
            raise KeyError("unknown record id")
        obs = self._base_obs()
        obs.update({"action": "get_record", "table": table, "record": record})
        return obs, 0.0

    def _get_document(self, action: dict) -> tuple[dict, float]:
        raw = action.get("doc_id")
        if not isinstance(raw, str) or not raw:
            raise ValueError("doc_id must be a non-empty string")
        assert self.episode is not None
        for inv in self.episode.invoices:
            if raw in (inv.doc_id, inv.invoice_id):
                obs = self._base_obs()
                obs.update({"action": "get_document", "doc_type": "invoice", "document": inv.to_dict()})
                return obs, 0.0
        for pay in self.episode.payments:
            if raw in (pay.doc_id, pay.payment_id):
                obs = self._base_obs()
                obs.update({"action": "get_document", "doc_type": "payment", "document": pay.to_dict()})
                return obs, 0.0
        raise KeyError("unknown document id")

    def _calculator(self, action: dict) -> tuple[dict, float]:
        expression = action.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a non-empty string")
        if len(expression) > 500:
            raise ValueError("expression too long")
        value = _safe_decimal_eval(expression)
        obs = self._base_obs()
        obs.update({"action": "calculator", "expression": expression, "result_value": _decimal_text(value)})
        return obs, 0.0

    def _submit(self, action: dict) -> tuple[dict, float]:
        payload = action.get("findings")
        if payload is None:
            payload = action.get("payload")
        if payload is None:
            raise ValueError("submit requires findings")
        self.submitted = True
        self.inspection_steps = max(0, self.step_count - 1)
        self.submission = payload if isinstance(payload, dict) else {"invalid": True}
        self.last_result = score_submission(
            self.episode,
            self.submission,
            inspection_steps=self.inspection_steps,
        )
        self.done = True
        self.terminated = True
        self.truncated = False
        obs = self._base_obs()
        obs.update({"action": "submit", "submitted": True})
        if not self.last_result.valid:
            obs["error"] = self.last_result.error
        reward = self.last_result.reward if self.expose_reward else 0.0
        return obs, reward

    # ---- evaluator --------------------------------------------------------
    def grade(self) -> MatchResult | None:
        """Evaluator-only result.  Never exposed through PolicyEnv."""
        return self.last_result

    def evaluate_empty(self) -> MatchResult:
        assert self.episode is not None
        return score_submission(
            self.episode,
            empty_submission(self.episode, inspection_steps=self.step_count),
            inspection_steps=self.step_count,
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# safe calculator


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


def _safe_decimal_eval(expression: str) -> Decimal:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid arithmetic expression") from exc

    def ev(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only decimal numbers are allowed")
            text = str(node.value)
            if len(text) > 80:
                raise ValueError("number too long")
            return Decimal(text)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = ev(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left = ev(node.left)
            right = ev(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
        raise ValueError("only + - * / and parentheses are allowed")

    with localcontext() as ctx:
        ctx.prec = 28
        try:
            result = ev(tree)
        except (DivisionByZero, InvalidOperation) as exc:
            raise ValueError("invalid arithmetic") from exc
    if not result.is_finite():
        raise ValueError("non-finite result")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if len(text) > 1000:
        raise ValueError("result too large")
    return text


# ---------------------------------------------------------------------------
# policy-facing wrapper


class PolicyEnv:
    """Narrow interface: reset/step only, no truth, seed or private paths."""

    def __init__(self, env: AuditEnv):
        object.__setattr__(self, "_env", env)

    def reset(self, seed: int | None = None, config: dict | None = None):
        return self._env.reset(seed, config)

    def step(self, action: Any):
        return self._env.step(action)

    def __getattr__(self, name: str):
        raise AttributeError(name)
