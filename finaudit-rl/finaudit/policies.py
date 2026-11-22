"""Baseline policies for the FinAudit environment.

The four required baselines are:

``random``
    Lists the tables, then submits a few arbitrary but structurally valid
    findings.  It is a lower bound, not a serious auditor.
``reconciliation``
    A deterministic, rule-based auditor that uses only the public records and
    the published policy.  It reconciles invoices/payments to the ledger and
    flags duplicate postings, allocation breaks, FX differences and period
    cutoffs.  It is intentionally not an oracle.
``mark_all``
    Flags every plausible-looking pattern (all repeated rows, all foreign
    documents, all payments).  It demonstrates why precision matters.
``oracle``
    Evaluator-only upper bound that converts hidden truth into a submission.

A fifth ``silence`` policy submits no findings; it is used to demonstrate the
clean-book and erroneous-book reward boundaries.

Policies are deterministic given their seed and the environment's public
observations.  They never access ``episode.truth`` except for the explicitly
marked evaluator-only oracle.
"""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

from .inject import ALL_TYPES
from .matcher import MAX_EVIDENCE_IDS, MAX_FINDINGS, oracle_submission
from .money import convert_minor, parse_iso_datetime

ACCOUNT_CASH = "1000"
ACCOUNT_AR = "1100"
ACCOUNT_AP = "2000"
ACCOUNT_SUSPENSE = "9999"


class Policy:
    """Minimal policy protocol: ``reset`` then repeated ``act`` calls."""

    name = "policy"
    evaluator_only = False

    def reset(self, obs: dict, info: dict) -> None:
        pass

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# helpers


def _finding(
    ftype: str,
    affected_ids: Iterable[str],
    amount_minor: int,
    currency: str,
    evidence_ids: Iterable[str],
    explanation: str,
) -> dict:
    affected = list(dict.fromkeys(str(x) for x in affected_ids))
    evidence = list(dict.fromkeys(str(x) for x in evidence_ids))
    if len(evidence) > MAX_EVIDENCE_IDS:
        evidence = evidence[:MAX_EVIDENCE_IDS]
    return {
        "type": ftype,
        "affected_ids": affected,
        "claimed_amount_minor": int(max(0, amount_minor)),
        "currency": currency,
        "evidence_ids": evidence,
        "explanation": explanation[:1000],
    }


def _period_of(value: str, start: datetime, end: datetime) -> int:
    dt = parse_iso_datetime(value)
    if dt < start:
        return -1
    if dt > end:
        return 1
    return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# collection base


class CollectingPolicy(Policy):
    """Fetch all public tables page by page, then decide what to submit."""

    TABLES = ("policy", "invoices", "payments", "ledger", "fx")

    def __init__(self) -> None:
        self._hard_reset()

    def _hard_reset(self) -> None:
        self._tables = list(self.TABLES)
        self._current: str | None = None
        self._cursor: str | None = None
        self._records: dict[str, list[dict]] = {t: [] for t in self.TABLES}
        self._manifest: dict = {}
        self._policy: dict = {}
        self._submitted = False
        self._pages = 0

    def reset(self, obs: dict, info: dict) -> None:
        self._hard_reset()
        self._manifest = dict(obs.get("manifest") or {})
        self._policy = dict(obs.get("policy") or {})

    def act(self, obs: dict) -> dict:
        self._ingest(obs)
        if self._submitted:
            # The environment should have terminated already; this is a safe
            # fallback that still produces a valid action.
            return {"type": "submit", "findings": {"episode_id": self._episode_id(), "findings": []}}
        if self._current is None:
            if self._tables:
                self._current = self._tables.pop(0)
                self._cursor = None
            else:
                self._submitted = True
                return {"type": "submit", "findings": self._decide()}
        return {
            "type": "list_records",
            "table": self._current,
            "cursor": self._cursor,
            "limit": 50,
        }

    def _ingest(self, obs: dict) -> None:
        if obs.get("action") != "list_records":
            return
        table = obs.get("table")
        if table != self._current:
            return
        if obs.get("error"):
            # Do not loop on a bad page; skip the rest of this table.
            self._current = None
            self._cursor = None
            return
        self._records.setdefault(str(table), []).extend(obs.get("records") or [])
        self._pages += 1
        self._cursor = obs.get("next_cursor")
        if self._cursor is None:
            self._current = None

    def _episode_id(self) -> str:
        return str(self._manifest.get("episode_id", ""))

    def _decide(self) -> dict:
        raise NotImplementedError

    # ---- indexes ---------------------------------------------------------
    def _indexes(self) -> dict:
        invoices = self._records.get("invoices", [])
        payments = self._records.get("payments", [])
        ledger = self._records.get("ledger", [])
        fx = self._records.get("fx", [])
        inv_by_doc = {i["doc_id"]: i for i in invoices}
        inv_by_business = {i["invoice_id"]: i for i in invoices}
        pay_by_doc = {p["doc_id"]: p for p in payments}
        pay_by_business = {p["payment_id"]: p for p in payments}
        rates: dict[tuple[str, str], dict[str, Decimal]] = defaultdict(dict)
        for row in fx:
            rates[(row["source_currency"], row["target_currency"])][row["date"]] = Decimal(str(row["rate"]))
        return {
            "invoices": invoices,
            "payments": payments,
            "ledger": ledger,
            "fx": fx,
            "inv_by_doc": inv_by_doc,
            "inv_by_business": inv_by_business,
            "pay_by_doc": pay_by_doc,
            "pay_by_business": pay_by_business,
            "rates": rates,
        }

    def _rate(self, source: str, target: str, day: date, rates: dict) -> Decimal:
        if source == target:
            return Decimal(1)
        table = rates.get((source, target))
        if table is None:
            table = rates.get((target, source))
            if table is None:
                raise KeyError(f"no rate {source}->{target}")
            # Invert the direct table.
            dates = [d for d in table if d <= day.isoformat()]
            chosen = max(dates) if dates else min(table)
            return Decimal(1) / table[chosen]
        dates = [d for d in table if d <= day.isoformat()]
        chosen = max(dates) if dates else min(table)
        return table[chosen]

    def _tolerance(self, expected: int) -> int:
        rounding = self._manifest.get("rounding", {})
        absolute = _safe_int(rounding.get("amount_tolerance_minor", 2), 2)
        bps = float(rounding.get("amount_tolerance_bps", 5))
        return max(absolute, int(round(expected * bps / 10000.0)))

    def _base_currency(self) -> str:
        return str(self._manifest.get("base_currency", "USD"))

    def _bounds(self) -> tuple[datetime, datetime]:
        return (
            parse_iso_datetime(self._manifest["period_start"]),
            parse_iso_datetime(self._manifest["period_end"]),
        )

    def _doc_for_row(self, row: dict, idx: dict) -> tuple[str | None, dict | None]:
        if row.get("payment_id"):
            pay = idx["pay_by_business"].get(row["payment_id"])
            if pay is not None:
                return pay["doc_id"], pay
        if row.get("invoice_id"):
            inv = idx["inv_by_business"].get(row["invoice_id"])
            if inv is not None:
                return inv["doc_id"], inv
        return None, None


# ---------------------------------------------------------------------------
# random


class RandomPolicy(CollectingPolicy):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = random.Random(int(seed))

    def _hard_reset(self) -> None:
        super()._hard_reset()
        self._rng = getattr(self, "_rng", random.Random(0))

    def _decide(self) -> dict:
        idx = self._indexes()
        base = self._base_currency()
        ids = [r["row_id"] for r in idx["ledger"]]
        ids += [i["doc_id"] for i in idx["invoices"]]
        ids += [p["doc_id"] for p in idx["payments"]]
        ids += [r["fx_id"] for r in idx["fx"]]
        ids.append("policy")
        if not ids:
            return {"episode_id": self._episode_id(), "findings": []}
        findings = []
        for _ in range(self._rng.randint(0, 3)):
            affected = self._rng.sample(ids, k=min(len(ids), self._rng.randint(1, 3)))
            evidence = self._rng.sample(ids, k=min(len(ids), self._rng.randint(1, 5)))
            findings.append(
                _finding(
                    self._rng.choice(ALL_TYPES),
                    affected,
                    self._rng.randint(1, 5_000_000),
                    base,
                    evidence,
                    "Random baseline finding.",
                )
            )
        return {"episode_id": self._episode_id(), "findings": findings[:MAX_FINDINGS]}


# ---------------------------------------------------------------------------
# silence


class SilencePolicy(Policy):
    name = "silence"

    def __init__(self) -> None:
        self._episode_id = ""

    def reset(self, obs: dict, info: dict) -> None:
        self._episode_id = str(obs.get("episode_id", ""))

    def act(self, obs: dict) -> dict:
        return {"type": "submit", "findings": {"episode_id": self._episode_id, "findings": []}}


# ---------------------------------------------------------------------------
# reconciliation


class ReconciliationPolicy(CollectingPolicy):
    """Deterministic rule-based auditor using only public records."""

    name = "reconciliation"

    def _decide(self) -> dict:
        idx = self._indexes()
        findings: list[dict] = []
        findings.extend(self._duplicates(idx))
        findings.extend(self._period_cutoffs(idx))
        findings.extend(self._allocation_breaks(idx))
        findings.extend(self._fx_differences(idx))
        # Stable order makes the policy reproducible.
        findings.sort(
            key=lambda f: (
                f["type"],
                tuple(f["affected_ids"]),
                int(f["claimed_amount_minor"]),
            )
        )
        return {"episode_id": self._episode_id(), "findings": findings[:MAX_FINDINGS]}

    # ---- duplicates ------------------------------------------------------
    def _duplicates(self, idx: dict) -> list[dict]:
        base = self._base_currency()
        # A duplicate posting is an exact copy of an entry, typically with a
        # timestamp a second or two later.  Match rows by their economic
        # signature, then require a very small timestamp gap so ordinary
        # recurring amounts are not treated as duplicates.
        by_signature: dict[tuple, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            if not (row.get("invoice_id") or row.get("payment_id")):
                continue
            key = (
                row["account"],
                _safe_int(row["debit_minor"]),
                _safe_int(row["credit_minor"]),
                row["entity_id"],
                row.get("invoice_id"),
                row.get("payment_id"),
            )
            by_signature[key].append(row)

        twin_ids: set[str] = set()
        for rows in by_signature.values():
            if len(rows) < 2:
                continue
            rows = sorted(rows, key=lambda r: r["timestamp"])
            for i in range(len(rows) - 1):
                try:
                    t0 = parse_iso_datetime(rows[i]["timestamp"])
                    t1 = parse_iso_datetime(rows[i + 1]["timestamp"])
                except ValueError:
                    continue
                if abs((t1 - t0).total_seconds()) <= 5:
                    twin_ids.add(rows[i]["row_id"])
                    twin_ids.add(rows[i + 1]["row_id"])

        # Group twin rows back into their economic journal entry (one source
        # document), regardless of the one-second posting offset.
        by_entry: dict[tuple, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            if row["row_id"] not in twin_ids:
                continue
            if row.get("payment_id"):
                ref = ("pay", row["payment_id"])
            elif row.get("invoice_id"):
                ref = ("inv", row["invoice_id"])
            else:
                continue
            by_entry[ref].append(row)

        out = []
        for ref, rows in sorted(by_entry.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            # A duplicated entry has at least one identical row pair.  Require
            # a pair for every affected row so a coincidental same-account
            # split is not treated as a duplicate.
            row_keys: dict[tuple, int] = defaultdict(int)
            for row in rows:
                row_keys[
                    (
                        row["account"],
                        _safe_int(row["debit_minor"]),
                        _safe_int(row["credit_minor"]),
                        row["entity_id"],
                    )
                ] += 1
            if not row_keys or any(count < 2 for count in row_keys.values()):
                continue
            if len(rows) % 2 != 0:
                continue
            doc_id, _ = self._doc_for_row(rows[0], idx)
            if doc_id is None:
                continue
            debit_total = sum(_safe_int(r["debit_minor"]) for r in rows)
            amount = debit_total // 2
            if amount <= 0:
                continue
            affected = [r["row_id"] for r in rows]
            evidence = affected + [doc_id]
            out.append(
                _finding(
                    "duplicate_account",
                    affected,
                    amount,
                    base,
                    evidence,
                    "Identical journal lines for one source document were posted twice; the "
                    "second balanced entry is a duplicate.",
                )
            )
        return out

    # ---- period cutoff ---------------------------------------------------
    def _period_cutoffs(self, idx: dict) -> list[dict]:
        base = self._base_currency()
        start, end = self._bounds()
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            if row.get("payment_id"):
                key = ("pay", row["payment_id"], row["timestamp"])
            elif row.get("invoice_id"):
                key = ("inv", row["invoice_id"], row["timestamp"])
            else:
                continue
            groups[key].append(row)

        out = []
        for (kind, business_id, timestamp), rows in sorted(groups.items()):
            if kind == "pay":
                doc = idx["pay_by_business"].get(business_id)
            else:
                doc = idx["inv_by_business"].get(business_id)
            if doc is None:
                continue
            doc_time = doc.get("timestamp") if kind == "pay" else doc.get("issued_at")
            if not doc_time:
                continue
            doc_period = _period_of(doc_time, start, end)
            row_period = _period_of(timestamp, start, end)
            if doc_period == row_period:
                continue
            amount = sum(_safe_int(r["debit_minor"]) for r in rows)
            if amount <= 0:
                continue
            affected = [doc["doc_id"]] + [r["row_id"] for r in rows]
            evidence = affected + ["policy"]
            out.append(
                _finding(
                    "period_cutoff",
                    affected,
                    amount,
                    base,
                    evidence,
                    "The journal entry is posted in a different reporting period than its "
                    "source document, contrary to the recognition rule.",
                )
            )
        return out

    # ---- payment allocation ---------------------------------------------
    def _allocation_breaks(self, idx: dict) -> list[dict]:
        base = self._base_currency()
        rows_by_payment: dict[str, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            if row.get("payment_id"):
                rows_by_payment[row["payment_id"]].append(row)

        out = []
        for business_id, rows in sorted(rows_by_payment.items()):
            pay = idx["pay_by_business"].get(business_id)
            if pay is None or pay.get("kind") == "refund":
                continue
            refs = list(pay.get("invoice_refs") or [])
            if not refs:
                continue
            intended = idx["inv_by_business"].get(refs[0])
            if intended is None:
                continue
            cash_rows = [r for r in rows if r["account"] == ACCOUNT_CASH]
            suspense = [r for r in rows if r["account"] == ACCOUNT_SUSPENSE]
            source_rows = [r for r in rows if r["account"] in (ACCOUNT_AR, ACCOUNT_AP)]
            cash_ids = [r["row_id"] for r in cash_rows]

            if suspense:
                amount = sum(_safe_int(r["debit_minor"]) + _safe_int(r["credit_minor"]) for r in suspense)
                if amount <= 0:
                    continue
                affected = [pay["doc_id"], intended["doc_id"]]
                affected += [r["row_id"] for r in suspense]
                affected += [r["row_id"] for r in source_rows]
                evidence = [pay["doc_id"], intended["doc_id"]]
                evidence += [r["row_id"] for r in source_rows]
                evidence += [r["row_id"] for r in suspense]
                evidence += cash_ids
                evidence.append("policy")
                out.append(
                    _finding(
                        "payment_allocation",
                        affected,
                        amount,
                        base,
                        evidence,
                        "Part or all of the payment is parked in suspense instead of being "
                        "allocated to the referenced invoice.",
                    )
                )
                continue

            # Wrong invoice: the ledger row points at an invoice the payment
            # does not reference.
            for src in source_rows:
                wrong_business = src.get("invoice_id")
                if not wrong_business or wrong_business in refs:
                    continue
                wrong = idx["inv_by_business"].get(wrong_business)
                if wrong is None:
                    continue
                amount = _safe_int(src["debit_minor"]) + _safe_int(src["credit_minor"])
                if amount <= 0:
                    continue
                affected = [pay["doc_id"], intended["doc_id"], wrong["doc_id"], src["row_id"]]
                evidence = affected + cash_ids + ["policy"]
                out.append(
                    _finding(
                        "payment_allocation",
                        affected,
                        amount,
                        base,
                        evidence,
                        "The payment references one invoice but the ledger allocation was "
                        "posted to a different invoice.",
                    )
                )
        return out

    # ---- FX ---------------------------------------------------------------
    def _fx_differences(self, idx: dict) -> list[dict]:
        base = self._base_currency()
        rates = idx["rates"]
        out: list[dict] = []

        # Invoice accruals.
        invoice_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            # Accrual rows carry the invoice reference and no payment
            # reference.  Settlement rows also carry the invoice reference,
            # but their cash/AP leg is not an accrual translation.
            if row.get("invoice_id") and not row.get("payment_id"):
                invoice_rows[(row["invoice_id"], row["timestamp"])].append(row)
        for (business_id, timestamp), rows in sorted(invoice_rows.items()):
            inv = idx["inv_by_business"].get(business_id)
            if inv is None or inv.get("currency") == base:
                continue
            debits = [r for r in rows if _safe_int(r["debit_minor"]) > 0]
            if not debits:
                continue
            try:
                day = parse_iso_datetime(inv["issued_at"]).date()
                rate = self._rate(inv["currency"], base, day, rates)
            except (KeyError, ValueError):
                continue
            expected = convert_minor(int(inv["total_minor"]), inv["currency"], base, rate)
            booked = _safe_int(debits[0]["debit_minor"])
            diff = abs(booked - expected)
            if diff <= self._tolerance(expected):
                continue
            credit = next((r for r in rows if _safe_int(r["credit_minor"]) > 0), None)
            affected = [inv["doc_id"], debits[0]["row_id"]]
            if credit is not None:
                affected.append(credit["row_id"])
            correct_fx = f"fx:{day.isoformat()}:{inv['currency']}:{base}"
            evidence = affected + [correct_fx, "policy"]
            out.append(
                _finding(
                    "fx_conversion",
                    affected,
                    diff,
                    base,
                    evidence,
                    f"The {inv['currency']} invoice was not translated at the policy rate for "
                    f"{day.isoformat()}.",
                )
            )

        # Payment settlements.
        payment_rows: dict[str, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            if row.get("payment_id"):
                payment_rows[row["payment_id"]].append(row)
        for business_id, rows in sorted(payment_rows.items()):
            pay = idx["pay_by_business"].get(business_id)
            if pay is None or pay.get("kind") == "refund" or pay.get("currency") == base:
                continue
            cash = next((r for r in rows if r["account"] == ACCOUNT_CASH), None)
            if cash is None:
                continue
            try:
                day = parse_iso_datetime(pay["timestamp"]).date()
                rate = self._rate(pay["currency"], base, day, rates)
            except (KeyError, ValueError):
                continue
            expected = convert_minor(int(pay["amount_minor"]), pay["currency"], base, rate)
            booked = _safe_int(cash["debit_minor"]) + _safe_int(cash["credit_minor"])
            diff = abs(abs(booked) - expected)
            if diff <= self._tolerance(expected):
                continue
            affected = [pay["doc_id"], cash["row_id"]]
            fx_rows = [r for r in rows if r["account"] == "5100"]
            affected += [r["row_id"] for r in fx_rows]
            correct_fx = f"fx:{day.isoformat()}:{pay['currency']}:{base}"
            evidence = affected + [correct_fx, "policy"]
            out.append(
                _finding(
                    "fx_conversion",
                    affected,
                    diff,
                    base,
                    evidence,
                    f"The {pay['currency']} payment was not translated at the policy rate for "
                    f"{day.isoformat()}.",
                )
            )
        return out


# ---------------------------------------------------------------------------
# mark-all


class MarkAllPolicy(CollectingPolicy):
    """Low-precision policy: flag every plausible pattern up to the limit."""

    name = "mark_all"

    def _decide(self) -> dict:
        idx = self._indexes()
        base = self._base_currency()
        findings: list[dict] = []

        # Every repeated-looking ledger group is called a duplicate.
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in idx["ledger"]:
            key = (
                row["timestamp"],
                row["account"],
                _safe_int(row["debit_minor"]),
                _safe_int(row["credit_minor"]),
                row["entity_id"],
            )
            groups[key].append(row)
        for rows in groups.values():
            if len(rows) < 2:
                continue
            amount = sum(_safe_int(r["debit_minor"]) for r in rows) // 2
            if amount <= 0:
                continue
            affected = [r["row_id"] for r in rows]
            findings.append(
                _finding(
                    "duplicate_account",
                    affected,
                    amount,
                    base,
                    affected,
                    "Mark-all: repeated amount/account pattern.",
                )
            )
            if len(findings) >= MAX_FINDINGS:
                return {"episode_id": self._episode_id(), "findings": findings}

        # Every foreign document is called an FX error.
        for inv in idx["invoices"]:
            if inv.get("currency") == base:
                continue
            affected = [inv["doc_id"]]
            findings.append(
                _finding(
                    "fx_conversion",
                    affected,
                    1,
                    base,
                    affected + ["policy"],
                    "Mark-all: foreign-currency document.",
                )
            )
            if len(findings) >= MAX_FINDINGS:
                return {"episode_id": self._episode_id(), "findings": findings}

        # Every payment is called an allocation error.
        for pay in idx["payments"]:
            if pay.get("kind") == "refund":
                continue
            affected = [pay["doc_id"]]
            findings.append(
                _finding(
                    "payment_allocation",
                    affected,
                    1,
                    base,
                    affected + ["policy"],
                    "Mark-all: payment exists.",
                )
            )
            if len(findings) >= MAX_FINDINGS:
                return {"episode_id": self._episode_id(), "findings": findings}

        return {"episode_id": self._episode_id(), "findings": findings[:MAX_FINDINGS]}


# ---------------------------------------------------------------------------
# oracle


class OraclePolicy(Policy):
    """Evaluator-only upper bound; requires the hidden episode truth."""

    name = "oracle"
    evaluator_only = True

    def __init__(self, episode: Any) -> None:
        self._episode = episode
        self._submission = oracle_submission(episode)

    def act(self, obs: dict) -> dict:
        return {"type": "submit", "findings": self._submission}


POLICIES = {
    "random": RandomPolicy,
    "reconciliation": ReconciliationPolicy,
    "mark_all": MarkAllPolicy,
    "silence": SilencePolicy,
}


def make_policy(name: str, *, seed: int = 0, episode: Any = None) -> Policy:
    if name == "oracle":
        if episode is None:
            raise ValueError("oracle policy requires the hidden episode")
        return OraclePolicy(episode)
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "reconciliation":
        return ReconciliationPolicy()
    if name == "mark_all":
        return MarkAllPolicy()
    if name == "silence":
        return SilencePolicy()
    raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES) + ['oracle']}")
