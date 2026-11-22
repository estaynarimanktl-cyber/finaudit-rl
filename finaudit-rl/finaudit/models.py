"""Data models and on-disk episode format.

Open files (visible through the environment):
    manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv

Private files (evaluator only):
    truth.json, provenance.json
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from .money import FxRate, FxTable, format_rate, parse_iso_datetime
from .version import GENERATOR_VERSION, SCHEMA_VERSION

LEDGER_COLUMNS = [
    "row_id",
    "timestamp",
    "account",
    "debit_minor",
    "credit_minor",
    "currency",
    "entity_id",
    "invoice_id",
    "payment_id",
]

NULL = "null"


def _json_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")


@dataclass
class LedgerRow:
    row_id: str
    timestamp: str
    account: str
    debit_minor: int
    credit_minor: int
    currency: str
    entity_id: str
    invoice_id: str | None = None
    payment_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "timestamp": self.timestamp,
            "account": self.account,
            "debit_minor": int(self.debit_minor),
            "credit_minor": int(self.credit_minor),
            "currency": self.currency,
            "entity_id": self.entity_id,
            "invoice_id": self.invoice_id,
            "payment_id": self.payment_id,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "LedgerRow":
        return cls(
            row_id=row["row_id"],
            timestamp=row["timestamp"],
            account=row["account"],
            debit_minor=int(row["debit_minor"]),
            credit_minor=int(row["credit_minor"]),
            currency=row["currency"],
            entity_id=row["entity_id"],
            invoice_id=row.get("invoice_id") or None,
            payment_id=row.get("payment_id") or None,
        )

    @property
    def amount_minor(self) -> int:
        return int(self.debit_minor) + int(self.credit_minor)


@dataclass
class Invoice:
    doc_id: str
    invoice_id: str
    entity_id: str
    issued_at: str
    due_at: str
    currency: str
    total_minor: int
    line_items: list[dict]
    counterparty_id: str
    kind: str  # "sale" | "purchase"

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "invoice_id": self.invoice_id,
            "entity_id": self.entity_id,
            "issued_at": self.issued_at,
            "due_at": self.due_at,
            "currency": self.currency,
            "total_minor": int(self.total_minor),
            "line_items": self.line_items,
            "counterparty_id": self.counterparty_id,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "Invoice":
        return cls(
            doc_id=row["doc_id"],
            invoice_id=row["invoice_id"],
            entity_id=row["entity_id"],
            issued_at=row["issued_at"],
            due_at=row["due_at"],
            currency=row["currency"],
            total_minor=int(row["total_minor"]),
            line_items=list(row.get("line_items") or []),
            counterparty_id=row.get("counterparty_id", ""),
            kind=row.get("kind", "sale"),
        )


@dataclass
class Payment:
    doc_id: str
    payment_id: str
    entity_id: str
    timestamp: str
    currency: str
    amount_minor: int
    invoice_refs: list[str]
    counterparty_id: str
    kind: str  # "customer_receipt" | "vendor_payment" | "refund"

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "payment_id": self.payment_id,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "currency": self.currency,
            "amount_minor": int(self.amount_minor),
            "invoice_refs": list(self.invoice_refs),
            "counterparty_id": self.counterparty_id,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "Payment":
        return cls(
            doc_id=row["doc_id"],
            payment_id=row["payment_id"],
            entity_id=row["entity_id"],
            timestamp=row["timestamp"],
            currency=row["currency"],
            amount_minor=int(row["amount_minor"]),
            invoice_refs=list(row.get("invoice_refs") or []),
            counterparty_id=row.get("counterparty_id", ""),
            kind=row.get("kind", "customer_receipt"),
        )


@dataclass
class Episode:
    manifest: dict
    policy: dict
    ledger: list[LedgerRow]
    invoices: list[Invoice]
    payments: list[Payment]
    fx: list[FxRate]
    truth: dict | None = None
    provenance: dict | None = None

    # ---- basic properties -------------------------------------------------
    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    @property
    def base_currency(self) -> str:
        return self.manifest["base_currency"]

    @property
    def timezone(self) -> str:
        return self.manifest["timezone"]

    @property
    def period_start(self) -> str:
        return self.manifest["period_start"]

    @property
    def period_end(self) -> str:
        return self.manifest["period_end"]

    # ---- indexes ----------------------------------------------------------
    def ledger_index(self) -> dict[str, LedgerRow]:
        return {row.row_id: row for row in self.ledger}

    def invoice_index(self) -> dict[str, Invoice]:
        return {inv.doc_id: inv for inv in self.invoices}

    def payment_index(self) -> dict[str, Payment]:
        return {pay.doc_id: pay for pay in self.payments}

    def fx_index(self) -> dict[str, FxRate]:
        return {rate.fx_id: rate for rate in self.fx}

    def id_alias_map(self) -> dict[str, str]:
        """Map business IDs to canonical record IDs.

        Canonical IDs are ledger ``row_id``, invoice/payment ``doc_id``,
        ``fx:<date>:<src>:<dst>`` and the literal ``policy``.
        """
        aliases: dict[str, str] = {}
        for row in self.ledger:
            aliases[row.row_id] = row.row_id
        for inv in self.invoices:
            aliases[inv.doc_id] = inv.doc_id
            aliases[inv.invoice_id] = inv.doc_id
        for pay in self.payments:
            aliases[pay.doc_id] = pay.doc_id
            aliases[pay.payment_id] = pay.doc_id
        for rate in self.fx:
            aliases[rate.fx_id] = rate.fx_id
        aliases["policy"] = "policy"
        aliases["manifest"] = "manifest"
        return aliases

    def resolve_id(self, raw: str) -> str:
        try:
            return self.id_alias_map()[raw]
        except KeyError as exc:
            raise KeyError(f"unknown record id {raw!r}") from exc

    def fx_table(self) -> FxTable:
        fallback = self.policy.get("fx_fallback", "nearest_prior")
        return FxTable(self.fx, self.base_currency, fallback=fallback)

    # ---- validation -------------------------------------------------------
    def trial_balance(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for row in self.ledger:
            totals[row.account] = totals.get(row.account, 0) + int(row.debit_minor) - int(row.credit_minor)
        return totals

    def total_debits(self) -> int:
        return sum(int(r.debit_minor) for r in self.ledger)

    def total_credits(self) -> int:
        return sum(int(r.credit_minor) for r in self.ledger)

    def is_balanced(self) -> bool:
        return self.total_debits() == self.total_credits()

    def journal_groups(self) -> dict[tuple, list[LedgerRow]]:
        """Group rows that belong to one economic journal entry.

        A group key is (entity, invoice_id, payment_id, timestamp, account pair)
        is too fragile; the generator records explicit ``journal_id`` in
        provenance.  This helper is used only for tests and duplicate checks.
        """
        groups: dict[tuple, list[LedgerRow]] = {}
        for row in self.ledger:
            key = (row.entity_id, row.invoice_id, row.payment_id, row.timestamp)
            groups.setdefault(key, []).append(row)
        return groups

    # ---- serialization ----------------------------------------------------
    def open_dicts(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "policy": self.policy,
            "ledger": [r.to_dict() for r in self.ledger],
            "invoices": [i.to_dict() for i in self.invoices],
            "payments": [p.to_dict() for p in self.payments],
            "fx": [
                {
                    "date": r.date.isoformat(),
                    "source_currency": r.source_currency,
                    "target_currency": r.target_currency,
                    "rate": format_rate(r.rate),
                }
                for r in self.fx
            ],
        }

    def write(self, directory: str | os.PathLike, *, include_private: bool = True) -> Path:
        root = Path(directory)
        open_dir = root / "open"
        open_dir.mkdir(parents=True, exist_ok=True)
        _write_json(open_dir / "manifest.json", self.manifest)
        _write_json(open_dir / "policy.json", self.policy)

        with (open_dir / "ledger.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
            writer.writeheader()
            for row in self.ledger:
                data = row.to_dict()
                data["invoice_id"] = data["invoice_id"] if data["invoice_id"] is not None else NULL
                data["payment_id"] = data["payment_id"] if data["payment_id"] is not None else NULL
                writer.writerow(data)

        with (open_dir / "invoices.jsonl").open("w", encoding="utf-8") as fh:
            for inv in self.invoices:
                fh.write(json.dumps(inv.to_dict(), ensure_ascii=False, sort_keys=False) + "\n")
        with (open_dir / "payments.jsonl").open("w", encoding="utf-8") as fh:
            for pay in self.payments:
                fh.write(json.dumps(pay.to_dict(), ensure_ascii=False, sort_keys=False) + "\n")
        with (open_dir / "fx.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "source_currency", "target_currency", "rate"])
            for rate in sorted(self.fx, key=lambda r: (r.date, r.source_currency, r.target_currency)):
                writer.writerow([rate.date.isoformat(), rate.source_currency, rate.target_currency, format_rate(rate.rate)])

        if include_private:
            private_dir = root / "private"
            private_dir.mkdir(parents=True, exist_ok=True)
            _write_json(private_dir / "truth.json", self.truth or {})
            _write_json(private_dir / "provenance.json", self.provenance or {})
        return root

    @classmethod
    def load(cls, directory: str | os.PathLike, *, include_private: bool = True) -> "Episode":
        root = Path(directory)
        open_dir = root / "open" if (root / "open").is_dir() else root
        manifest = _read_json(open_dir / "manifest.json")
        policy = _read_json(open_dir / "policy.json")
        ledger: list[LedgerRow] = []
        with (open_dir / "ledger.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for key in ("invoice_id", "payment_id"):
                    if row.get(key) in ("", NULL, "None"):
                        row[key] = None
                ledger.append(LedgerRow.from_dict(row))
        invoices = [Invoice.from_dict(json.loads(line)) for line in (open_dir / "invoices.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        payments = [Payment.from_dict(json.loads(line)) for line in (open_dir / "payments.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        fx: list[FxRate] = []
        with (open_dir / "fx.csv").open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                fx.append(
                    FxRate(
                        date=date.fromisoformat(row["date"]),
                        source_currency=row["source_currency"],
                        target_currency=row["target_currency"],
                        rate=__import__("decimal").Decimal(row["rate"]),
                    )
                )
        truth = None
        provenance = None
        private_dir = root / "private"
        if include_private and private_dir.is_dir():
            truth_path = private_dir / "truth.json"
            prov_path = private_dir / "provenance.json"
            if truth_path.is_file():
                truth = _read_json(truth_path)
            if prov_path.is_file():
                provenance = _read_json(prov_path)
        return cls(manifest, policy, ledger, invoices, payments, fx, truth, provenance)

    # ---- ids --------------------------------------------------------------
    def all_record_ids(self) -> set[str]:
        ids = {r.row_id for r in self.ledger}
        ids.update(i.doc_id for i in self.invoices)
        ids.update(p.doc_id for p in self.payments)
        ids.update(r.fx_id for r in self.fx)
        ids.add("policy")
        return ids

    def clone(self) -> "Episode":
        import copy

        return copy.deepcopy(self)


def make_manifest(
    *,
    episode_id: str,
    base_currency: str,
    period_start: str,
    period_end: str,
    timezone: str,
    n_records: int,
    n_invoices: int,
    n_payments: int,
    seed_commitment: str,
    difficulty: str,
    config: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "episode_id": episode_id,
        "base_currency": base_currency,
        "period_start": period_start,
        "period_end": period_end,
        "timezone": timezone,
        "n_records": n_records,
        "n_documents": n_invoices + n_payments,
        "n_invoices": n_invoices,
        "n_payments": n_payments,
        "currency_minor_units": {
            "USD": 2,
            "EUR": 2,
            "GBP": 2,
            "KZT": 2,
            "AED": 2,
            "JPY": 0,
        },
        "rounding": {
            "money": "integer_minor_units",
            "mode": "half_up",
            "fx_decimals": 8,
            "amount_tolerance_minor": 2,
            "amount_tolerance_bps": 5,
        },
        "limits": {
            "max_steps": 100,
            "page_size": 50,
            "max_findings": 30,
            "max_evidence_ids": 10,
            "max_explanation_chars": 1000,
            "time_limit_seconds": 60,
        },
        "seed_commitment": seed_commitment,
        "difficulty": difficulty,
        "config": config,
    }


def make_policy(
    *,
    base_currency: str,
    period_start: str,
    period_end: str,
    timezone: str,
    accounts: dict,
) -> dict:
    return {
        "policy_id": "policy",
        "schema_version": SCHEMA_VERSION,
        "base_currency": base_currency,
        "ledger_currency": base_currency,
        "period_start": period_start,
        "period_end": period_end,
        "timezone": timezone,
        "fx_rate_date_rule": "document_date",
        "fx_fallback": "nearest_prior",
        "rounding": "half_up",
        "amount_tolerance_minor": 2,
        "amount_tolerance_bps": 5,
        "recognition_rule": "ledger_period_must_equal_document_period",
        "canonical_affected_ids_rule": (
            "Affected IDs are the records whose own values are wrong, not every record "
            "needed as evidence. Use ledger row_id, invoice/payment doc_id, fx composite ID "
            "or 'policy'. Business invoice_id/payment_id aliases are accepted and normalized."
        ),
        "accounts": accounts,
        "period_definition": (
            "A journal entry is in-period when its timestamp is inside "
            "[period_start, period_end] in the company timezone.  The recognition "
            "rule requires the entry period to equal the source document period "
            "(invoice issued_at for accruals; payment timestamp for settlements)."
        ),
    }
