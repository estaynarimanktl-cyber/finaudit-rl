"""Controlled error injection.

Every mutation preserves double-entry balance and records a causal change in
provenance.  Truth amounts and affected IDs are defined by the rules published
in README.md / policy.json.
"""

from __future__ import annotations

import copy
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Iterable

from .baseline import Baseline, Entry, _at, _clamp_same_period, _period_of
from .models import LedgerRow, Payment, Invoice
from .money import convert_minor, iso, local_date, minor_to_major, parse_iso_datetime, round_half_up
from .rng import Rng

TYPE_DUPLICATE = "duplicate_account"
TYPE_ALLOCATION = "payment_allocation"
TYPE_FX = "fx_conversion"
TYPE_PERIOD = "period_cutoff"

ALL_TYPES = [TYPE_DUPLICATE, TYPE_ALLOCATION, TYPE_FX, TYPE_PERIOD]

ACCOUNT_CASH = "1000"
ACCOUNT_AR = "1100"
ACCOUNT_PREPAID = "1200"
ACCOUNT_AP = "2000"
ACCOUNT_CUSTOMER_CREDIT = "2200"
ACCOUNT_REVENUE = "4000"
ACCOUNT_EXPENSE = "5000"
ACCOUNT_FX = "5100"
ACCOUNT_SUSPENSE = "9999"


# ---------------------------------------------------------------------------
# small helpers


def _rows(entry: Entry) -> list[LedgerRow]:
    return entry.rows


def _row_by_account(entry: Entry, account: str) -> LedgerRow | None:
    for row in entry.rows:
        if row.account == account:
            return row
    return None


def _source_row(entry: Entry) -> LedgerRow | None:
    for row in entry.rows:
        if row.account in (ACCOUNT_AR, ACCOUNT_AP):
            return row
    return None


def _cash_row(entry: Entry) -> LedgerRow | None:
    return _row_by_account(entry, ACCOUNT_CASH)


def _fx_row(entry: Entry) -> LedgerRow | None:
    return _row_by_account(entry, ACCOUNT_FX)


def _debit_row(entry: Entry) -> LedgerRow | None:
    for row in entry.rows:
        if int(row.debit_minor) > 0:
            return row
    return None


def _credit_row(entry: Entry) -> LedgerRow | None:
    for row in entry.rows:
        if int(row.credit_minor) > 0:
            return row
    return None


def _doc(baseline: Baseline, doc_id: str) -> Invoice | Payment | None:
    return baseline.metadata["invoice_by_doc"].get(doc_id) or baseline.metadata["payment_by_doc"].get(doc_id)


def _doc_dt(baseline: Baseline, doc_id: str) -> datetime:
    doc = _doc(baseline, doc_id)
    if isinstance(doc, Invoice):
        return parse_iso_datetime(doc.issued_at)
    if isinstance(doc, Payment):
        return parse_iso_datetime(doc.timestamp)
    raise KeyError(doc_id)


def _doc_period(baseline: Baseline, doc_id: str) -> int:
    return _period_of(
        _doc_dt(baseline, doc_id),
        baseline.metadata["period_start_dt"],
        baseline.metadata["period_end_dt"],
    )


def _fx_id(source: str, target: str, day: date) -> str:
    return f"fx:{day.isoformat()}:{source}:{target}"


def _clone_row(baseline: Baseline, row: LedgerRow, **overrides) -> LedgerRow:
    data = row.to_dict()
    data["row_id"] = baseline.new_row_id()
    data.update(overrides)
    return LedgerRow.from_dict(data)


def _set_row(row: LedgerRow, *, debit: int | None = None, credit: int | None = None) -> None:
    if debit is not None:
        row.debit_minor = int(debit)
    if credit is not None:
        row.credit_minor = int(credit)


def _new_suspense_row(baseline: Baseline, payment_row: LedgerRow, amount: int, side: str) -> LedgerRow:
    data = {
        "timestamp": payment_row.timestamp,
        "account": ACCOUNT_SUSPENSE,
        "debit_minor": amount if side == "debit" else 0,
        "credit_minor": amount if side == "credit" else 0,
        "currency": payment_row.currency,
        "entity_id": payment_row.entity_id,
        "invoice_id": None,
        "payment_id": payment_row.payment_id,
    }
    return LedgerRow(row_id=baseline.new_row_id(), **data)


def _rebuild_fx_plug(baseline: Baseline, entry: Entry) -> LedgerRow | None:
    """Recompute the 5100 plug so the entry balances after a cash change."""
    fx = _fx_row(entry)
    rows = [r for r in entry.rows if r is not fx]
    debit = sum(int(r.debit_minor) for r in rows)
    credit = sum(int(r.credit_minor) for r in rows)
    diff = debit - credit
    if fx is None:
        cash = _cash_row(entry)
        assert cash is not None
        fx = LedgerRow(
            row_id=baseline.new_row_id(),
            timestamp=cash.timestamp,
            account=ACCOUNT_FX,
            debit_minor=0,
            credit_minor=0,
            currency=cash.currency,
            entity_id=cash.entity_id,
            invoice_id=None,
            payment_id=cash.payment_id,
        )
        entry.rows.append(fx)
    if diff > 0:
        fx.debit_minor = 0
        fx.credit_minor = diff
    elif diff < 0:
        fx.debit_minor = -diff
        fx.credit_minor = 0
    else:
        # A zero plug is not a record. Remove it entirely.
        if fx in entry.rows:
            entry.rows.remove(fx)
        return None
    return fx


def _entry_balanced(entry: Entry) -> bool:
    return sum(int(r.debit_minor) for r in entry.rows) == sum(int(r.credit_minor) for r in entry.rows)


def _issue(
    issue_id: str,
    issue_type: str,
    subtype: str,
    affected: Iterable[str],
    amount: int,
    evidence_sets: list[list[str]],
    description: str,
    causal: dict,
) -> dict:
    return {
        "issue_id": issue_id,
        "type": issue_type,
        "subtype": subtype,
        "affected_ids": sorted(set(affected)),
        "expected_amount_minor": int(amount),
        "currency": None,  # filled by caller
        "evidence_sets": [sorted(set(s)) for s in evidence_sets],
        "description": description,
        "causal_change": causal,
    }


def _modification(issue: dict, action: str, before: dict, after: dict, rows_added: list[str], rows_modified: list[str]) -> dict:
    return {
        "issue_id": issue["issue_id"],
        "type": issue["type"],
        "subtype": issue["subtype"],
        "action": action,
        "before": before,
        "after": after,
        "rows_added": rows_added,
        "rows_modified": rows_modified,
        "expected_amount_minor": issue["expected_amount_minor"],
    }


# ---------------------------------------------------------------------------
# duplicate posting


def inject_duplicate(
    baseline: Baseline,
    rng: Rng,
    issue_id: str,
    *,
    target: Entry | None = None,
    max_rows: int = 4,
) -> tuple[dict, dict]:
    candidates = [
        e
        for e in baseline.entries
        if e.source_doc is not None
        and e.kind in ("invoice_accrual", "payment_settlement")
        and len(e.rows) <= max_rows
        and e.total_debit() > 0
    ]
    if target is None:
        if not candidates:
            raise RuntimeError("no eligible entry for duplicate injection")
        target = rng.choice(candidates)
    if len(target.rows) > max_rows:
        raise RuntimeError("duplicate target has too many rows")

    original_ids = [r.row_id for r in target.rows]
    before = {
        "entry_id": target.entry_id,
        "source_doc": target.source_doc,
        "rows": [r.to_dict() for r in target.rows],
        "total_debit": target.total_debit(),
    }
    # Same timestamp: still the same period, clearly a second posting.
    base_dt = parse_iso_datetime(target.rows[0].timestamp)
    dup_dt = base_dt
    if _period_of(base_dt, baseline.metadata["period_start_dt"], baseline.metadata["period_end_dt"]) == 0:
        candidate = base_dt + timedelta(seconds=1)
        if _period_of(candidate, baseline.metadata["period_start_dt"], baseline.metadata["period_end_dt"]) == 0:
            dup_dt = candidate
    dup_rows: list[LedgerRow] = []
    for row in target.rows:
        dup_rows.append(
            _clone_row(
                baseline,
                row,
                timestamp=iso(dup_dt),
            )
        )
    dup_entry = Entry(
        entry_id=baseline.new_entry_id(),
        rows=dup_rows,
        source_doc=target.source_doc,
        kind="duplicate_" + target.kind,
        invoice_id=target.invoice_id,
        payment_id=target.payment_id,
    )
    baseline.entries.append(dup_entry)
    dup_ids = [r.row_id for r in dup_rows]
    affected = original_ids + dup_ids
    expected = target.total_debit()
    evidence = affected + [target.source_doc]
    issue = _issue(
        issue_id,
        TYPE_DUPLICATE,
        "second_posting",
        affected,
        expected,
        [evidence],
        f"The source document {target.source_doc} was posted twice; the second balanced "
        f"journal entry duplicates the first. Expected overstatement is the booked debit total.",
        {
            "target_entry": target.entry_id,
            "duplicate_entry": dup_entry.entry_id,
            "source_doc": target.source_doc,
            "duplicate_timestamp": iso(dup_dt),
        },
    )
    mod = _modification(
        issue,
        "insert_duplicate_entry",
        before,
        {
            "entry_id": dup_entry.entry_id,
            "rows": [r.to_dict() for r in dup_rows],
        },
        dup_ids,
        [],
    )
    baseline.rebuild_ledger()
    return issue, mod


# ---------------------------------------------------------------------------
# payment allocation


def inject_payment_allocation(
    baseline: Baseline,
    rng: Rng,
    issue_id: str,
    *,
    target_payment_doc: str | None = None,
    subtype: str | None = None,
) -> tuple[dict, dict]:
    candidates = []
    for doc_id, entry in baseline.payment_entries.items():
        pay = baseline.metadata["payment_by_doc"].get(doc_id)
        if pay is None or pay.kind == "refund":
            continue
        src = _source_row(entry)
        cash = _cash_row(entry)
        if src is None or cash is None or src.amount_minor <= 0:
            continue
        candidates.append((doc_id, entry))
    if target_payment_doc is not None:
        entry = baseline.payment_entries[target_payment_doc]
        candidates = [(target_payment_doc, entry)]
    if not candidates:
        raise RuntimeError("no eligible payment for allocation injection")
    doc_id, entry = rng.choice(candidates)
    pay = baseline.metadata["payment_by_doc"][doc_id]
    src = _source_row(entry)
    cash = _cash_row(entry)
    assert src is not None and cash is not None
    invoice_business = src.invoice_id or (pay.invoice_refs[0] if pay.invoice_refs else None)
    if invoice_business is None:
        raise RuntimeError("payment has no invoice reference")
    invoice_doc = next((i.doc_id for i in baseline.episode.invoices if i.invoice_id == invoice_business), None)
    if invoice_doc is None:
        raise RuntimeError("invoice reference not found")
    if subtype is None:
        subtype = rng.weighted(
            ["missing_allocation", "partial_allocation", "wrong_invoice"],
            [0.30, 0.40, 0.30],
        )

    before = {
        "entry_id": entry.entry_id,
        "payment_doc": doc_id,
        "invoice_doc": invoice_doc,
        "rows": [r.to_dict() for r in entry.rows],
    }
    added: list[str] = []
    modified: list[str] = []

    if subtype == "missing_allocation":
        amount = src.amount_minor
        side = "credit" if src.credit_minor > 0 else "debit"
        suspense = _new_suspense_row(baseline, src, amount, side)
        entry.rows.remove(src)
        entry.rows.append(suspense)
        added.append(suspense.row_id)
        modified.append(src.row_id)  # removed
        affected = [doc_id, invoice_doc, suspense.row_id]
        expected = amount
        evidence = [doc_id, invoice_doc, suspense.row_id, cash.row_id, "policy"]
        description = (
            "The payment was received but its receivable/payable allocation was removed "
            "and parked in suspense; the referenced invoice remains open."
        )
    elif subtype == "partial_allocation":
        amount = max(1, int(round(src.amount_minor * rng.uniform(0.20, 0.60))))
        amount = min(amount, src.amount_minor - 1)
        if amount <= 0:
            raise RuntimeError("allocation amount too small")
        side = "credit" if src.credit_minor > 0 else "debit"
        if side == "credit":
            _set_row(src, credit=src.credit_minor - amount)
        else:
            _set_row(src, debit=src.debit_minor - amount)
        suspense = _new_suspense_row(baseline, src, amount, side)
        entry.rows.append(suspense)
        added.append(suspense.row_id)
        modified.append(src.row_id)
        affected = [doc_id, invoice_doc, src.row_id, suspense.row_id]
        expected = amount
        evidence = [doc_id, invoice_doc, src.row_id, suspense.row_id, cash.row_id, "policy"]
        description = (
            "Only part of the payment was allocated to the referenced invoice; the rest "
            "was parked in suspense, leaving an unexplained open balance."
        )
    elif subtype == "wrong_invoice":
        # Pick another invoice of the same kind and entity, different doc.
        wrong = None
        for inv in baseline.episode.invoices:
            if inv.doc_id == invoice_doc or inv.kind != pay.kind.replace("customer_receipt", "sale").replace("vendor_payment", "purchase"):
                continue
            if inv.entity_id != pay.entity_id:
                continue
            wrong = inv
            if rng.random() < 0.5:
                break
        if wrong is None:
            for inv in baseline.episode.invoices:
                if inv.doc_id != invoice_doc and inv.kind == ("sale" if pay.kind == "customer_receipt" else "purchase"):
                    wrong = inv
                    break
        if wrong is None:
            raise RuntimeError("no wrong-invoice target")
        before_invoice = src.invoice_id
        src.invoice_id = wrong.invoice_id
        modified.append(src.row_id)
        affected = [doc_id, invoice_doc, wrong.doc_id, src.row_id]
        expected = src.amount_minor
        evidence = [doc_id, invoice_doc, wrong.doc_id, src.row_id, cash.row_id, "policy"]
        description = (
            "The payment references one invoice but the ledger allocation was posted to a "
            "different invoice; the intended invoice remains open."
        )
    else:
        raise ValueError(subtype)

    if not _entry_balanced(entry):
        raise AssertionError("allocation injection unbalanced the entry")
    issue = _issue(
        issue_id,
        TYPE_ALLOCATION,
        subtype,
        affected,
        expected,
        [evidence],
        description,
        {
            "payment_doc": doc_id,
            "invoice_doc": invoice_doc,
            "subtype": subtype,
            "entry_id": entry.entry_id,
        },
    )
    mod = _modification(issue, "modify_allocation", before, {"rows": [r.to_dict() for r in entry.rows]}, added, modified)
    baseline.rebuild_ledger()
    return issue, mod


# ---------------------------------------------------------------------------
# FX conversion


def _available_dates(table, source: str, target: str) -> list[date]:
    dates = table.available_dates(source, target)
    return dates


def _choose_wrong_date(
    table,
    source: str,
    target: str,
    correct_day: date,
    rng: Rng,
    *,
    min_bps: float = 20.0,
    max_bps: float = 600.0,
    max_days: int = 45,
) -> tuple[date, Decimal] | None:
    correct = table.rate(source, target, correct_day)
    candidates = []
    for day in _available_dates(table, source, target):
        delta = abs((day - correct_day).days)
        if delta == 0 or delta > max_days:
            continue
        rate = table.rate(source, target, day)
        if correct == 0:
            continue
        bps = abs(float((rate - correct) / correct) * 10000)
        if min_bps <= bps <= max_bps:
            candidates.append((delta, bps, day, rate))
    if not candidates:
        return None
    preferred = [c for c in candidates if 3 <= c[0] <= 10] or candidates
    preferred.sort(key=lambda c: (-c[1], c[0], c[2]))
    chosen = rng.choice(preferred[:6])
    return chosen[2], chosen[3]


def inject_fx(
    baseline: Baseline,
    rng: Rng,
    issue_id: str,
    *,
    target: Entry | None = None,
    target_kind: str | None = None,
) -> tuple[dict, dict]:
    candidates: list[tuple[Entry, str]] = []
    for entry in baseline.entries:
        if entry.source_doc is None:
            continue
        if entry.kind == "invoice_accrual":
            inv = baseline.metadata["invoice_by_doc"].get(entry.source_doc)
            if inv is not None and inv.currency != baseline.episode.base_currency and entry.total_debit() >= 50_000:
                candidates.append((entry, "accrual"))
        elif entry.kind == "payment_settlement":
            pay = baseline.metadata["payment_by_doc"].get(entry.source_doc)
            if pay is not None and pay.currency != baseline.episode.base_currency and pay.amount_minor > 0:
                cash = _cash_row(entry)
                if cash is not None and cash.amount_minor >= 50_000:
                    candidates.append((entry, "settlement"))
    if target is not None:
        kind = target_kind or ("invoice_accrual" if target.kind == "invoice_accrual" else "settlement")
        candidates = [(target, kind)]
    if not candidates:
        raise RuntimeError("no eligible foreign entry for FX injection")
    if target_kind is not None:
        candidates = [c for c in candidates if c[1] == target_kind] or candidates
    entry, kind = rng.choice(candidates)

    table = baseline.episode.fx_table()
    base = baseline.episode.base_currency
    before = {"entry_id": entry.entry_id, "rows": [r.to_dict() for r in entry.rows], "kind": kind}
    added: list[str] = []
    modified: list[str] = []

    if kind == "accrual":
        inv = baseline.metadata["invoice_by_doc"][entry.source_doc]
        doc_day = parse_iso_datetime(inv.issued_at).date()
        correct_rate = table.rate(inv.currency, base, doc_day)
        correct_amount = convert_minor(inv.total_minor, inv.currency, base, correct_rate)
        wrong = _choose_wrong_date(table, inv.currency, base, doc_day, rng)
        if wrong is None:
            raise RuntimeError("no wrong FX date for accrual")
        wrong_day, wrong_rate = wrong
        wrong_amount = convert_minor(inv.total_minor, inv.currency, base, wrong_rate)
        diff = abs(wrong_amount - correct_amount)
        if diff <= 0:
            raise RuntimeError("FX difference is zero")
        debit = _debit_row(entry)
        credit = _credit_row(entry)
        assert debit is not None and credit is not None
        _set_row(debit, debit=wrong_amount)
        _set_row(credit, credit=wrong_amount)
        modified.extend([debit.row_id, credit.row_id])
        affected = [inv.doc_id, debit.row_id, credit.row_id]
        correct_fx = _fx_id(inv.currency, base, doc_day)
        wrong_fx = _fx_id(inv.currency, base, wrong_day)
        evidence_sets = [
            [inv.doc_id, debit.row_id, credit.row_id, correct_fx, "policy"],
            [inv.doc_id, debit.row_id, credit.row_id, correct_fx, wrong_fx, "policy"],
        ]
        description = (
            f"The {inv.currency} invoice was translated at the rate for {wrong_day.isoformat()} "
            f"instead of the policy date {doc_day.isoformat()}; the base-currency accrual is wrong."
        )
        expected = diff
    else:
        pay = baseline.metadata["payment_by_doc"][entry.source_doc]
        pay_day = parse_iso_datetime(pay.timestamp).date()
        correct_rate = table.rate(pay.currency, base, pay_day)
        correct_cash = convert_minor(pay.amount_minor, pay.currency, base, correct_rate)
        wrong = _choose_wrong_date(table, pay.currency, base, pay_day, rng)
        if wrong is None:
            raise RuntimeError("no wrong FX date for settlement")
        wrong_day, wrong_rate = wrong
        wrong_cash = convert_minor(pay.amount_minor, pay.currency, base, wrong_rate)
        diff = abs(wrong_cash - correct_cash)
        if diff <= 0:
            raise RuntimeError("FX difference is zero")
        before_ids = {r.row_id for r in entry.rows}
        cash = _cash_row(entry)
        assert cash is not None
        if cash.debit_minor > 0:
            _set_row(cash, debit=wrong_cash)
        else:
            _set_row(cash, credit=wrong_cash)
        fx = _rebuild_fx_plug(baseline, entry)
        added = [r.row_id for r in entry.rows if r.row_id not in before_ids]
        modified = [cash.row_id]
        if fx is not None and fx.row_id not in added:
            modified.append(fx.row_id)
        affected = [pay.doc_id, cash.row_id] + ([fx.row_id] if fx is not None else [])
        correct_fx = _fx_id(pay.currency, base, pay_day)
        wrong_fx = _fx_id(pay.currency, base, wrong_day)
        evidence_sets = [
            [pay.doc_id, cash.row_id] + ([fx.row_id] if fx is not None else []) + [correct_fx, "policy"],
            [pay.doc_id, cash.row_id] + ([fx.row_id] if fx is not None else []) + [correct_fx, wrong_fx, "policy"],
        ]
        plug_text = "cash and the FX plug are wrong" if fx is not None else "cash is wrong"
        description = (
            f"The {pay.currency} payment was translated at the rate for {wrong_day.isoformat()} "
            f"instead of the policy date {pay_day.isoformat()}; {plug_text}."
        )
        expected = diff

    if not _entry_balanced(entry):
        raise AssertionError("FX injection unbalanced the entry")
    issue = _issue(
        issue_id,
        TYPE_FX,
        "wrong_rate_date",
        affected,
        expected,
        evidence_sets,
        description,
        {
            "entry_id": entry.entry_id,
            "kind": kind,
            "correct_day": doc_day.isoformat() if kind == "accrual" else pay_day.isoformat(),
            "wrong_day": wrong_day.isoformat(),
            "correct_rate": str(correct_rate),
            "wrong_rate": str(wrong_rate),
        },
    )
    mod = _modification(issue, "change_fx_translation", before, {"rows": [r.to_dict() for r in entry.rows]}, added, modified)
    baseline.rebuild_ledger()
    return issue, mod


# ---------------------------------------------------------------------------
# period cutoff


def inject_period_cutoff(
    baseline: Baseline,
    rng: Rng,
    issue_id: str,
    *,
    target: Entry | None = None,
) -> tuple[dict, dict]:
    candidates = [
        e
        for e in baseline.entries
        if e.source_doc is not None
        and e.kind in ("invoice_accrual", "payment_settlement")
        and len(e.rows) <= 9
        and e.total_debit() > 0
    ]
    if target is not None:
        candidates = [target]
    if not candidates:
        raise RuntimeError("no eligible entry for period cutoff")
    entry = rng.choice(candidates)
    doc_period = _doc_period(baseline, entry.source_doc)
    start = baseline.metadata["period_start_dt"]
    end = baseline.metadata["period_end_dt"]
    tz = baseline.metadata["tz"]

    if doc_period == 0:
        if rng.random() < 0.5:
            new_day = (start - timedelta(days=rng.randint(1, 5))).date()
        else:
            new_day = (end + timedelta(days=rng.randint(1, 5))).date()
    elif doc_period < 0:
        if rng.random() < 0.6:
            new_day = (start + timedelta(days=rng.randint(1, 5))).date()
        else:
            new_day = (end + timedelta(days=rng.randint(1, 5))).date()
    else:
        if rng.random() < 0.6:
            new_day = (end - timedelta(days=rng.randint(1, 5))).date()
        else:
            new_day = (start - timedelta(days=rng.randint(1, 5))).date()
    new_dt = _at(new_day, rng.randint(9, 17), rng.randint(0, 59), tz)
    if new_dt == parse_iso_datetime(entry.rows[0].timestamp):
        new_dt = new_dt + timedelta(hours=1)

    before = {"entry_id": entry.entry_id, "rows": [r.to_dict() for r in entry.rows], "doc_period": doc_period}
    old_ts = entry.rows[0].timestamp
    for row in entry.rows:
        row.timestamp = iso(new_dt)
    affected = [entry.source_doc] + [r.row_id for r in entry.rows]
    expected = entry.total_debit()
    evidence = [entry.source_doc] + [r.row_id for r in entry.rows] + ["policy"]
    issue = _issue(
        issue_id,
        TYPE_PERIOD,
        "cross_period_posting",
        affected,
        expected,
        [evidence],
        f"The source document belongs to period {doc_period} but the journal entry was "
        f"posted at {iso(new_dt)}, which is in a different period; the policy requires "
        f"the entry period to equal the document period.",
        {
            "entry_id": entry.entry_id,
            "doc_period": doc_period,
            "old_timestamp": old_ts,
            "new_timestamp": iso(new_dt),
        },
    )
    mod = _modification(issue, "move_entry_across_period", before, {"rows": [r.to_dict() for r in entry.rows]}, [], [r.row_id for r in entry.rows])
    baseline.rebuild_ledger()
    return issue, mod
