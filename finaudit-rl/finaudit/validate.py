"""Independent validation of generated data and truth.

The checks here are intentionally written against the public records and the
policy, not against the generator's private bookkeeping.  They are used by the
test suite and by the ``finaudit validate`` command.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Iterable

from .inject import (
    ACCOUNT_AR,
    ACCOUNT_AP,
    ACCOUNT_CASH,
    ACCOUNT_FX,
    ACCOUNT_SUSPENSE,
    TYPE_ALLOCATION,
    TYPE_DUPLICATE,
    TYPE_FX,
    TYPE_PERIOD,
)
from .models import Episode, Invoice, LedgerRow, Payment
from .money import convert_minor, local_date, parse_iso_datetime


def _period_of(dt: datetime, episode: Episode) -> int:
    start = parse_iso_datetime(episode.period_start)
    end = parse_iso_datetime(episode.period_end)
    if dt < start:
        return -1
    if dt > end:
        return 1
    return 0


def _doc_period(episode: Episode, doc: Invoice | Payment) -> int:
    if isinstance(doc, Invoice):
        return _period_of(parse_iso_datetime(doc.issued_at), episode)
    return _period_of(parse_iso_datetime(doc.timestamp), episode)


def _resolve(episode: Episode, ids: Iterable[str]) -> tuple[list[LedgerRow], list[Invoice], list[Payment], list]:
    ledger_idx = episode.ledger_index()
    inv_idx = episode.invoice_index()
    pay_idx = episode.payment_index()
    fx_idx = episode.fx_index()
    rows, invoices, payments, fx = [], [], [], []
    for raw in ids:
        try:
            cid = episode.resolve_id(raw)
        except KeyError as exc:
            raise AssertionError(f"unknown id in truth: {raw}") from exc
        if cid in ledger_idx:
            rows.append(ledger_idx[cid])
        elif cid in inv_idx:
            invoices.append(inv_idx[cid])
        elif cid in pay_idx:
            payments.append(pay_idx[cid])
        elif cid in fx_idx:
            fx.append(fx_idx[cid])
        elif cid == "policy":
            pass
        else:
            raise AssertionError(f"unresolved id {raw}")
    return rows, invoices, payments, fx


def _tolerance(episode: Episode, expected: int) -> int:
    rounding = episode.manifest.get("rounding", {})
    abs_tol = int(rounding.get("amount_tolerance_minor", 2))
    bps = float(rounding.get("amount_tolerance_bps", 5))
    return max(abs_tol, int(round(expected * bps / 10000.0)))


def _source_doc_for_rows(episode: Episode, rows: list[LedgerRow]) -> Invoice | Payment:
    inv_by_business = {i.invoice_id: i for i in episode.invoices}
    pay_by_business = {p.payment_id: p for p in episode.payments}
    for row in rows:
        if row.invoice_id and row.invoice_id in inv_by_business:
            return inv_by_business[row.invoice_id]
        if row.payment_id and row.payment_id in pay_by_business:
            return pay_by_business[row.payment_id]
    raise AssertionError("cannot identify source document from affected rows")


def _check_duplicate(episode: Episode, issue: dict) -> None:
    rows, _, _, _ = _resolve(episode, issue["affected_ids"])
    if not rows:
        raise AssertionError("duplicate issue has no ledger rows")
    doc = _source_doc_for_rows(episode, rows)
    # Every affected row must have a twin with the same account, sides and amount.
    seen: set[str] = set()
    groups = 0
    for row in rows:
        if row.row_id in seen:
            continue
        twin = None
        for other in rows:
            if other.row_id == row.row_id or other.row_id in seen:
                continue
            if (
                other.account == row.account
                and int(other.debit_minor) == int(row.debit_minor)
                and int(other.credit_minor) == int(row.credit_minor)
                and other.entity_id == row.entity_id
            ):
                twin = other
                break
        if twin is None:
            raise AssertionError(f"duplicate row {row.row_id} has no twin")
        seen.add(row.row_id)
        seen.add(twin.row_id)
        groups += 1
    if groups * 2 != len(rows):
        raise AssertionError("duplicate affected rows are not paired")
    debit_total = sum(int(r.debit_minor) for r in rows) // 2
    if abs(debit_total - issue["expected_amount_minor"]) > _tolerance(episode, issue["expected_amount_minor"]):
        raise AssertionError(
            f"duplicate amount mismatch: {debit_total} vs {issue['expected_amount_minor']}"
        )
    if not isinstance(doc, (Invoice, Payment)):
        raise AssertionError("duplicate source doc missing")


def _check_allocation(episode: Episode, issue: dict) -> None:
    rows, invoices, payments, _ = _resolve(episode, issue["affected_ids"])
    if not payments:
        raise AssertionError("allocation issue has no payment doc")
    pay = payments[0]
    expected = issue["expected_amount_minor"]
    subtype = issue.get("subtype")
    suspense = [r for r in rows if r.account == ACCOUNT_SUSPENSE]
    source_rows = [r for r in rows if r.account in (ACCOUNT_AR, ACCOUNT_AP)]
    if subtype in ("missing_allocation", "partial_allocation"):
        if not suspense:
            raise AssertionError("allocation issue has no suspense row")
        if abs(suspense[0].amount_minor - expected) > _tolerance(episode, expected):
            raise AssertionError(
                f"suspense amount {suspense[0].amount_minor} != expected {expected}"
            )
        if not invoices:
            raise AssertionError("allocation issue has no invoice doc")
        inv = invoices[0]
        if inv.invoice_id not in (pay.invoice_refs or []):
            raise AssertionError("payment does not reference the affected invoice")
        if subtype == "missing_allocation":
            if source_rows:
                raise AssertionError("missing allocation still has a source row")
        else:
            if not source_rows:
                raise AssertionError("partial allocation lost its source row")
    elif subtype == "wrong_invoice":
        if len(invoices) < 2:
            raise AssertionError("wrong-invoice issue needs two invoices")
        if not source_rows:
            raise AssertionError("wrong-invoice issue has no source row")
        src = source_rows[0]
        wrong = [i for i in invoices if i.invoice_id == src.invoice_id]
        if not wrong:
            raise AssertionError("source row does not point at the wrong invoice")
        if wrong[0].invoice_id in (pay.invoice_refs or []):
            raise AssertionError("wrong invoice is also referenced by the payment")
        if abs(src.amount_minor - expected) > _tolerance(episode, expected):
            raise AssertionError("wrong-invoice amount mismatch")
    else:
        raise AssertionError(f"unknown allocation subtype {subtype}")


def _check_fx(episode: Episode, issue: dict) -> None:
    rows, invoices, payments, fx = _resolve(episode, issue["affected_ids"])
    if not invoices and not payments:
        raise AssertionError("FX issue has no source document")
    doc = invoices[0] if invoices else payments[0]
    base = episode.base_currency
    evidence_ids = {raw for s in issue.get("evidence_sets", []) for raw in s}
    if isinstance(doc, Invoice):
        day = parse_iso_datetime(doc.issued_at).date()
        correct = convert_minor(
            doc.total_minor, doc.currency, base, episode.fx_table().rate(doc.currency, base, day)
        )
        debit_rows = [r for r in rows if int(r.debit_minor) > 0]
        if not debit_rows:
            raise AssertionError("FX accrual has no debit row")
        booked = sum(int(r.debit_minor) for r in debit_rows)
        if abs(abs(booked - correct) - issue["expected_amount_minor"]) > _tolerance(
            episode, issue["expected_amount_minor"]
        ):
            raise AssertionError(
                f"FX accrual difference {abs(booked-correct)} != {issue['expected_amount_minor']}"
            )
        correct_fx = episode.fx_table().fx_id(doc.currency, base, day)
    elif isinstance(doc, Payment):
        day = parse_iso_datetime(doc.timestamp).date()
        correct = convert_minor(
            doc.amount_minor, doc.currency, base, episode.fx_table().rate(doc.currency, base, day)
        )
        cash = [r for r in rows if r.account == ACCOUNT_CASH]
        if not cash:
            raise AssertionError("FX payment has no cash row")
        booked = cash[0].amount_minor
        if abs(abs(booked - correct) - issue["expected_amount_minor"]) > _tolerance(
            episode, issue["expected_amount_minor"]
        ):
            raise AssertionError(
                f"FX payment difference {abs(booked-correct)} != {issue['expected_amount_minor']}"
            )
        correct_fx = episode.fx_table().fx_id(doc.currency, base, day)
        # A zero FX plug may legitimately be absent; cash is the primary record.
    else:
        raise AssertionError("bad FX source doc")
    if correct_fx not in evidence_ids:
        raise AssertionError(f"correct FX row {correct_fx} missing from evidence")


def _check_period(episode: Episode, issue: dict) -> None:
    rows, invoices, payments, _ = _resolve(episode, issue["affected_ids"])
    if not invoices and not payments:
        raise AssertionError("period issue has no source document")
    doc = invoices[0] if invoices else payments[0]
    doc_p = _doc_period(episode, doc)
    row_periods = {_period_of(parse_iso_datetime(r.timestamp), episode) for r in rows}
    if not row_periods:
        raise AssertionError("period issue has no rows")
    if doc_p in row_periods:
        raise AssertionError("period issue rows are in the document period")
    debit_total = sum(int(r.debit_minor) for r in rows)
    if abs(debit_total - issue["expected_amount_minor"]) > _tolerance(episode, issue["expected_amount_minor"]):
        raise AssertionError(
            f"period amount {debit_total} != {issue['expected_amount_minor']}"
        )


def validate_episode(episode: Episode, *, check_truth: bool = True) -> list[str]:
    errors: list[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            errors.append(message)

    # Manifest and policy
    check(episode.manifest.get("schema_version") == "1.0", "manifest schema_version")
    check(episode.manifest.get("episode_id") == episode.episode_id, "manifest episode_id")
    check(episode.base_currency in episode.manifest.get("currency_minor_units", {}), "base currency units")
    check(episode.manifest.get("n_records") == len(episode.ledger), "manifest n_records")
    check(
        episode.manifest.get("n_documents") == len(episode.invoices) + len(episode.payments),
        "manifest n_documents",
    )
    check(200 <= len(episode.ledger) <= 2000, "ledger size outside 200..2000")
    check(20 <= len(episode.invoices) + len(episode.payments) <= 200, "document count outside 20..200")
    check(episode.policy.get("base_currency") == episode.base_currency, "policy base currency")
    check(episode.policy.get("period_start") == episode.period_start, "policy period start")
    check(episode.policy.get("period_end") == episode.period_end, "policy period end")

    # IDs
    row_ids = [r.row_id for r in episode.ledger]
    check(len(row_ids) == len(set(row_ids)), "duplicate ledger row ids")
    inv_ids = [i.doc_id for i in episode.invoices]
    pay_ids = [p.doc_id for p in episode.payments]
    check(len(inv_ids) == len(set(inv_ids)), "duplicate invoice doc ids")
    check(len(pay_ids) == len(set(pay_ids)), "duplicate payment doc ids")
    check(len(set(inv_ids) & set(pay_ids)) == 0, "invoice/payment doc id collision")

    inv_by_business = {i.invoice_id: i for i in episode.invoices}
    pay_by_business = {p.payment_id: p for p in episode.payments}
    check(len(inv_by_business) == len(episode.invoices), "duplicate invoice business ids")
    check(len(pay_by_business) == len(episode.payments), "duplicate payment business ids")

    # Documents
    for inv in episode.invoices:
        check(inv.total_minor > 0, f"{inv.doc_id}: non-positive total")
        check(sum(int(x["amount_minor"]) for x in inv.line_items) == inv.total_minor, f"{inv.doc_id}: line item sum")
        for item in inv.line_items:
            check(
                int(item["amount_minor"]) == int(item["quantity"]) * int(item["unit_price_minor"]),
                f"{inv.doc_id}: line item arithmetic",
            )
        check(parse_iso_datetime(inv.due_at) >= parse_iso_datetime(inv.issued_at), f"{inv.doc_id}: due before issued")
    for pay in episode.payments:
        check(pay.amount_minor != 0, f"{pay.doc_id}: zero amount")
        for ref in pay.invoice_refs:
            check(ref in inv_by_business, f"{pay.doc_id}: missing invoice ref {ref}")

    # Ledger references and balance
    for row in episode.ledger:
        check(row.currency == episode.base_currency, f"{row.row_id}: ledger currency")
        check(row.debit_minor >= 0 and row.credit_minor >= 0, f"{row.row_id}: negative side")
        check((row.debit_minor == 0) != (row.credit_minor == 0), f"{row.row_id}: exactly one side")
        check(row.invoice_id is None or row.invoice_id in inv_by_business, f"{row.row_id}: bad invoice ref")
        check(row.payment_id is None or row.payment_id in pay_by_business, f"{row.row_id}: bad payment ref")
    check(episode.is_balanced(), "ledger not balanced")

    # FX table
    for rate in episode.fx:
        check(rate.rate > 0, f"{rate.fx_id}: non-positive rate")
        check(rate.source_currency != rate.target_currency, f"{rate.fx_id}: same currency")
        check(rate.target_currency == episode.base_currency, f"{rate.fx_id}: target is not base")

    # Truth
    if check_truth:
        truth = episode.truth
        check(truth is not None, "missing truth")
        if truth:
            check(truth.get("episode_id") == episode.episode_id, "truth episode_id")
            check(truth.get("n_issues") == len(truth.get("issues", [])), "truth n_issues")
            check(truth.get("clean") == (len(truth.get("issues", [])) == 0), "truth clean flag")
            all_ids = episode.all_record_ids()
            seen_issue_ids = set()
            for issue in truth.get("issues", []):
                iid = issue.get("issue_id")
                check(iid not in seen_issue_ids, f"duplicate issue id {iid}")
                seen_issue_ids.add(iid)
                check(issue.get("type") in (TYPE_DUPLICATE, TYPE_ALLOCATION, TYPE_FX, TYPE_PERIOD), f"{iid}: bad type")
                check(int(issue.get("expected_amount_minor", 0)) > 0, f"{iid}: non-positive amount")
                check(issue.get("currency") == episode.base_currency, f"{iid}: wrong currency")
                check(len(issue.get("affected_ids", [])) > 0, f"{iid}: no affected ids")
                for raw in issue.get("affected_ids", []):
                    check(raw in all_ids, f"{iid}: affected id {raw} missing")
                check(1 <= len(issue.get("evidence_sets", [])) <= 3, f"{iid}: evidence set count")
                for evidence in issue.get("evidence_sets", []):
                    check(1 <= len(evidence) <= 10, f"{iid}: evidence size")
                    for raw in evidence:
                        check(raw in all_ids, f"{iid}: evidence id {raw} missing")
                # At least one complete evidence set must be non-empty.
                check(any(issue.get("evidence_sets", [])), f"{iid}: empty evidence")
                try:
                    if issue["type"] == TYPE_DUPLICATE:
                        _check_duplicate(episode, issue)
                    elif issue["type"] == TYPE_ALLOCATION:
                        _check_allocation(episode, issue)
                    elif issue["type"] == TYPE_FX:
                        _check_fx(episode, issue)
                    elif issue["type"] == TYPE_PERIOD:
                        _check_period(episode, issue)
                except AssertionError as exc:
                    errors.append(f"{iid}: {exc}")
            # Ambiguity: no two issues with the same type and affected set.
            seen = set()
            for issue in truth.get("issues", []):
                key = (issue["type"], tuple(sorted(issue["affected_ids"])))
                check(key not in seen, f"ambiguous duplicate truth issue {key}")
                seen.add(key)

    return errors


def assert_valid(episode: Episode, *, check_truth: bool = True) -> None:
    errors = validate_episode(episode, check_truth=check_truth)
    if errors:
        raise AssertionError("episode validation failed:\n- " + "\n- ".join(errors))
