"""Clean baseline generation: a coherent hidden company story.

The builder creates invoices, payments, FX rates and a balanced double-entry
ledger in the reporting (base) currency.  Errors are injected later by
``finaudit.inject``; this module deliberately contains no errors.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable
from zoneinfo import ZoneInfo

from . import profiles
from .models import (
    Episode,
    Invoice,
    LedgerRow,
    Payment,
    make_manifest,
    make_policy,
)
from .money import (
    CURRENCY_MINOR_UNITS,
    FxRate,
    FxTable,
    USD_PER_MAJOR,
    convert_minor,
    format_rate,
    iso,
    local_date,
    minor_units,
    parse_iso_datetime,
    round_half_up,
)
from .rng import Rng
from .version import GENERATOR_VERSION

EPS = Decimal("0.00000001")


# ---------------------------------------------------------------------------
# date helpers


def _period_bounds(period_start: str, period_end: str) -> tuple[datetime, datetime]:
    return parse_iso_datetime(period_start), parse_iso_datetime(period_end)


def _period_of(dt: datetime, start: datetime, end: datetime) -> int:
    if dt < start:
        return -1
    if dt > end:
        return 1
    return 0


def _at(day: date, hour: int, minute: int, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time(hour, minute, 0), tzinfo=tz)


def _rand_day(rng: Rng, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, max(0, (end - start).days)))


def _clamp_same_period(
    ref: datetime,
    lag_days: int,
    start: datetime,
    end: datetime,
) -> datetime:
    """Posting timestamp with the same period as ``ref`` (clean baseline)."""
    candidate = ref + timedelta(days=lag_days)
    p = _period_of(ref, start, end)
    if p == 0:
        if candidate > end:
            candidate = end - timedelta(hours=1)
        if candidate < start:
            candidate = start + timedelta(hours=1)
        return candidate
    if p < 0:
        if candidate >= start:
            candidate = start - timedelta(hours=1)
        return candidate
    if candidate <= end:
        candidate = end + timedelta(hours=1)
    return candidate


def _random_business_dt(rng: Rng, start: date, end: date, tz: ZoneInfo) -> datetime:
    day = _rand_day(rng, start, end)
    return _at(day, rng.randint(8, 18), rng.randint(0, 59), tz)


def _add_days(dt: datetime, days: int) -> datetime:
    return dt + timedelta(days=days)


# ---------------------------------------------------------------------------
# internal structures


@dataclass
class Entry:
    entry_id: str
    rows: list[LedgerRow]
    source_doc: str | None
    kind: str
    invoice_id: str | None = None
    payment_id: str | None = None

    def total_debit(self) -> int:
        return sum(int(r.debit_minor) for r in self.rows)

    def total_credit(self) -> int:
        return sum(int(r.credit_minor) for r in self.rows)

    def is_balanced(self) -> bool:
        return self.total_debit() == self.total_credit()


@dataclass
class Baseline:
    episode: Episode
    entries: list[Entry]
    invoice_remaining_foreign: dict[str, int]
    invoice_remaining_base: dict[str, int]
    invoice_entries: dict[str, Entry]
    payment_entries: dict[str, Entry]
    metadata: dict = field(default_factory=dict)
    _next_row: int = 1
    _next_entry: int = 1

    def new_row_id(self) -> str:
        value = f"L{self._next_row:06d}"
        self._next_row += 1
        return value

    def new_entry_id(self) -> str:
        value = f"E{self._next_entry:05d}"
        self._next_entry += 1
        return value

    def rebuild_ledger(self) -> None:
        self.episode.ledger = [row for entry in self.entries for row in entry.rows]


# ---------------------------------------------------------------------------
# builder


class Builder:
    def __init__(self, seed: int, config: dict):
        self.seed = int(seed)
        self.config = dict(config)
        self.rng = Rng(seed)
        self.rng_company = self.rng.fork("company")
        self.rng_fx = self.rng.fork("fx")
        self.rng_invoices = self.rng.fork("invoices")
        self.rng_payments = self.rng.fork("payments")
        self.rng_extras = self.rng.fork("extras")

        self.company = self.rng_company.choice(profiles.COMPANIES)
        self.period_start_date, self.period_end_date, self.quarter = self.rng_company.choice(profiles.PERIODS)
        self.timezone_name = self.config.get("timezone") or self.rng_company.choice(profiles.TIMEZONES)
        self.tz = ZoneInfo(self.timezone_name)
        self.base_currency = self.config.get("base_currency") or self.rng_company.choice(profiles.BASE_CURRENCIES)
        self.foreign_currencies = [
            c for c in profiles.FOREIGN_CURRENCIES[self.base_currency] if c != self.base_currency
        ]

        start_day = date.fromisoformat(self.period_start_date)
        end_day = date.fromisoformat(self.period_end_date)
        self.period_start_dt = datetime.combine(start_day, time(0, 0, 0), tzinfo=self.tz)
        self.period_end_dt = datetime.combine(end_day, time(23, 59, 59), tzinfo=self.tz)
        self.start_day = start_day
        self.end_day = end_day

        self.entity_ids = [e[0] for e in profiles.ENTITIES[: self.rng_company.randint(1, 3)]]
        self.counterparties = list(profiles.COUNTERPARTIES)
        self.rng_company.shuffle(self.counterparties)

        self.fx_rates: list[FxRate] = []
        self.fx_table: FxTable | None = None
        self.invoices: list[Invoice] = []
        self.payments: list[Payment] = []
        self.entries: list[Entry] = []
        self.invoice_remaining_foreign: dict[str, int] = {}
        self.invoice_remaining_base: dict[str, int] = {}
        self.invoice_entries: dict[str, Entry] = {}
        self.payment_entries: dict[str, Entry] = {}
        self.invoice_by_doc: dict[str, Invoice] = {}
        self.payment_by_doc: dict[str, Payment] = {}
        self.margin_invoices: list[str] = []
        self.cross_currency_payments: list[str] = []
        self.overpayment_payments: list[str] = []

        self.doc_prefix = self.rng_company.hex(2).upper()
        self.invoice_seq = 0
        self.payment_seq = 0
        self.entry_seq = 0

    def _new_entry_id(self) -> str:
        self.entry_seq += 1
        return f"E{self.entry_seq:05d}"

    # ---- ids --------------------------------------------------------------
    def _new_invoice_ids(self) -> tuple[str, str]:
        self.invoice_seq += 1
        return (
            f"DOC-I-{self.doc_prefix}-{self.invoice_seq:04d}",
            f"INV-{self.doc_prefix}-{self.invoice_seq:04d}",
        )

    def _new_payment_ids(self) -> tuple[str, str]:
        self.payment_seq += 1
        return (
            f"DOC-P-{self.doc_prefix}-{self.payment_seq:04d}",
            f"PAY-{self.doc_prefix}-{self.payment_seq:04d}",
        )

    # ---- FX ---------------------------------------------------------------
    def _build_fx(self) -> None:
        assert self.fx_table is None
        start = self.start_day - timedelta(days=60)
        end = self.end_day + timedelta(days=60)
        base_usd = USD_PER_MAJOR[self.base_currency]
        for foreign in self.foreign_currencies:
            base_rate = USD_PER_MAJOR[foreign] / base_usd
            factor = Decimal("1")
            day = start
            while day <= end:
                # Small mean-reverting random walk, deterministic per day.
                shock = Decimal(str(self.rng_fx.uniform(-0.0035, 0.0035)))
                factor = factor * (Decimal("1") + shock)
                factor = factor * Decimal("0.995") + Decimal("1") * Decimal("0.005")
                factor = min(Decimal("1.06"), max(Decimal("0.94"), factor))
                rate = (base_rate * factor).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
                if rate <= 0:
                    rate = Decimal("0.00000001")
                self.fx_rates.append(FxRate(day, foreign, self.base_currency, rate))
                day += timedelta(days=1)
        self.fx_rates.sort(key=lambda r: (r.date, r.source_currency, r.target_currency))
        self.fx_table = FxTable(self.fx_rates, self.base_currency, fallback="nearest_prior")

    def _rate(self, source: str, target: str, day: date) -> Decimal:
        assert self.fx_table is not None
        return self.fx_table.rate(source, target, day)

    # ---- invoice amounts --------------------------------------------------
    def _invoice_total_range(self, currency: str) -> tuple[int, int]:
        units = minor_units(currency)
        if currency in ("USD", "EUR", "GBP"):
            lo, hi = 500, 28_000
        elif currency == "AED":
            lo, hi = 1_500, 80_000
        elif currency == "KZT":
            lo, hi = 120_000, 4_000_000
        elif currency == "JPY":
            lo, hi = 60_000, 2_000_000
        else:
            lo, hi = 1_000, 50_000
        return lo * (10**units), hi * (10**units)

    def _make_line_items(self, rng: Rng, kind: str, currency: str, total_minor: int) -> list[dict]:
        descriptions = profiles.LINE_ITEMS[kind]
        n = rng.randint(1, min(4, len(descriptions)))
        chosen = rng.sample(descriptions, n)
        # Split the total into n positive integer amounts.
        if n == 1:
            parts = [total_minor]
        else:
            cuts = sorted(rng.sample(range(1, total_minor), n - 1)) if total_minor > n else []
            if len(cuts) < n - 1:
                parts = [total_minor] + [0] * (n - 1)
            else:
                prev = 0
                parts = []
                for cut in cuts:
                    parts.append(cut - prev)
                    prev = cut
                parts.append(total_minor - prev)
        items = []
        for idx, (desc, amount) in enumerate(zip(chosen, parts), start=1):
            quantity = rng.randint(1, 6)
            # Keep amount == quantity * unit_price exactly.
            unit = max(1, amount // quantity)
            amount = unit * quantity
            items.append(
                {
                    "line_id": f"LI-{idx:02d}",
                    "description": desc,
                    "quantity": quantity,
                    "unit_price_minor": int(unit),
                    "amount_minor": int(amount),
                }
            )
        # Any rounding drift goes into the last line.
        diff = total_minor - sum(i["amount_minor"] for i in items)
        if diff:
            items[-1]["amount_minor"] += diff
            items[-1]["unit_price_minor"] = max(1, items[-1]["amount_minor"] // items[-1]["quantity"])
            items[-1]["amount_minor"] = items[-1]["unit_price_minor"] * items[-1]["quantity"]
            diff2 = total_minor - sum(i["amount_minor"] for i in items)
            if diff2:
                items[-1]["amount_minor"] += diff2
                items[-1]["quantity"] = 1
                items[-1]["unit_price_minor"] = items[-1]["amount_minor"]
        return items

    # ---- invoices ---------------------------------------------------------
    def _build_invoices(self) -> None:
        rng = self.rng_invoices
        n_invoices = int(self.config.get("n_invoices") or rng.randint(45, 75))
        n_invoices = max(20, min(200, n_invoices))
        n_margin = min(int(self.config.get("margin_invoices", 2)), max(0, n_invoices // 10))
        margin_indices = set(rng.sample(range(n_invoices), n_margin)) if n_margin else set()

        for idx in range(n_invoices):
            is_margin = idx in margin_indices
            kind = "sale" if rng.random() < 0.60 else "purchase"
            entity = rng.choice(self.entity_ids)
            if rng.random() < 0.55:
                currency = self.base_currency
            else:
                currency = rng.choice(self.foreign_currencies)
            lo, hi = self._invoice_total_range(currency)
            total = rng.randint(lo, hi)
            # Round to a plausible invoice total.
            if currency == "JPY":
                total = max(1000, total // 100 * 100)
            else:
                total = max(100, total // 100 * 100)

            if is_margin:
                if rng.random() < 0.5:
                    issued_day = self.start_day - timedelta(days=rng.randint(1, 12))
                else:
                    issued_day = self.end_day + timedelta(days=rng.randint(1, 12))
            else:
                issued_day = _rand_day(rng, self.start_day, self.end_day)
            issued_dt = _at(issued_day, rng.randint(8, 18), rng.randint(0, 59), self.tz)
            due_dt = _add_days(issued_dt, rng.randint(15, 60))
            doc_id, invoice_id = self._new_invoice_ids()
            line_items = self._make_line_items(rng, kind, currency, total)
            total = sum(int(i["amount_minor"]) for i in line_items)
            inv = Invoice(
                doc_id=doc_id,
                invoice_id=invoice_id,
                entity_id=entity,
                issued_at=iso(issued_dt),
                due_at=iso(due_dt),
                currency=currency,
                total_minor=int(total),
                line_items=line_items,
                counterparty_id=rng.choice(self.counterparties),
                kind=kind,
            )
            self.invoices.append(inv)
            self.invoice_by_doc[doc_id] = inv
            if is_margin:
                self.margin_invoices.append(doc_id)

            issued_date = issued_dt.date()
            base_total = convert_minor(inv.total_minor, currency, self.base_currency, self._rate(currency, self.base_currency, issued_date))
            if base_total <= 0:
                base_total = 1
            self.invoice_remaining_foreign[doc_id] = inv.total_minor
            self.invoice_remaining_base[doc_id] = base_total

            lag = rng.randint(0, 2)
            posting_dt = _clamp_same_period(issued_dt, lag, self.period_start_dt, self.period_end_dt)
            rows: list[LedgerRow] = []
            if kind == "sale":
                rows.append(self._row(entity, posting_dt, "1100", base_total, 0, invoice_id=invoice_id))
                rows.append(self._row(entity, posting_dt, "4000", 0, base_total, invoice_id=invoice_id))
            else:
                rows.append(self._row(entity, posting_dt, "5000", base_total, 0, invoice_id=invoice_id))
                rows.append(self._row(entity, posting_dt, "2000", 0, base_total, invoice_id=invoice_id))
            entry = Entry(
                entry_id=self._new_entry_id(),
                rows=rows,
                source_doc=doc_id,
                kind="invoice_accrual",
                invoice_id=invoice_id,
            )
            self.entries.append(entry)
            self.invoice_entries[doc_id] = entry

    def _row(
        self,
        entity_id: str,
        dt: datetime,
        account: str,
        debit: int,
        credit: int,
        *,
        invoice_id: str | None = None,
        payment_id: str | None = None,
    ) -> LedgerRow:
        if debit < 0 or credit < 0:
            raise ValueError("negative ledger side")
        return LedgerRow(
            row_id=f"__pending_{len(self.entries)}_{len(self.entries)}",
            timestamp=iso(dt),
            account=account,
            debit_minor=int(debit),
            credit_minor=int(credit),
            currency=self.base_currency,
            entity_id=entity_id,
            invoice_id=invoice_id,
            payment_id=payment_id,
        )

    def _assign_row_ids(self) -> None:
        # Assign ids in entry order, then shuffle file order later.
        counter = 1
        for entry in self.entries:
            for row in entry.rows:
                row.row_id = f"L{counter:06d}"
                counter += 1
        self._next_row = counter

    # ---- payment helpers --------------------------------------------------
    def _payment_dt(self, inv: Invoice, rng: Rng, *, after_period: bool = False) -> datetime:
        issued = parse_iso_datetime(inv.issued_at)
        due = parse_iso_datetime(inv.due_at)
        low = issued + timedelta(days=1)
        high = due + timedelta(days=10)
        if high <= low:
            high = low + timedelta(days=1)
        span = max(1, (high.date() - low.date()).days)
        day = low.date() + timedelta(days=rng.randint(0, span))
        dt = _at(day, rng.randint(9, 17), rng.randint(0, 59), self.tz)
        if after_period and dt <= self.period_end_dt:
            dt = self.period_end_dt + timedelta(days=rng.randint(1, 10), hours=rng.randint(0, 8))
        return dt

    def _posting_dt(self, payment_dt: datetime, rng: Rng) -> datetime:
        return _clamp_same_period(payment_dt, rng.randint(0, 1), self.period_start_dt, self.period_end_dt)

    def _make_payment(
        self,
        inv: Invoice,
        rng: Rng,
        amount_minor: int,
        payment_currency: str,
        payment_dt: datetime,
        *,
        kind: str = "customer_receipt",
        refs: list[str] | None = None,
    ) -> Payment:
        doc_id, payment_id = self._new_payment_ids()
        pay = Payment(
            doc_id=doc_id,
            payment_id=payment_id,
            entity_id=inv.entity_id,
            timestamp=iso(payment_dt),
            currency=payment_currency,
            amount_minor=int(amount_minor),
            invoice_refs=refs if refs is not None else [inv.invoice_id],
            counterparty_id=inv.counterparty_id,
            kind=kind,
        )
        self.payments.append(pay)
        self.payment_by_doc[doc_id] = pay
        return pay

    def _settle_one(
        self,
        inv: Invoice,
        pay: Payment,
        settled_foreign: int,
        rng: Rng,
        *,
        overpay_foreign: int = 0,
        wrong_invoice: Invoice | None = None,
    ) -> Entry:
        """Create a balanced settlement entry for one invoice."""
        doc = inv.doc_id
        remaining_foreign = self.invoice_remaining_foreign[doc]
        remaining_base = self.invoice_remaining_base[doc]
        settled_foreign = min(int(settled_foreign), remaining_foreign)
        if settled_foreign <= 0:
            raise ValueError("settled amount must be positive")
        if settled_foreign == remaining_foreign:
            relief = remaining_base
        else:
            relief = round_half_up(Decimal(remaining_base) * Decimal(settled_foreign) / Decimal(remaining_foreign))
            relief = max(1, min(remaining_base - 1, relief))
        self.invoice_remaining_foreign[doc] = remaining_foreign - settled_foreign
        self.invoice_remaining_base[doc] = remaining_base - relief

        pay_date = local_date(pay.timestamp)
        cash_base = convert_minor(
            pay.amount_minor,
            pay.currency,
            self.base_currency,
            self._rate(pay.currency, self.base_currency, pay_date),
        )
        credit_base = 0
        if overpay_foreign:
            credit_base = convert_minor(
                overpay_foreign,
                inv.currency,
                self.base_currency,
                self._rate(inv.currency, self.base_currency, pay_date),
            )
        if cash_base < 0:
            raise ValueError("negative payment cash base")

        posting_dt = self._posting_dt(parse_iso_datetime(pay.timestamp), rng)
        rows: list[LedgerRow] = []
        target_invoice_id = wrong_invoice.invoice_id if wrong_invoice else inv.invoice_id
        is_sale = inv.kind == "sale"
        # Cash side
        if is_sale:
            rows.append(self._row(inv.entity_id, posting_dt, "1000", cash_base, 0, payment_id=pay.payment_id))
            rows.append(self._row(inv.entity_id, posting_dt, "1100", 0, relief, invoice_id=target_invoice_id, payment_id=pay.payment_id))
        else:
            rows.append(self._row(inv.entity_id, posting_dt, "2000", relief, 0, invoice_id=target_invoice_id, payment_id=pay.payment_id))
            rows.append(self._row(inv.entity_id, posting_dt, "1000", 0, cash_base, payment_id=pay.payment_id))
        # Customer credit / prepayment for overpayment
        if credit_base:
            if is_sale:
                rows.append(self._row(inv.entity_id, posting_dt, "2200", 0, credit_base, payment_id=pay.payment_id))
            else:
                rows.append(self._row(inv.entity_id, posting_dt, "1200", credit_base, 0, payment_id=pay.payment_id))
        # FX plug
        fx_diff = cash_base - relief - credit_base
        if fx_diff:
            if is_sale:
                if fx_diff > 0:
                    rows.append(self._row(inv.entity_id, posting_dt, "5100", 0, fx_diff, payment_id=pay.payment_id))
                else:
                    rows.append(self._row(inv.entity_id, posting_dt, "5100", -fx_diff, 0, payment_id=pay.payment_id))
            else:
                if fx_diff > 0:
                    rows.append(self._row(inv.entity_id, posting_dt, "5100", fx_diff, 0, payment_id=pay.payment_id))
                else:
                    rows.append(self._row(inv.entity_id, posting_dt, "5100", 0, -fx_diff, payment_id=pay.payment_id))
        entry = Entry(
            entry_id=self._new_entry_id(),
            rows=rows,
            source_doc=pay.doc_id,
            kind="payment_settlement",
            invoice_id=inv.invoice_id,
            payment_id=pay.payment_id,
        )
        self.entries.append(entry)
        self.payment_entries[pay.doc_id] = entry
        if not entry.is_balanced():
            raise ValueError(f"unbalanced settlement {entry.entry_id}")
        return entry

    def _settle_multi_same_currency(self, invoices: list[Invoice], pay: Payment, rng: Rng) -> Entry:
        """Settle several same-currency invoices with one payment."""
        # Allocate the payment in invoice order.  The payment amount was chosen
        # as the exact sum of remaining amounts, so it settles all of them.
        rows: list[LedgerRow] = []
        pay_date = local_date(pay.timestamp)
        cash_base = convert_minor(pay.amount_minor, pay.currency, self.base_currency, self._rate(pay.currency, self.base_currency, pay_date))
        posting_dt = self._posting_dt(parse_iso_datetime(pay.timestamp), rng)
        total_relief = 0
        is_sale = invoices[0].kind == "sale"
        for inv in invoices:
            doc = inv.doc_id
            rem_f = self.invoice_remaining_foreign[doc]
            rem_b = self.invoice_remaining_base[doc]
            if rem_f <= 0:
                continue
            self.invoice_remaining_foreign[doc] = 0
            self.invoice_remaining_base[doc] = 0
            total_relief += rem_b
            if is_sale:
                rows.append(self._row(inv.entity_id, posting_dt, "1100", 0, rem_b, invoice_id=inv.invoice_id, payment_id=pay.payment_id))
            else:
                rows.append(self._row(inv.entity_id, posting_dt, "2000", rem_b, 0, invoice_id=inv.invoice_id, payment_id=pay.payment_id))
        if is_sale:
            rows.insert(0, self._row(invoices[0].entity_id, posting_dt, "1000", cash_base, 0, payment_id=pay.payment_id))
        else:
            rows.insert(0, self._row(invoices[0].entity_id, posting_dt, "1000", 0, cash_base, payment_id=pay.payment_id))
        fx_diff = cash_base - total_relief
        if fx_diff:
            if is_sale:
                if fx_diff > 0:
                    rows.append(self._row(invoices[0].entity_id, posting_dt, "5100", 0, fx_diff, payment_id=pay.payment_id))
                else:
                    rows.append(self._row(invoices[0].entity_id, posting_dt, "5100", -fx_diff, 0, payment_id=pay.payment_id))
            else:
                if fx_diff > 0:
                    rows.append(self._row(invoices[0].entity_id, posting_dt, "5100", fx_diff, 0, payment_id=pay.payment_id))
                else:
                    rows.append(self._row(invoices[0].entity_id, posting_dt, "5100", 0, -fx_diff, payment_id=pay.payment_id))
        entry = Entry(
            entry_id=self._new_entry_id(),
            rows=rows,
            source_doc=pay.doc_id,
            kind="payment_settlement",
            invoice_id=invoices[0].invoice_id if len(invoices) == 1 else None,
            payment_id=pay.payment_id,
        )
        self.entries.append(entry)
        self.payment_entries[pay.doc_id] = entry
        if not entry.is_balanced():
            raise ValueError(f"unbalanced multi settlement {entry.entry_id}")
        return entry

    def _make_refund(self, inv: Invoice, rng: Rng, amount_base: int) -> None:
        if amount_base <= 0:
            return
        doc_id, payment_id = self._new_payment_ids()
        # Refunds are deliberately base-currency only: a clean distractor.
        pay_dt = self.period_end_dt - timedelta(days=rng.randint(1, 30), hours=rng.randint(0, 8))
        if pay_dt < self.period_start_dt:
            pay_dt = self.period_start_dt + timedelta(days=1)
        pay = Payment(
            doc_id=doc_id,
            payment_id=payment_id,
            entity_id=inv.entity_id,
            timestamp=iso(pay_dt),
            currency=self.base_currency,
            amount_minor=-amount_base,
            invoice_refs=[inv.invoice_id],
            counterparty_id=inv.counterparty_id,
            kind="refund",
        )
        self.payments.append(pay)
        self.payment_by_doc[doc_id] = pay
        posting_dt = self._posting_dt(pay_dt, rng)
        rows = [
            self._row(inv.entity_id, posting_dt, "2200", amount_base, 0, payment_id=pay.payment_id),
            self._row(inv.entity_id, posting_dt, "1000", 0, amount_base, payment_id=pay.payment_id),
        ]
        entry = Entry(
            entry_id=self._new_entry_id(),
            rows=rows,
            source_doc=pay.doc_id,
            kind="refund",
            payment_id=pay.payment_id,
        )
        self.entries.append(entry)
        self.payment_entries[pay.doc_id] = entry

    # ---- payments ---------------------------------------------------------
    def _build_payments(self) -> None:
        rng = self.rng_payments
        order = list(self.invoices)
        rng.shuffle(order)
        for inv in order:
            doc = inv.doc_id
            rem = self.invoice_remaining_foreign.get(doc, 0)
            if rem <= 0:
                continue
            # Margin invoices are paid in their own (adjacent) period or later.
            is_margin = doc in self.margin_invoices
            roll = rng.random()
            if is_margin:
                # Always pay margin invoices so they look like ordinary context.
                strategy = "full"
            elif roll < 0.55:
                strategy = "full"
            elif roll < 0.70:
                strategy = "split"
            elif roll < 0.80:
                strategy = "partial"
            elif roll < 0.88:
                strategy = "cross_partial"
            elif roll < 0.95:
                strategy = "overpay"
            else:
                strategy = "unpaid"

            if strategy == "unpaid":
                continue
            if strategy == "full":
                pay_dt = self._payment_dt(inv, rng, after_period=is_margin and parse_iso_datetime(inv.issued_at) > self.period_end_dt)
                pay = self._make_payment(inv, rng, rem, inv.currency, pay_dt, kind=("customer_receipt" if inv.kind == "sale" else "vendor_payment"))
                self._settle_one(inv, pay, rem, rng)
            elif strategy == "split":
                parts = rng.randint(2, 3)
                remaining = rem
                for part_idx in range(parts):
                    if remaining <= 0:
                        break
                    if part_idx == parts - 1:
                        amount = remaining
                    else:
                        fraction = rng.uniform(0.20, 0.45)
                        amount = max(1, min(remaining - 1, int(round(remaining * fraction))))
                    pay_dt = self._payment_dt(inv, rng, after_period=is_margin and parse_iso_datetime(inv.issued_at) > self.period_end_dt)
                    pay = self._make_payment(inv, rng, amount, inv.currency, pay_dt, kind=("customer_receipt" if inv.kind == "sale" else "vendor_payment"))
                    self._settle_one(inv, pay, amount, rng)
                    remaining -= amount
            elif strategy == "partial":
                fraction = rng.uniform(0.25, 0.75)
                amount = max(1, min(rem - 1, int(round(rem * fraction))))
                pay_dt = self._payment_dt(inv, rng, after_period=is_margin and parse_iso_datetime(inv.issued_at) > self.period_end_dt)
                pay = self._make_payment(inv, rng, amount, inv.currency, pay_dt, kind=("customer_receipt" if inv.kind == "sale" else "vendor_payment"))
                self._settle_one(inv, pay, amount, rng)
            elif strategy == "cross_partial":
                foreign = rng.choice([c for c in self.foreign_currencies if c != inv.currency])
                fraction = rng.uniform(0.30, 0.65)
                target_foreign = max(1, min(rem - 1, int(round(rem * fraction))))
                pay_dt = self._payment_dt(inv, rng, after_period=is_margin and parse_iso_datetime(inv.issued_at) > self.period_end_dt)
                pay_date = pay_dt.date()
                cross = self._rate(inv.currency, foreign, pay_date)
                amount = convert_minor(target_foreign, inv.currency, foreign, cross)
                if amount <= 0:
                    amount = 1
                # Convert back; rounding can shift by a minor unit.
                settled = convert_minor(amount, foreign, inv.currency, self._rate(foreign, inv.currency, pay_date))
                if settled >= rem:
                    amount = max(1, amount - 1)
                    settled = convert_minor(amount, foreign, inv.currency, self._rate(foreign, inv.currency, pay_date))
                if settled <= 0:
                    amount = max(1, amount + 1)
                    settled = convert_minor(amount, foreign, inv.currency, self._rate(foreign, inv.currency, pay_date))
                if settled <= 0 or settled >= rem:
                    continue
                pay = self._make_payment(inv, rng, amount, foreign, pay_dt, kind=("customer_receipt" if inv.kind == "sale" else "vendor_payment"))
                self._settle_one(inv, pay, settled, rng)
                self.cross_currency_payments.append(pay.doc_id)
            elif strategy == "overpay":
                extra = max(1, int(round(rem * rng.uniform(0.05, 0.15))))
                pay_dt = self._payment_dt(inv, rng, after_period=is_margin and parse_iso_datetime(inv.issued_at) > self.period_end_dt)
                pay = self._make_payment(inv, rng, rem + extra, inv.currency, pay_dt, kind=("customer_receipt" if inv.kind == "sale" else "vendor_payment"))
                self._settle_one(inv, pay, rem, rng, overpay_foreign=extra)
                self.overpayment_payments.append(pay.doc_id)
                if inv.kind == "sale" and inv.currency == self.base_currency and rng.random() < 0.5:
                    self._make_refund(inv, rng, max(1, extra // 2))

    # ---- other context entries -------------------------------------------
    def _build_context_entries(self) -> None:
        rng = self.rng_extras
        # Opening balance, before the period.  Not a source document and not
        # subject to the invoice/payment recognition rule.
        opening_dt = self.period_start_dt - timedelta(hours=2)
        amount = rng.randint(2_000_000, 12_000_000)
        rows = [
            self._row(self.entity_ids[0], opening_dt, "1000", amount, 0),
            self._row(self.entity_ids[0], opening_dt, "3000", 0, amount),
        ]
        self.entries.append(Entry(self._new_entry_id(), rows, None, "opening_balance"))

        # Monthly operating entries spread through the period.
        month_starts: list[date] = []
        cursor = self.start_day.replace(day=1)
        while cursor <= self.end_day:
            month_starts.append(max(cursor, self.start_day))
            if cursor.month == 12:
                cursor = date(cursor.year + 1, 1, 1)
            else:
                cursor = date(cursor.year, cursor.month + 1, 1)
        for m in month_starts:
            for label, lo, hi in (
                ("rent", 200_000, 900_000),
                ("payroll", 1_500_000, 6_000_000),
                ("bank_fees", 10_000, 80_000),
            ):
                day = min(self.end_day, m + timedelta(days=rng.randint(0, 5)))
                dt = _at(day, rng.randint(9, 17), rng.randint(0, 59), self.tz)
                value = rng.randint(lo, hi)
                rows = [
                    self._row(self.entity_ids[0], dt, "5000", value, 0),
                    self._row(self.entity_ids[0], dt, "1000", 0, value),
                ]
                self.entries.append(Entry(self._new_entry_id(), rows, None, f"operating_{label}"))

        # Ensure the suggested 200-row floor without creating audit targets.
        while sum(len(e.rows) for e in self.entries) < 200:
            day = _rand_day(rng, self.start_day, self.end_day)
            dt = _at(day, rng.randint(9, 17), rng.randint(0, 59), self.tz)
            value = rng.randint(5_000, 250_000)
            rows = [
                self._row(self.entity_ids[0], dt, "5000", value, 0),
                self._row(self.entity_ids[0], dt, "1000", 0, value),
            ]
            self.entries.append(Entry(self._new_entry_id(), rows, None, "operating_other"))

    # ---- finalize ---------------------------------------------------------
    def build(self) -> Baseline:
        self._build_fx()
        self._build_invoices()
        self._build_payments()
        self._build_context_entries()
        self._assign_row_ids()

        ledger = [row for entry in self.entries for row in entry.rows]
        # Shuffle file order, not ids.
        file_rng = self.rng.fork("file_order")
        file_rng.shuffle(ledger)
        invoices = list(self.invoices)
        payments = list(self.payments)
        file_rng.shuffle(invoices)
        file_rng.shuffle(payments)

        config_public = {
            "difficulty": self.config.get("difficulty", "medium"),
            "max_steps": int(self.config.get("max_steps", 100)),
        }
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        seed_commitment = hashlib.sha256(
            f"{self.seed}|{config_hash}|{GENERATOR_VERSION}".encode("utf-8")
        ).hexdigest()
        episode_id = f"ep-{seed_commitment[:16]}"
        manifest = make_manifest(
            episode_id=episode_id,
            base_currency=self.base_currency,
            period_start=iso(self.period_start_dt),
            period_end=iso(self.period_end_dt),
            timezone=self.timezone_name,
            n_records=len(ledger),
            n_invoices=len(invoices),
            n_payments=len(payments),
            seed_commitment=seed_commitment,
            difficulty=config_public["difficulty"],
            config=config_public,
        )
        policy = make_policy(
            base_currency=self.base_currency,
            period_start=iso(self.period_start_dt),
            period_end=iso(self.period_end_dt),
            timezone=self.timezone_name,
            accounts=profiles.ACCOUNTS,
        )
        episode = Episode(
            manifest=manifest,
            policy=policy,
            ledger=ledger,
            invoices=invoices,
            payments=payments,
            fx=list(self.fx_rates),
            truth={
                "schema_version": "1.0",
                "episode_id": episode_id,
                "base_currency": self.base_currency,
                "period_start": iso(self.period_start_dt),
                "period_end": iso(self.period_end_dt),
                "n_issues": 0,
                "clean": True,
                "issues": [],
            },
            provenance={
                "schema_version": "1.0",
                "generator_version": GENERATOR_VERSION,
                "episode_id": episode_id,
                "seed": self.seed,
                "config": self.config,
                "company": {
                    "name": self.company,
                    "entities": self.entity_ids,
                    "base_currency": self.base_currency,
                    "timezone": self.timezone_name,
                    "period": self.quarter,
                },
                "baseline": {
                    "n_ledger_rows": len(ledger),
                    "n_invoices": len(invoices),
                    "n_payments": len(payments),
                    "n_entries": len(self.entries),
                    "balanced": sum(r.debit_minor for r in ledger) == sum(r.credit_minor for r in ledger),
                },
                "modifications": [],
                "validation": {},
            },
        )
        baseline = Baseline(
            episode=episode,
            entries=self.entries,
            invoice_remaining_foreign=self.invoice_remaining_foreign,
            invoice_remaining_base=self.invoice_remaining_base,
            invoice_entries=self.invoice_entries,
            payment_entries=self.payment_entries,
            metadata={
                "margin_invoices": self.margin_invoices,
                "cross_currency_payments": self.cross_currency_payments,
                "overpayment_payments": self.overpayment_payments,
                "invoice_by_doc": self.invoice_by_doc,
                "payment_by_doc": self.payment_by_doc,
                "doc_prefix": self.doc_prefix,
                "company": self.company,
                "entities": self.entity_ids,
                "period_start_dt": self.period_start_dt,
                "period_end_dt": self.period_end_dt,
                "tz": self.tz,
            },
            _next_row=self._next_row,
            _next_entry=self.entry_seq + 1,
        )
        # Verify the clean baseline before returning it.
        if not episode.is_balanced():
            raise AssertionError("clean baseline ledger is not balanced")
        for entry in self.entries:
            if not entry.is_balanced():
                raise AssertionError(f"unbalanced baseline entry {entry.entry_id}")
        return baseline
