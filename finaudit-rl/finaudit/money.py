"""Money, currency and FX helpers.

All monetary values in the environment are integers in the currency's minor
unit (e.g. cents).  FX rates are decimal strings.  Conversion is explicit and
uses round-half-up to the target currency's minor unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, getcontext
from typing import Iterable

getcontext().prec = 34

# currency -> number of decimal places in the major unit
CURRENCY_MINOR_UNITS: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "KZT": 2,
    "AED": 2,
    "JPY": 0,
}

# Approximate USD value of one major unit, used only to generate plausible
# synthetic rates.  It is never used for grading.
USD_PER_MAJOR: dict[str, Decimal] = {
    "USD": Decimal("1"),
    "EUR": Decimal("1.08"),
    "GBP": Decimal("1.27"),
    "KZT": Decimal("0.00220"),
    "AED": Decimal("0.2723"),
    "JPY": Decimal("0.00670"),
}

ZERO = Decimal("0")


def minor_units(currency: str) -> int:
    try:
        return CURRENCY_MINOR_UNITS[currency]
    except KeyError as exc:
        raise ValueError(f"unknown currency {currency!r}") from exc


def as_decimal(value: str | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal value {value!r}") from exc


def round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def major_to_minor(amount_major: Decimal, currency: str) -> int:
    return round_half_up(amount_major * (Decimal(10) ** minor_units(currency)))


def minor_to_major(amount_minor: int, currency: str) -> Decimal:
    return Decimal(amount_minor) / (Decimal(10) ** minor_units(currency))


def convert_minor(amount_minor: int, source: str, target: str, rate: Decimal) -> int:
    """Convert an integer minor amount at ``rate`` (source major -> target major)."""
    if source == target:
        return int(amount_minor)
    source_major = minor_to_major(int(amount_minor), source)
    target_major = source_major * as_decimal(rate)
    return major_to_minor(target_major, target)


def format_rate(rate: Decimal) -> str:
    """Canonical 8-decimal FX string (no scientific notation)."""
    q = as_decimal(rate).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    return format(q, "f")


def parse_iso_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return dt


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.isoformat(timespec="seconds")


def local_date(value: str | datetime) -> date:
    dt = parse_iso_datetime(value) if isinstance(value, str) else value
    return dt.date()


@dataclass(frozen=True)
class FxRate:
    date: date
    source_currency: str
    target_currency: str
    rate: Decimal

    @property
    def fx_id(self) -> str:
        return f"fx:{self.date.isoformat()}:{self.source_currency}:{self.target_currency}"

    def to_row(self) -> dict:
        return {
            "fx_id": self.fx_id,
            "date": self.date.isoformat(),
            "source_currency": self.source_currency,
            "target_currency": self.target_currency,
            "rate": format_rate(self.rate),
        }


class FxTable:
    """Deterministic FX table with explicit cross-rate support.

    Only foreign->base rates are stored.  Cross rates are derived exactly as
    ``rate(src->base) / rate(dst->base)``.  Missing dates can optionally fall
    back to the nearest earlier business date, as declared by the policy.
    """

    def __init__(self, rates: Iterable[FxRate], base_currency: str, fallback: str = "nearest_prior"):
        self.base_currency = base_currency
        self.fallback = fallback
        self._by_pair: dict[tuple[str, str], dict[date, Decimal]] = {}
        for r in rates:
            self._by_pair.setdefault((r.source_currency, r.target_currency), {})[r.date] = r.rate
        self._dates: dict[tuple[str, str], list[date]] = {
            key: sorted(v) for key, v in self._by_pair.items()
        }

    def has_pair(self, source: str, target: str) -> bool:
        return (source, target) in self._by_pair or (
            source == self.base_currency or target == self.base_currency
        )

    def _direct(self, source: str, target: str, day: date) -> Decimal | None:
        table = self._by_pair.get((source, target))
        if table is None:
            return None
        if day in table:
            return table[day]
        if self.fallback == "nearest_prior":
            prior = [d for d in self._dates[(source, target)] if d <= day]
            if prior:
                return table[prior[-1]]
        if self.fallback == "nearest":
            all_dates = self._dates[(source, target)]
            if all_dates:
                nearest = min(all_dates, key=lambda d: (abs((d - day).days), d))
                return table[nearest]
        return None

    def rate(self, source: str, target: str, day: date) -> Decimal:
        if source == target:
            return Decimal(1)
        direct = self._direct(source, target, day)
        if direct is not None:
            return direct
        # Derive through the base currency.
        if source != self.base_currency and target == self.base_currency:
            raise KeyError(f"no FX rate {source}->{target} on {day}")
        if source == self.base_currency:
            inverse = self._direct(target, source, day)
            if inverse is None or inverse == 0:
                raise KeyError(f"no FX rate {source}->{target} on {day}")
            return Decimal(1) / inverse
        if target == self.base_currency:
            return self._direct(source, target, day)  # type: ignore[return-value]
        src_base = self._direct(source, self.base_currency, day)
        dst_base = self._direct(target, self.base_currency, day)
        if src_base is None or dst_base is None or dst_base == 0:
            raise KeyError(f"no cross rate {source}->{target} on {day}")
        return src_base / dst_base

    def fx_id(self, source: str, target: str, day: date) -> str:
        return f"fx:{day.isoformat()}:{source}:{target}"

    def available_dates(self, source: str, target: str) -> list[date]:
        """Dates for which a direct or inverse rate is stored."""
        if (source, target) in self._dates:
            return list(self._dates[(source, target)])
        if source == self.base_currency and (target, source) in self._dates:
            return list(self._dates[(target, source)])
        if target == self.base_currency and (source, target) in self._dates:
            return list(self._dates[(source, target)])
        # Cross rate: dates present in both legs.
        left = self._dates.get((source, self.base_currency), [])
        right = self._dates.get((target, self.base_currency), [])
        if left and right:
            return sorted(set(left) & set(right))
        return []

    def row(self, source: str, target: str, day: date) -> dict:
        return {
            "fx_id": self.fx_id(source, target, day),
            "date": day.isoformat(),
            "source_currency": source,
            "target_currency": target,
            "rate": format_rate(self.rate(source, target, day)),
        }


def previous_day(day: date, n: int = 1) -> date:
    return day - timedelta(days=n)


def next_day(day: date, n: int = 1) -> date:
    return day + timedelta(days=n)
