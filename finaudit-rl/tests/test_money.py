import unittest
from datetime import date
from decimal import Decimal

from finaudit.money import (
    FxRate,
    FxTable,
    convert_minor,
    format_rate,
    major_to_minor,
    minor_to_major,
    parse_iso_datetime,
)


class MoneyTests(unittest.TestCase):
    def test_minor_major_roundtrip(self):
        self.assertEqual(major_to_minor(Decimal("12.34"), "USD"), 1234)
        self.assertEqual(minor_to_major(1234, "USD"), Decimal("12.34"))
        self.assertEqual(major_to_minor(Decimal("125"), "JPY"), 125)
        self.assertEqual(minor_to_major(125, "JPY"), Decimal("125"))

    def test_conversion_uses_target_minor_units(self):
        self.assertEqual(convert_minor(10000, "EUR", "USD", Decimal("1.10")), 11000)
        self.assertEqual(convert_minor(10000, "JPY", "USD", Decimal("0.0067")), 6700)
        self.assertEqual(convert_minor(10000, "USD", "USD", Decimal("9")), 10000)

    def test_format_rate_is_plain_decimal(self):
        self.assertEqual(format_rate(Decimal("1.2")), "1.20000000")
        self.assertEqual(format_rate(Decimal("0.000000001")), "0.00000000")

    def test_fx_table_nearest_prior_and_cross(self):
        rates = [
            FxRate(date(2026, 1, 1), "EUR", "USD", Decimal("1.10")),
            FxRate(date(2026, 1, 5), "EUR", "USD", Decimal("1.20")),
            FxRate(date(2026, 1, 1), "GBP", "USD", Decimal("1.30")),
            FxRate(date(2026, 1, 5), "GBP", "USD", Decimal("1.40")),
        ]
        table = FxTable(rates, "USD", fallback="nearest_prior")
        self.assertEqual(table.rate("EUR", "USD", date(2026, 1, 3)), Decimal("1.10"))
        self.assertEqual(table.rate("EUR", "USD", date(2026, 1, 5)), Decimal("1.20"))
        cross = table.rate("EUR", "GBP", date(2026, 1, 5))
        self.assertEqual(cross, Decimal("1.20") / Decimal("1.40"))
        self.assertEqual(table.rate("USD", "EUR", date(2026, 1, 5)), Decimal(1) / Decimal("1.20"))

    def test_fx_table_missing_prior_raises(self):
        table = FxTable([FxRate(date(2026, 1, 10), "EUR", "USD", Decimal("1.1"))], "USD")
        with self.assertRaises(KeyError):
            table.rate("EUR", "USD", date(2026, 1, 1))

    def test_parse_requires_offset(self):
        self.assertIsNotNone(parse_iso_datetime("2026-01-01T10:00:00+05:00"))
        with self.assertRaises(ValueError):
            parse_iso_datetime("2026-01-01T10:00:00")


if __name__ == "__main__":
    unittest.main()
