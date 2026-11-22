import copy
import unittest

from finaudit.generator import generate_episode
from decimal import Decimal

from finaudit.money import FxRate
from finaudit.models import Episode
from finaudit.validate import assert_valid, validate_episode


def make_episode(n_issues: int = 0, seed: int = 999) -> Episode:
    ep = generate_episode(seed, {"n_issues": n_issues, "max_steps": 100})
    assert_valid(ep)
    return ep


class OpenRecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.ep = make_episode()

    def assert_invalid(self, ep):
        self.assertTrue(validate_episode(ep), "expected validation errors")

    def test_valid_episode_passes(self):
        self.assertEqual(validate_episode(self.ep), [])

    def test_unbalanced_ledger_detected(self):
        ep = self.ep.clone()
        ep.ledger[0].debit_minor += 1
        self.assert_invalid(ep)

    def test_zero_sides_detected(self):
        ep = self.ep.clone()
        row = ep.ledger[0]
        row.debit_minor = 0
        row.credit_minor = 0
        self.assert_invalid(ep)

    def test_negative_amount_detected(self):
        ep = self.ep.clone()
        row = ep.ledger[0]
        row.debit_minor -= 1
        row.debit_minor = max(0, row.debit_minor)
        row.credit_minor -= 1  # now credit may be 0 or less
        self.assert_invalid(ep)

    def test_bad_ledger_currency_detected(self):
        ep = self.ep.clone()
        ep.ledger[0].currency = "USD" if ep.base_currency != "USD" else "EUR"
        self.assert_invalid(ep)

    def test_bad_invoice_reference_detected(self):
        ep = self.ep.clone()
        if ep.invoices and ep.invoices[0].doc_id:
            # Find a ledger row with an invoice reference and break it.
            row = next(r for r in ep.ledger if r.invoice_id is not None)
            row.invoice_id = "BAD-REF"
            self.assert_invalid(ep)

    def test_invoice_total_mismatch_detected(self):
        ep = self.ep.clone()
        ep.invoices[0].total_minor -= 1
        self.assert_invalid(ep)

    def test_payment_invoice_ref_detected(self):
        ep = self.ep.clone()
        pay = next((p for p in ep.payments if p.invoice_refs), None)
        if pay is not None:
            pay.invoice_refs = ["BAD-INVOICE"]
            self.assert_invalid(ep)

    def test_bad_fx_rate_detected(self):
        ep = self.ep.clone()
        if ep.fx:
            rate = ep.fx[0]
            ep.fx[0] = FxRate(rate.date, rate.source_currency, rate.target_currency, Decimal(0))
            self.assert_invalid(ep)

    def test_fx_same_currency_detected(self):
        ep = self.ep.clone()
        if ep.fx:
            rate = ep.fx[0]
            ep.fx[0] = FxRate(rate.date, rate.target_currency, rate.target_currency, rate.rate)
            self.assert_invalid(ep)

    def test_line_item_arithmetic_detected(self):
        ep = self.ep.clone()
        ep.invoices[0].line_items[0]["amount_minor"] += 1
        self.assert_invalid(ep)


class TruthValidationTests(unittest.TestCase):
    def setUp(self):
        self.ep = make_episode(n_issues=4)
        self.issues = self.ep.truth["issues"]
        self.all_ids = self.ep.all_record_ids()

    def assert_invalid(self, ep):
        self.assertTrue(validate_episode(ep), "expected validation errors")

    def test_non_positive_amount_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["expected_amount_minor"] = 0
        self.assert_invalid(ep)

    def test_bad_type_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["type"] = "made_up_type"
        self.assert_invalid(ep)

    def test_empty_affected_ids_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["affected_ids"] = []
        self.assert_invalid(ep)

    def test_missing_affected_id_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["affected_ids"] = ["L999999"]
        self.assert_invalid(ep)

    def test_bad_evidence_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["evidence_sets"][0][0] = "L999999"
        self.assert_invalid(ep)

    def test_empty_evidence_set_detected(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["evidence_sets"] = [[]]
        self.assert_invalid(ep)

    def test_duplicate_issue_ids_detected(self):
        ep = self.ep.clone()
        issue = copy.deepcopy(ep.truth["issues"][0])
        issue["issue_id"] = ep.truth["issues"][1]["issue_id"]
        ep.truth["issues"][0] = issue
        self.assert_invalid(ep)

    def test_ambiguous_issue_detected(self):
        ep = self.ep.clone()
        issue = copy.deepcopy(ep.truth["issues"][0])
        ep.truth["issues"].append(issue)
        self.assert_invalid(ep)

    def test_check_truth_can_be_disabled(self):
        ep = self.ep.clone()
        ep.truth["issues"][0]["expected_amount_minor"] = 0
        # Open records are still valid; disabling truth makes the episode pass.
        self.assertEqual(validate_episode(ep, check_truth=False), [])
        self.assertTrue(validate_episode(ep))


if __name__ == "__main__":
    unittest.main()
