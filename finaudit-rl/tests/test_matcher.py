import unittest

from finaudit.inject import ALL_TYPES
from finaudit.matcher import (
    MAX_EVIDENCE_IDS,
    MAX_EXPLANATION_CHARS,
    MAX_FINDINGS,
    normalize_id,
    oracle_submission,
    score_submission,
    terminal_reward,
    tolerance_for,
    validate_and_normalize,
)
from finaudit.models import Episode
from finaudit.generator import generate_episode


def episode_with_issues(seed: int = 123, n_issues: int = 2) -> Episode:
    return generate_episode(seed, {"n_issues": n_issues, "max_steps": 100})


def oracle_finding(ep: Episode, index: int = 0) -> dict:
    findings = oracle_submission(ep)["findings"]
    assert findings, "expected at least one oracle finding"
    return findings[index]


def issue_by_id(ep: Episode, issue_id: str) -> dict:
    return next(i for i in ep.truth["issues"] if i["issue_id"] == issue_id)


def issue_for_finding(ep: Episode, finding: dict) -> dict:
    expected = int(finding["claimed_amount_minor"])
    for issue in ep.truth["issues"]:
        if issue["type"] != finding["type"]:
            continue
        if set(issue["affected_ids"]) != set(finding["affected_ids"]):
            continue
        if abs(int(issue["expected_amount_minor"]) - expected) > tolerance_for(ep, expected):
            continue
        return issue
    raise AssertionError("no matching issue")


class RewardTests(unittest.TestCase):
    def test_terminal_reward_formula(self):
        cases = [
            (1.0, 0, 1.0),
            (1.0, 10, 0.98),
            (1.0, 50, 0.9),
            (0.5, 2, -0.004),
            (0.0, 3, -1.0),
            (-1.0, 5, -1.0),
        ]
        for f1, steps, expected in cases:
            with self.subTest(f1=f1, steps=steps):
                self.assertAlmostEqual(terminal_reward(f1, steps), expected, places=9)

    def test_tolerance_formula(self):
        ep = episode_with_issues()
        self.assertEqual(tolerance_for(ep, 1), 2)
        self.assertEqual(tolerance_for(ep, 1000), 2)
        self.assertEqual(tolerance_for(ep, 10000), 5)
        self.assertEqual(tolerance_for(ep, 100000), 50)

    def test_empty_empty_clean_f1(self):
        ep = generate_episode(500, {"n_issues": 0})
        result = score_submission(ep, {"episode_id": ep.episode_id, "findings": []}, inspection_steps=0)
        self.assertTrue(result.valid)
        self.assertEqual(result.f1, 1.0)
        self.assertEqual(result.reward, 1.0)

    def test_empty_erroneous_f1(self):
        ep = episode_with_issues(n_issues=3)
        result = score_submission(ep, {"episode_id": ep.episode_id, "findings": []}, inspection_steps=0)
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.reward, -1.0)

    def test_truth_with_empty_findings_f1(self):
        ep = episode_with_issues(n_issues=2)
        result = score_submission(ep, {"episode_id": ep.episode_id, "findings": []}, inspection_steps=0)
        self.assertEqual(result.f1, 0.0)

    def test_step_economy_slope(self):
        ep = episode_with_issues(n_issues=2)
        answer = oracle_submission(ep)
        r0 = score_submission(ep, answer, inspection_steps=0).reward
        r40 = score_submission(ep, answer, inspection_steps=40).reward
        self.assertAlmostEqual(r0 - r40, 0.08, places=9)


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.ep = episode_with_issues(n_issues=2)
        self.issue_ids = [i["issue_id"] for i in self.ep.truth["issues"]]

    def result(self, findings, inspection_steps=0):
        return score_submission(self.ep, {"episode_id": self.ep.episode_id, "findings": findings}, inspection_steps=inspection_steps)

    def test_oracle_is_perfect(self):
        result = self.result(oracle_submission(self.ep)["findings"])
        self.assertTrue(result.valid)
        self.assertEqual(result.tp, 2)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, 0)
        self.assertEqual(result.f1, 1.0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.reward, 1.0)

    def test_oracle_reward_respects_inspection_steps(self):
        result = self.result(oracle_submission(self.ep)["findings"], inspection_steps=4)
        self.assertEqual(result.reward, 1.0 - 0.002 * 4)

    def test_type_mismatch_is_fp(self):
        finding = oracle_finding(self.ep)
        other_type = next(t for t in ALL_TYPES if t != finding["type"])
        finding["type"] = other_type
        result = self.result([finding])
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 2)

    def test_currency_mismatch_is_fp(self):
        finding = oracle_finding(self.ep)
        other_currency = "EUR" if finding["currency"] == "USD" else "USD"
        finding["currency"] = other_currency
        result = self.result([finding])
        self.assertEqual(result.tp, 0)

    def test_affected_ids_must_match_exactly(self):
        finding = oracle_finding(self.ep)
        finding["affected_ids"] = list(reversed(finding["affected_ids"]))
        self.assertEqual(self.result([finding]).tp, 1)  # set comparison, order-insensitive
        finding["affected_ids"] = finding["affected_ids"][:-1]
        self.assertEqual(self.result([finding]).tp, 0)
        finding["affected_ids"] = finding["affected_ids"] + ["L999999"]
        self.assertEqual(self.result([finding]).tp, 0)

    def test_amount_tolerance_window(self):
        finding = oracle_finding(self.ep)
        issue = issue_for_finding(self.ep, finding)
        expected = int(issue["expected_amount_minor"])
        tolerance = tolerance_for(self.ep, expected)
        finding["claimed_amount_minor"] = expected + tolerance
        self.assertEqual(self.result([finding]).tp, 1)
        finding["claimed_amount_minor"] = expected + tolerance + 1
        self.assertEqual(self.result([finding]).tp, 0)

    def test_evidence_requires_complete_allowed_set(self):
        finding = oracle_finding(self.ep)
        issue = issue_for_finding(self.ep, finding)
        allowed = issue["evidence_sets"][0]

        # Unknown evidence is never accepted.
        broken = dict(finding)
        broken["evidence_ids"] = ["L999999"]
        self.assertEqual(self.result([broken]).tp, 0)

        # Incomplete evidence is never accepted.
        incomplete = [x for x in finding["evidence_ids"] if x in allowed][:-1]
        broken = dict(finding)
        broken["evidence_ids"] = incomplete
        self.assertEqual(self.result([broken]).tp, 0)

        # Extra existing evidence is allowed.
        others = [x for x in self.ep.all_record_ids() if x not in finding["evidence_ids"]]
        extra = dict(finding)
        extra["evidence_ids"] = finding["evidence_ids"] + others[:2]
        self.assertEqual(self.result([extra]).tp, 1)

    def test_evidence_subset_from_later_allowed_set_accepted(self):
        # If there are two allowed sets, citing the second set alone is valid.
        finding = oracle_finding(self.ep)
        issue = issue_for_finding(self.ep, finding)
        if len(issue["evidence_sets"]) > 1:
            second = issue["evidence_sets"][1]
            if set(second) <= set(self.ep.all_record_ids()):
                finding["evidence_ids"] = sorted(set(second))
                self.assertEqual(self.result([finding]).tp, 1)

    def test_business_id_aliases_accepted(self):
        finding = oracle_finding(self.ep)
        business = {}
        for inv in self.ep.invoices:
            business[inv.doc_id] = inv.invoice_id
        for pay in self.ep.payments:
            business[pay.doc_id] = pay.payment_id
        remapped_affected = [business.get(x, x) for x in finding["affected_ids"]]
        remapped_evidence = [business.get(x, x) for x in finding["evidence_ids"]]
        alias = dict(finding)
        alias["affected_ids"] = remapped_affected
        alias["evidence_ids"] = remapped_evidence
        if business:
            self.assertEqual(self.result([alias]).tp, 1)

    def test_duplicates_are_false_positives(self):
        finding = oracle_finding(self.ep)
        duplicated = [finding, dict(finding)]
        result = self.result(duplicated)
        self.assertEqual(result.tp, 1)
        self.assertEqual(result.fp, 1)
        # One of the two truth issues remains unfound.
        self.assertEqual(result.fn, 1)
        self.assertAlmostEqual(result.f1, 0.5, places=9)
        self.assertAlmostEqual(result.reward, 0.0, places=9)

    def test_one_to_one_matching(self):
        findings = [oracle_finding(self.ep, 0), oracle_finding(self.ep, 1)]
        # Add a third finding that duplicates the first; it must not earn a TP.
        findings.append(dict(findings[0]))
        result = self.result(findings)
        self.assertEqual(result.tp, 2)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.fn, 0)

    def test_mark_all_on_clean_is_bad(self):
        ep = generate_episode(777, {"n_issues": 0})
        findings = []
        for row in ep.ledger[:20]:
            findings.append(
                {
                    "type": "duplicate_account",
                    "affected_ids": [row.row_id],
                    "claimed_amount_minor": 1,
                    "currency": ep.base_currency,
                    "evidence_ids": [row.row_id],
                    "explanation": "mark all",
                }
            )
        result = score_submission(ep, {"episode_id": ep.episode_id, "findings": findings}, inspection_steps=0)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, len(findings))
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.reward, -1.0)

    def test_matching_is_deterministic(self):
        findings = [oracle_finding(self.ep, 0), oracle_finding(self.ep, 1)]
        findings_swapped = [findings[1], findings[0]]
        r1 = self.result(findings)
        r2 = self.result(findings_swapped)
        self.assertEqual((r1.tp, r1.fp, r1.fn), (r2.tp, r2.fp, r2.fn))
        self.assertEqual(sorted(r1.issue_status), sorted(r2.issue_status))
        r3 = self.result([findings[0], findings[1], dict(findings[0])])
        self.assertEqual((r3.tp, r3.fp, r3.fn), (2, 1, 0))


class StructureValidationTests(unittest.TestCase):
    def setUp(self):
        self.ep = episode_with_issues(n_issues=2)
        self.finding = oracle_finding(self.ep)

    def assert_invalid(self, payload):
        findings, error = validate_and_normalize(self.ep, payload)
        self.assertIsNone(findings)
        self.assertTrue(error)
        result = score_submission(self.ep, payload, inspection_steps=0)
        self.assertFalse(result.valid)
        self.assertEqual(result.reward, -1.0)

    def assert_valid(self, payload):
        findings, error = validate_and_normalize(self.ep, payload)
        self.assertIsNotNone(findings)
        self.assertIsNone(error)

    def base_payload(self, findings):
        return {"episode_id": self.ep.episode_id, "findings": findings}

    def test_bad_payload_shapes(self):
        for payload in (
            None,
            [],
            "x",
            {},
            {"episode_id": "wrong", "findings": []},
            {"episode_id": None, "findings": []},
            {"findings": []},
            {"episode_id": self.ep.episode_id, "findings": "no"},
            {"episode_id": self.ep.episode_id, "findings": {0: 1}},
        ):
            with self.subTest(payload=payload):
                self.assert_invalid(payload)

    def test_finding_shape_limits(self):
        too_many_findings = [dict(self.finding) for _ in range(MAX_FINDINGS + 1)]
        self.assert_invalid(self.base_payload(too_many_findings))

        def with_field(field, value):
            f = dict(self.finding)
            f[field] = value
            return f

        for field, bad in (
            ("type", ""),
            ("type", 123),
            ("affected_ids", "L1"),
            ("affected_ids", [1]),
            ("claimed_amount_minor", "1"),
            ("claimed_amount_minor", 1.5),
            ("claimed_amount_minor", -1),
            ("currency", 123),
            ("evidence_ids", "L1"),
            ("evidence_ids", ["L1"] * (MAX_EVIDENCE_IDS + 1)),
            ("explanation", 123),
            ("explanation", "x" * (MAX_EXPLANATION_CHARS + 1)),
        ):
            with self.subTest(field=field, value=bad):
                self.assert_invalid(self.base_payload([with_field(field, bad)]))

        self.assert_invalid(self.base_payload([None]))
        self.assert_invalid(self.base_payload([["duplicate_account"]]))

    def test_normalize_id_handles_unknown_and_business(self):
        row = self.ep.ledger[0]
        self.assertEqual(normalize_id(self.ep, row.row_id), row.row_id)
        invoice = self.ep.invoices[0]
        self.assertEqual(normalize_id(self.ep, invoice.invoice_id), invoice.doc_id)
        # Unknown IDs pass through.
        self.assertEqual(normalize_id(self.ep, "unknown-id"), "unknown-id")

    def test_invalid_structure_does_not_inherit_f1(self):
        result = score_submission(self.ep, {"episode_id": "nope", "findings": []}, inspection_steps=0)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 0)
        self.assertEqual(result.fn, len(self.ep.truth["issues"]))
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.reward, -1.0)


if __name__ == "__main__":
    unittest.main()
