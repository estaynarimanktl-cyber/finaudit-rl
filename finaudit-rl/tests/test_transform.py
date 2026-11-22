import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from finaudit.env import AuditEnv
from finaudit.generator import generate_episode
from finaudit.models import Episode
from finaudit.trajectory import run_named_policy
from finaudit.transform import remap_episode
from finaudit.testing import assert_valid, episode_fingerprint, open_fingerprint
from finaudit.validate import validate_episode


class RemapEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episode = generate_episode(1234, {"n_issues": 5})
        cls.remapped = remap_episode(cls.episode, seed=17)
        cls.remapped_again = remap_episode(cls.episode, seed=17)

    def test_remapped_episode_valid(self):
        self.assertEqual(validate_episode(self.remapped), [])
        self.assertTrue(self.remapped.is_balanced())
        self.assertTrue(self.remapped.truth)

    def test_same_seed_is_deterministic(self):
        self.assertEqual(episode_fingerprint(self.remapped), episode_fingerprint(self.remapped_again))

    def test_identifiers_changed(self):
        self.assertNotEqual(self.episode.episode_id, self.remapped.episode_id)
        self.assertNotEqual(self.episode.manifest["episode_id"], self.remapped.manifest["episode_id"])
        old_ids = {r.row_id for r in self.episode.ledger}
        new_ids = {r.row_id for r in self.remapped.ledger}
        self.assertFalse(old_ids & new_ids)
        self.assertEqual(len(new_ids), len(old_ids))
        self.assertNotEqual(
            {i.invoice_id for i in self.episode.invoices},
            {i.invoice_id for i in self.remapped.invoices},
        )

    def test_economic_content_is_isomorphic(self):
        old = self.episode
        new = self.remapped
        self.assertEqual(len(old.ledger), len(new.ledger))
        self.assertEqual(len(old.invoices), len(new.invoices))
        self.assertEqual(len(old.payments), len(new.payments))
        self.assertEqual(
            Counter((r.account, r.debit_minor, r.credit_minor, r.entity_id) for r in old.ledger),
            Counter((r.account, r.debit_minor, r.credit_minor, r.entity_id) for r in new.ledger),
        )
        self.assertEqual(old.is_balanced(), new.is_balanced())
        self.assertEqual({i.total_minor for i in old.invoices}, {i.total_minor for i in new.invoices})
        self.assertEqual({p.amount_minor for p in old.payments}, {p.amount_minor for p in new.payments})
        self.assertEqual(
            {f.rate for f in old.fx},
            {f.rate for f in new.fx},
        )

    def test_truth_is_isomorphic(self):
        old = self.episode
        new = self.remapped
        self.assertEqual(len(old.truth["issues"]), len(new.truth["issues"]))
        old_types = sorted(i["type"] for i in old.truth["issues"])
        new_types = sorted(i["type"] for i in new.truth["issues"])
        self.assertEqual(old_types, new_types)
        self.assertEqual([i["expected_amount_minor"] for i in old.truth["issues"]], [i["expected_amount_minor"] for i in new.truth["issues"]])
        ids = new.all_record_ids()
        for issue in new.truth["issues"]:
            self.assertTrue(set(issue["affected_ids"]) <= ids)
            for evidence in issue["evidence_sets"]:
                self.assertTrue(set(evidence) <= ids)
                self.assertLessEqual(len(evidence), 10)

    def test_oracle_and_reconciliation_metrics_unchanged(self):
        for name in ("oracle", "reconciliation"):
            with self.subTest(policy=name):
                a = run_named_policy(self.episode, name)
                b = run_named_policy(self.remapped, name)
                self.assertEqual(a["graded_reward"], b["graded_reward"])
                self.assertEqual((a["grade"]["tp"], a["grade"]["fp"], a["grade"]["fn"]),
                                 (b["grade"]["tp"], b["grade"]["fp"], b["grade"]["fn"]))
                self.assertEqual(a["grade"]["f1"], b["grade"]["f1"])

    def test_remapped_episode_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.remapped.write(tmp)
            loaded = Episode.load(tmp)
            self.assertEqual(validate_episode(loaded), [])
            self.assertEqual(open_fingerprint(loaded), open_fingerprint(self.remapped))
            self.assertEqual(loaded.truth, self.remapped.truth)

    def test_truth_is_deep_copied(self):
        before = copy.deepcopy(self.episode.truth)
        remap_episode(self.episode, seed=99)
        self.assertEqual(self.episode.truth, before)


if __name__ == "__main__":
    unittest.main()
