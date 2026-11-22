import json
import tempfile
import unittest
from pathlib import Path

from finaudit.generator import generate_episode
from finaudit.matcher import tolerance_for
from finaudit.models import Episode
from finaudit.splits import ALL_SPECS, DEV_SPECS, TRAIN_SPECS
from finaudit.testing import assert_valid, episode_fingerprint, open_fingerprint
from finaudit.validate import validate_episode


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = {}
        for spec in ALL_SPECS:
            cls.episodes[spec.name] = generate_episode(spec.seed, spec.config)

    def test_all_named_specs_valid(self):
        for name, episode in self.episodes.items():
            with self.subTest(name=name):
                self.assertEqual(validate_episode(episode), [])

    def test_size_bounds(self):
        for name, episode in self.episodes.items():
            with self.subTest(name=name):
                self.assertGreaterEqual(len(episode.ledger), 200)
                self.assertLessEqual(len(episode.ledger), 2000)
                docs = len(episode.invoices) + len(episode.payments)
                self.assertGreaterEqual(docs, 20)
                self.assertLessEqual(docs, 200)
                self.assertEqual(episode.manifest["n_records"], len(episode.ledger))
                self.assertEqual(episode.manifest["n_documents"], docs)

    def test_ledger_balanced_and_base_currency(self):
        for name, episode in self.episodes.items():
            with self.subTest(name=name):
                self.assertTrue(episode.is_balanced())
                self.assertEqual(episode.total_debits(), episode.total_credits())
                self.assertTrue(all(r.currency == episode.base_currency for r in episode.ledger))

    def test_requested_issue_count(self):
        for spec in ALL_SPECS:
            episode = self.episodes[spec.name]
            with self.subTest(name=spec.name):
                self.assertEqual(len(episode.truth["issues"]), spec.config["n_issues"])
                self.assertEqual(episode.truth["n_issues"], spec.config["n_issues"])
                self.assertEqual(episode.truth["clean"], spec.config["n_issues"] == 0)

    def test_all_error_types_and_interactions(self):
        all_types = {
            issue["type"]
            for episode in self.episodes.values()
            for issue in episode.truth["issues"]
        }
        self.assertEqual(
            all_types,
            {"duplicate_account", "payment_allocation", "fx_conversion", "period_cutoff"},
        )
        train09 = {i["type"] for i in self.episodes["train-09"].truth["issues"]}
        self.assertEqual(train09, {"payment_allocation", "fx_conversion", "period_cutoff"})
        train10 = {i["type"] for i in self.episodes["train-10"].truth["issues"]}
        self.assertEqual(train10, {"duplicate_account", "fx_conversion"})
        self.assertEqual(len(self.episodes["dev-10"].truth["issues"]), 3)

    def test_clean_episodes_are_clean(self):
        for name in ("train-01", "dev-01"):
            episode = self.episodes[name]
            self.assertEqual(episode.truth["issues"], [])
            self.assertTrue(episode.truth["clean"])

    def test_truth_ids_exist_and_amounts_positive(self):
        for name, episode in self.episodes.items():
            ids = episode.all_record_ids()
            with self.subTest(name=name):
                for issue in episode.truth["issues"]:
                    self.assertGreater(issue["expected_amount_minor"], 0)
                    self.assertEqual(issue["currency"], episode.base_currency)
                    self.assertTrue(issue["affected_ids"])
                    self.assertTrue(set(issue["affected_ids"]) <= ids)
                    self.assertTrue(issue["evidence_sets"])
                    for evidence in issue["evidence_sets"]:
                        self.assertTrue(evidence)
                        self.assertLessEqual(len(evidence), 10)
                        self.assertTrue(set(evidence) <= ids)

    def test_no_two_issues_share_type_and_affected_set(self):
        for name, episode in self.episodes.items():
            seen = set()
            for issue in episode.truth["issues"]:
                key = (issue["type"], tuple(sorted(issue["affected_ids"])))
                self.assertNotIn(key, seen, name)
                seen.add(key)

    def test_deterministic_generation(self):
        a = generate_episode(4242, {"n_issues": 4})
        b = generate_episode(4242, {"n_issues": 4})
        c = generate_episode(4243, {"n_issues": 4})
        self.assertEqual(episode_fingerprint(a), episode_fingerprint(b))
        self.assertNotEqual(episode_fingerprint(a), episode_fingerprint(c))

    def test_config_changes_episode(self):
        a = generate_episode(77, {"n_issues": 2})
        b = generate_episode(77, {"n_issues": 2, "n_invoices": 70})
        self.assertNotEqual(episode_fingerprint(a), episode_fingerprint(b))

    def test_write_and_load_roundtrip(self):
        episode = self.episodes["train-05"]
        with tempfile.TemporaryDirectory() as tmp:
            episode.write(tmp)
            root = Path(tmp)
            for name in ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv"):
                self.assertTrue((root / "open" / name).is_file(), name)
            self.assertTrue((root / "private" / "truth.json").is_file())
            self.assertTrue((root / "private" / "provenance.json").is_file())
            loaded = Episode.load(tmp)
            self.assertEqual(open_fingerprint(episode), open_fingerprint(loaded))
            self.assertEqual(episode.truth, loaded.truth)
            public = Episode.load(tmp, include_private=False)
            self.assertIsNone(public.truth)
            self.assertIsNone(public.provenance)
            self.assertEqual(open_fingerprint(episode), open_fingerprint(public))

    def test_manifest_does_not_leak_truth(self):
        for name, episode in self.episodes.items():
            manifest = episode.manifest
            self.assertNotIn("n_issues", manifest.get("config", {}))
            self.assertNotIn("seed", manifest)
            self.assertNotIn("truth", manifest)
            self.assertNotIn("provenance", manifest)
            self.assertIn("seed", episode.provenance)
            self.assertEqual(len(manifest["seed_commitment"]), 64)
            # The public policy has no hidden issue list either.
            self.assertNotIn("issues", episode.policy)

    def test_amount_tolerance_rule(self):
        episode = self.episodes["train-03"]
        self.assertEqual(tolerance_for(episode, 1), 2)
        self.assertEqual(tolerance_for(episode, 4000), 2)
        self.assertEqual(tolerance_for(episode, 40000), 20)
        self.assertEqual(tolerance_for(episode, 1_000_000), 500)

    def test_line_item_arithmetic(self):
        for episode in self.episodes.values():
            for invoice in episode.invoices:
                self.assertEqual(
                    sum(int(item["amount_minor"]) for item in invoice.line_items),
                    invoice.total_minor,
                )
                for item in invoice.line_items:
                    self.assertEqual(
                        int(item["amount_minor"]),
                        int(item["quantity"]) * int(item["unit_price_minor"]),
                    )

    def test_fx_table_targets_base_and_covers_documents(self):
        for episode in self.episodes.values():
            self.assertTrue(episode.fx)
            for rate in episode.fx:
                self.assertEqual(rate.target_currency, episode.base_currency)
                self.assertNotEqual(rate.source_currency, rate.target_currency)
                self.assertGreater(rate.rate, 0)
            table = episode.fx_table()
            for invoice in episode.invoices:
                if invoice.currency != episode.base_currency:
                    table.rate(invoice.currency, episode.base_currency, invoice.issued_at and __import__("datetime").date.fromisoformat(invoice.issued_at[:10]))

    def test_split_lists_are_disjoint_and_complete(self):
        self.assertEqual(len(TRAIN_SPECS), 10)
        self.assertEqual(len(DEV_SPECS), 10)
        names = [s.name for s in ALL_SPECS]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(s.split == "train" for s in TRAIN_SPECS))
        self.assertTrue(all(s.split == "dev" for s in DEV_SPECS))

    def test_open_files_do_not_contain_truth_json(self):
        episode = self.episodes["dev-05"]
        with tempfile.TemporaryDirectory() as tmp:
            episode.write(tmp)
            open_dir = Path(tmp) / "open"
            for path in open_dir.iterdir():
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("evidence_sets", text)
                self.assertNotIn("issue_id", text)
            payload = json.loads((open_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertNotIn("issues", payload)


if __name__ == "__main__":
    unittest.main()
