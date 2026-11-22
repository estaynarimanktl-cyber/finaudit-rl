import unittest
from decimal import Decimal

from finaudit.env import AuditEnv, PolicyEnv
from finaudit.generator import generate_episode
from finaudit.models import Episode
from finaudit.testing import episode_fingerprint


def make_episode(n_issues: int = 2, seed: int = 123) -> Episode:
    return generate_episode(seed, {"n_issues": n_issues, "max_steps": 100})


class AuditEnvTests(unittest.TestCase):
    def setUp(self):
        self.episode = make_episode()
        self.env = AuditEnv(episode=self.episode)
        self.obs, self.info = self.env.reset()

    def step(self, action, env=None):
        return (env or self.env).step(action)

    def test_reset_fixed_episode_reused(self):
        self.assertEqual(self.obs["episode_id"], self.episode.episode_id)
        obs2, _ = self.env.reset()
        self.assertEqual(obs2["episode_id"], self.episode.episode_id)
        self.assertEqual(self.env.episode, self.episode)

    def test_reset_seed_regenerates(self):
        env = AuditEnv(seed=9, config={"n_issues": 2})
        obs, _ = env.reset()
        expected = generate_episode(9, {"n_issues": 2})
        self.assertEqual(obs["episode_id"], expected.episode_id)
        self.assertEqual(episode_fingerprint(env.episode), episode_fingerprint(expected))
        obs2, _ = env.reset(seed=10)
        self.assertEqual(obs2["episode_id"], generate_episode(10, {"n_issues": 2}).episode_id)

    def test_reset_requires_call(self):
        env = AuditEnv()
        with self.assertRaises(RuntimeError):
            env.step({"type": "list_records", "table": "ledger"})

    def test_observation_shape(self):
        obs = self.obs
        for key in ("episode_id", "step", "budget_remaining", "task", "manifest", "policy", "action_schema", "submission_schema", "error", "result"):
            self.assertIn(key, obs)
        self.assertNotIn("truth", obs)
        self.assertNotIn("provenance", obs)
        self.assertEqual(obs["step"], 0)
        self.assertEqual(obs["manifest"]["episode_id"], self.episode.episode_id)
        self.assertIsNone(obs["result"])

    def test_list_records_pagination(self):
        rows = []
        cursor = None
        env = AuditEnv(episode=self.episode)
        env.reset()
        while True:
            obs, _, term, trunc, _ = env.step({"type": "list_records", "table": "ledger", "cursor": cursor, "limit": 50})
            self.assertFalse(term or trunc)
            self.assertLessEqual(len(obs["records"]), 50)
            rows.extend(obs["records"])
            cursor = obs["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(len(rows), len(self.episode.ledger))
        self.assertEqual({r["row_id"] for r in rows}, {r.row_id for r in self.episode.ledger})

    def test_list_records_limit_capped(self):
        obs, _, _, _, _ = self.env.step({"type": "list_records", "table": "ledger", "limit": 1000})
        self.assertEqual(obs["limit"], 50)
        self.assertLessEqual(len(obs["records"]), 50)

    def test_list_records_validates_limit(self):
        for bad in (0, -1, 51, "50", True, 1.5):
            with self.subTest(bad=bad):
                obs, _, _, _, _ = self.env.step({"type": "list_records", "table": "ledger", "limit": bad})
                self.assertIn("error", obs)

    def test_list_records_table_names(self):
        for table in ("ledger", "invoices", "payments", "fx", "policy"):
            obs, _, _, _, _ = self.env.step({"type": "list_records", "table": table, "limit": 1})
            self.assertEqual(obs["action"], "list_records")
            self.assertEqual(obs["table"], table)

    def test_list_records_stale_or_foreign_cursor(self):
        cursor = self.env._encode_cursor("ledger", 10)
        self.assertEqual(self.env._decode_cursor(cursor, "ledger"), 10)
        with self.assertRaises(ValueError):
            self.env._decode_cursor(cursor, "invoices")
        with self.assertRaises(ValueError):
            self.env._decode_cursor(cursor + "x", "ledger")
        with self.assertRaises(ValueError):
            self.env._decode_cursor("!!!not-a-cursor!!!", "ledger")

    def test_get_record_canonical_and_alias(self):
        row = self.episode.ledger[0]
        obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "ledger", "id": row.row_id})
        self.assertEqual(obs["record"]["row_id"], row.row_id)

        inv = self.episode.invoices[0]
        obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "invoices", "id": inv.invoice_id})
        self.assertEqual(obs["record"]["doc_id"], inv.doc_id)
        obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "invoices", "id": inv.doc_id})
        self.assertEqual(obs["record"]["doc_id"], inv.doc_id)

        pay = self.episode.payments[0]
        obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "payments", "id": pay.payment_id})
        self.assertEqual(obs["record"]["doc_id"], pay.doc_id)

        if self.episode.fx:
            rate = self.episode.fx[0]
            obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "fx", "id": rate.fx_id})
            self.assertEqual(obs["record"]["fx_id"], rate.fx_id)

        obs, _, _, _, _ = self.env.step({"type": "get_record", "table": "policy", "id": "policy"})
        self.assertEqual(obs["record"], self.episode.policy)

    def test_get_record_unknown_consumes_step(self):
        before = self.env.step_count
        obs, reward, _, _, _ = self.env.step({"type": "get_record", "table": "ledger", "id": "L000001"})
        self.assertIn("error", obs)
        self.assertEqual(reward, 0.0)
        self.assertEqual(self.env.step_count, before + 1)

    def test_get_document(self):
        inv = self.episode.invoices[0]
        obs, _, _, _, _ = self.env.step({"type": "get_document", "doc_id": inv.doc_id})
        self.assertEqual(obs["doc_type"], "invoice")
        self.assertEqual(obs["document"]["doc_id"], inv.doc_id)
        pay = self.episode.payments[0]
        obs, _, _, _, _ = self.env.step({"type": "get_document", "doc_id": pay.doc_id})
        self.assertEqual(obs["doc_type"], "payment")
        obs, _, _, _, _ = self.env.step({"type": "get_document", "doc_id": "nope"})
        self.assertIn("error", obs)

    def test_calculator(self):
        for expression, expected in (("1+2", "3"), ("(125000 * 0.0375)", "4687.5000"), ("-3*4", "-12"), ("10/4", "2.5")):
            obs, _, _, _, _ = self.env.step({"type": "calculator", "expression": expression})
            self.assertEqual(obs["result_value"], expected, expression)

    def test_calculator_rejects_code_and_files(self):
        bad = [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "10 ** 10 ** 10 ** 10",
            "1//0",
            "a=1",
            "",
            "x" * 600,
        ]
        for expression in bad:
            with self.subTest(expression=expression):
                obs, _, _, _, _ = self.env.step({"type": "calculator", "expression": expression})
                self.assertIn("error", obs)

    def test_unknown_action_consumes_step(self):
        obs, reward, _, _, _ = self.env.step({"type": "definitely_not_an_action"})
        self.assertEqual(obs["error"], "invalid action: ValueError")
        self.assertEqual(reward, 0.0)
        self.assertEqual(self.env.step_count, 1)
        obs, reward, _, _, _ = self.env.step("nope")
        self.assertEqual(obs["error"], "invalid action: ValueError")

    def test_submit_terminates(self):
        payload = {"episode_id": self.episode.episode_id, "findings": []}
        obs, _, term, trunc, _ = self.env.step({"type": "submit", "findings": payload})
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertTrue(self.env.done)
        obs2, reward2, term2, trunc2, _ = self.env.step({"type": "list_records", "table": "ledger"})
        self.assertEqual(obs2["error"], "episode already finished")
        self.assertEqual(reward2, 0.0)

    def test_invalid_structure_returns_minus_one(self):
        for payload in (
            {"episode_id": "wrong", "findings": []},
            {"episode_id": self.episode.episode_id, "findings": "no"},
            {"episode_id": self.episode.episode_id, "findings": [1]},
            {"episode_id": self.episode.episode_id},
        ):
            env = AuditEnv(episode=self.episode)
            env.reset()
            _, _, term, _, _ = env.step({"type": "submit", "findings": payload})
            result = env.grade()
            self.assertFalse(result.valid)
            self.assertEqual(result.reward, -1.0)

    def test_step_budget_truncates(self):
        env = AuditEnv(episode=self.episode, max_steps=2)
        env.reset()
        obs, _, _, _, _ = env.step({"type": "list_records", "table": "ledger"})
        obs, _, _, _, _ = env.step({"type": "list_records", "table": "ledger"})
        self.assertTrue(env.done)
        self.assertTrue(env.truncated)
        self.assertFalse(env.terminated)
        result = env.grade()
        self.assertTrue(result.valid)
        self.assertEqual(result.f1, 0.0)
        # Empty truth with findings is F1=0; two inspection steps are beyond
        # the lower clip bound, so the reward is -1.0.
        self.assertAlmostEqual(result.reward, -1.0, places=6)

    def test_time_limit_truncates(self):
        env = AuditEnv(episode=self.episode, time_limit_s=0.0)
        env.reset()
        obs, _, term, trunc, _ = env.step({"type": "list_records", "table": "ledger"})
        self.assertTrue(env.done)
        self.assertTrue(env.truncated)

    def test_train_reward_visible_eval_hidden(self):
        payload = {"episode_id": self.episode.episode_id, "findings": []}
        env = AuditEnv(episode=self.episode, mode="eval")
        env.reset()
        obs, reward, _, _, info = env.step({"type": "submit", "findings": payload})
        self.assertEqual(reward, 0.0)
        self.assertIsNone(obs["result"])
        self.assertNotIn("result", info)
        self.assertNotIn("budget_remaining", info)
        grade = env.grade()
        self.assertEqual(grade.reward, -1.0)  # erroneous episode, empty findings
        # A clean episode: reward 1.0 in train mode, hidden in eval.
        clean = generate_episode(500, {"n_issues": 0})
        train_env = AuditEnv(episode=clean, mode="train")
        train_env.reset()
        _, reward_train, _, _, _ = train_env.step({"type": "submit", "findings": {"episode_id": clean.episode_id, "findings": []}})
        self.assertEqual(reward_train, 1.0)

    def test_policy_env_isolation(self):
        env = AuditEnv(episode=self.episode)
        policy = PolicyEnv(env)
        policy.reset()
        for attr in ("episode", "truth", "provenance", "seed", "grade", "expose_reward", "mode", "max_steps", "time_limit_s", "evaluate_empty"):
            with self.assertRaises(AttributeError):
                getattr(policy, attr)

    def test_policy_env_reset_seed_generation(self):
        env = AuditEnv(seed=33, config={"n_issues": 1})
        policy = PolicyEnv(env)
        policy.reset(seed=33)
        expected = generate_episode(33, {"n_issues": 1})
        obs, _ = env.reset()
        # PolicyEnv exposes the same reset contract as AuditEnv without truth.
        self.assertEqual(obs["episode_id"], expected.episode_id)

    def test_result_is_graded_only_in_train_mode(self):
        payload = {"episode_id": self.episode.episode_id, "findings": []}
        train = AuditEnv(episode=self.episode, mode="train")
        train.reset()
        obs, reward, _, _, info = train.step({"type": "submit", "findings": payload})
        self.assertIsNotNone(obs["result"])
        self.assertIn("result", info)

    def test_malformed_actions_do_not_crash_and_consume_steps(self):
        actions = [
            None,
            [],
            {},
            {"type": 5},
            {"type": "list_records", "table": None},
            {"type": "get_record", "table": "ledger", "id": None},
            {"type": "get_document", "doc_id": None},
            {"type": "calculator", "expression": None},
            {"type": "submit", "findings": None},
        ]
        for action in actions:
            env = AuditEnv(episode=self.episode)
            env.reset()
            obs, reward, _, _, _ = env.step(action)
            self.assertIn("error", obs)
            self.assertEqual(reward, 0.0)
            self.assertEqual(env.step_count, 1)

    def test_reward_formula_examples(self):
        from finaudit.matcher import terminal_reward

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
                self.assertAlmostEqual(terminal_reward(f1, steps), expected)

        clean = generate_episode(500, {"n_issues": 0})
        env = AuditEnv(episode=clean)
        env.reset()
        _, reward, _, _, _ = env.step({"type": "submit", "findings": {"episode_id": clean.episode_id, "findings": []}})
        self.assertAlmostEqual(reward, 1.0)

    def test_calculator_decimal_precision(self):
        obs, _, _, _, _ = self.env.step({"type": "calculator", "expression": "0.1 + 0.2"})
        self.assertEqual(obs["result_value"], "0.3")
        self.assertEqual(Decimal(obs["result_value"]), Decimal("0.3"))


if __name__ == "__main__":
    unittest.main()
