import json
import unittest

from finaudit.env import AuditEnv, PolicyEnv
from finaudit.generator import generate_episode
from finaudit.policies import OraclePolicy, ReconciliationPolicy, SilencePolicy
from finaudit.trajectory import _action_summary, _obs_summary, run_policy, trajectory_jsonl


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.episode = generate_episode(321, {"n_issues": 2})
        self.env = AuditEnv(episode=self.episode)
        self.env.reset()
        self.policy = OraclePolicy(self.episode)

    def test_run_policy_returns_contract(self):
        result = run_policy(self.env, self.policy)
        self.assertEqual(result["policy"], "oracle")
        self.assertEqual(result["episode_id"], self.episode.episode_id)
        self.assertEqual(result["steps"], 1)
        self.assertTrue(result["terminated"])
        self.assertIsNotNone(result["grade"])
        self.assertEqual(len(result["trajectory"]), 1)
        step = result["trajectory"][0]
        self.assertEqual(step["action"]["type"], "submit")
        self.assertEqual(step["action"]["n_findings"], 2)

    def test_run_policy_records_zero_reward_exploration(self):
        env = AuditEnv(episode=self.episode)
        policy = ReconciliationPolicy()
        result = run_policy(env, policy)
        self.assertGreater(result["steps"], 1)
        self.assertTrue(all(step["reward"] == 0.0 for step in result["trajectory"][:-1]))

    def test_run_policy_compact_vs_full_observations(self):
        compact = run_policy(AuditEnv(episode=self.episode), SilencePolicy())
        full = run_policy(AuditEnv(episode=self.episode), SilencePolicy(), include_observations=True)
        self.assertIn("obs_digest", compact["trajectory"][0])
        self.assertNotIn("obs_digest", full["trajectory"][0])
        self.assertIn("episode_id", full["trajectory"][0]["observation"])

    def test_obs_summary_fields(self):
        obs = self.env.step({"type": "list_records", "table": "ledger", "limit": 5})[0]
        summary = _obs_summary(obs)
        self.assertEqual(summary["action"], "list_records")
        self.assertEqual(summary["table"], "ledger")
        self.assertLessEqual(summary["returned"], 5)
        self.assertIn("has_next", summary)
        obs = self.env.step({"type": "calculator", "expression": "2*3"})[0]
        self.assertEqual(_obs_summary(obs)["result_value"], "6")
        obs = self.env.step({"type": "submit", "findings": {"episode_id": self.episode.episode_id, "findings": []}})[0]
        self.assertEqual(_obs_summary(obs)["action"], "submit")

    def test_action_summary_fields(self):
        self.assertEqual(_action_summary({"type": "bogus"})["type"], "bogus")
        self.assertEqual(_action_summary(["x"])["type"], "list")
        self.assertEqual(_action_summary({"type": "get_record", "table": "ledger", "id": "L1"})["id"], "L1")
        self.assertEqual(_action_summary({"type": "calculator", "expression": "1+1"})["expression"], "1+1")

    def test_trajectory_jsonl_is_valid(self):
        result = run_policy(AuditEnv(episode=self.episode), SilencePolicy())
        text = trajectory_jsonl([result])
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["policy"], "silence")

    def test_run_policy_via_policy_env(self):
        env = AuditEnv(episode=self.episode)
        pe = PolicyEnv(env)
        result = run_policy(pe, SilencePolicy())
        self.assertEqual(result["steps"], 1)
        # PolicyEnv deliberately hides grading diagnostics.
        self.assertIsNone(result["grade"])
        self.assertIsNone(result["graded_reward"])

    def test_deterministic_trajectories(self):
        a = run_policy(AuditEnv(episode=self.episode), ReconciliationPolicy())
        b = run_policy(AuditEnv(episode=self.episode), ReconciliationPolicy())
        self.assertEqual(a["trajectory"], b["trajectory"])


if __name__ == "__main__":
    unittest.main()
