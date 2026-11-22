import copy
import unittest

from finaudit.env import AuditEnv, PolicyEnv
from finaudit.generator import generate_episode
from finaudit.inject import ALL_TYPES
from finaudit.policies import (
    MarkAllPolicy,
    OraclePolicy,
    Policy,
    RandomPolicy,
    ReconciliationPolicy,
    SilencePolicy,
    make_policy,
)
from finaudit.trajectory import run_named_policy


def episode_with_issue_type(issue_type: str, seed: int = 1001) -> object:
    return generate_episode(seed, {"n_issues": 1, "error_types": [issue_type]})


def run_via_policy_env(episode, policy):
    env = AuditEnv(episode=episode)
    pe = PolicyEnv(env)
    obs, info = env.reset()
    policy.reset(obs, info)
    while not env.done:
        pe.step(policy.act(obs))
        obs = env.last_observation
    return env.grade()


class BaselinePolicyTests(unittest.TestCase):
    def setUp(self):
        self.ep = generate_episode(123, {"n_issues": 2})
        self.clean = generate_episode(456, {"n_issues": 0})

    def test_make_policy_factory(self):
        for name in ("random", "reconciliation", "mark_all", "silence", "oracle"):
            with self.subTest(name=name):
                policy = make_policy(name, episode=self.ep if name == "oracle" else None)
                self.assertEqual(policy.name, name)
        with self.assertRaises(ValueError):
            make_policy("nope")
        with self.assertRaises(ValueError):
            make_policy("oracle")

    def test_oracle_is_perfect_and_deterministic(self):
        policy = OraclePolicy(self.ep)
        result = run_named_policy(self.ep, "oracle")
        self.assertEqual(result["graded_reward"], 1.0)
        self.assertEqual(result["grade"]["tp"], 2)
        self.assertEqual(result["grade"]["fp"], 0)
        self.assertEqual(result["grade"]["fn"], 0)
        again = run_named_policy(self.ep, "oracle")
        self.assertEqual(again["trajectory"], result["trajectory"])

    def test_oracle_requires_episode(self):
        with self.assertRaises(ValueError):
            make_policy("oracle")

    def test_silence_on_clean_and_erroneous(self):
        clean = run_named_policy(self.clean, "silence")
        err = run_named_policy(self.ep, "silence")
        self.assertEqual(clean["graded_reward"], 1.0)
        self.assertEqual(err["graded_reward"], -1.0)

    def test_mark_all_on_clean_is_bad(self):
        result = run_named_policy(self.clean, "mark_all")
        self.assertEqual(result["graded_reward"], -1.0)
        self.assertGreater(result["grade"]["fp"], 0)

    def test_mark_all_on_erroneous_has_fps(self):
        result = run_named_policy(self.ep, "mark_all")
        self.assertGreater(result["grade"]["fp"], 0)
        self.assertGreater(result["grade"]["fn"], 0)

    def test_reconciliation_on_clean_has_no_false_positives(self):
        result = run_named_policy(self.clean, "reconciliation")
        self.assertEqual(result["grade"]["fp"], 0)
        self.assertGreaterEqual(result["graded_reward"], 0.9)

    def test_reconciliation_detects_each_error_type(self):
        for issue_type in ALL_TYPES:
            with self.subTest(issue_type=issue_type):
                episode = episode_with_issue_type(issue_type)
                result = run_named_policy(episode, "reconciliation")
                self.assertEqual(result["grade"]["tp"], 1, issue_type)
                self.assertEqual(result["grade"]["fp"], 0, issue_type)
                self.assertEqual(result["grade"]["fn"], 0, issue_type)

    def test_reconciliation_detects_interacting_scenario(self):
        episode = generate_episode(
            7,
            {
                "n_issues": 3,
                "interacting": True,
                "error_types": ["payment_allocation", "fx_conversion", "period_cutoff"],
            },
        )
        result = run_named_policy(episode, "reconciliation")
        self.assertEqual(result["grade"]["tp"], 3)
        self.assertEqual(result["grade"]["fp"], 0)

    def test_random_is_deterministic_and_narrower_than_silence(self):
        a = run_named_policy(self.ep, "random", seed=17)
        b = run_named_policy(self.ep, "random", seed=17)
        self.assertEqual(a["trajectory"], b["trajectory"])
        # Random occasionally proposes no false findings; silence always does.
        self.assertLessEqual(a["graded_reward"], 1.0)
        self.assertLessEqual(run_named_policy(self.clean, "random", seed=17)["graded_reward"], 1.0)

    def test_reconciliation_is_deterministic(self):
        a = run_named_policy(self.ep, "reconciliation")
        b = run_named_policy(self.ep, "reconciliation")
        self.assertEqual(a["trajectory"], b["trajectory"])

    def test_correct_answer_beats_silence_on_erroneous_books(self):
        oracle = run_named_policy(self.ep, "oracle")
        silence = run_named_policy(self.ep, "silence")
        self.assertGreater(oracle["graded_reward"], silence["graded_reward"])
        self.assertEqual(oracle["graded_reward"], 1.0)

    def test_silence_beats_false_accusations_on_clean_books(self):
        silence = run_named_policy(self.clean, "silence")
        mark_all = run_named_policy(self.clean, "mark_all")
        self.assertEqual(silence["graded_reward"], 1.0)
        self.assertLess(mark_all["graded_reward"], silence["graded_reward"])

    def test_duplicates_do_not_give_benefit(self):
        from finaudit.matcher import oracle_submission, score_submission

        base = run_named_policy(self.ep, "oracle")
        finding = oracle_submission(self.ep)["findings"][0]
        payload = oracle_submission(self.ep)
        payload["findings"] = payload["findings"] + [dict(finding)]
        dup = score_submission(self.ep, payload, inspection_steps=0).reward
        self.assertLess(dup, base["graded_reward"])

    def test_step_economy_differentiates_equal_answers_slightly(self):
        from finaudit.matcher import oracle_submission, score_submission

        payload = oracle_submission(self.ep)
        r0 = score_submission(self.ep, payload, inspection_steps=0).reward
        r50 = score_submission(self.ep, payload, inspection_steps=50).reward
        self.assertAlmostEqual(r0 - r50, 0.1, places=9)

    def test_policies_do_not_read_truth(self):
        episode = self.ep.clone()
        for policy in (RandomPolicy(), ReconciliationPolicy(), MarkAllPolicy(), SilencePolicy()):
            self.assertFalse(hasattr(policy, "truth"))
        before = copy.deepcopy(episode.truth)
        grade = run_via_policy_env(episode, ReconciliationPolicy())
        self.assertEqual(episode.truth, before)
        self.assertTrue(grade.valid)
        self.assertNotIn("truth", ReconciliationPolicy().__dict__)

    def test_reconciliation_runs_on_policy_env(self):
        for issue_type in ALL_TYPES:
            with self.subTest(issue_type=issue_type):
                episode = episode_with_issue_type(issue_type, seed=3001)
                grade = run_via_policy_env(episode, ReconciliationPolicy())
                self.assertEqual(grade.tp, 1)
                self.assertEqual(grade.fp, 0)
                self.assertEqual(grade.fn, 0)

    def test_policy_name_protocol(self):
        self.assertIsInstance(RandomPolicy(), Policy)
        self.assertIsInstance(ReconciliationPolicy(), Policy)
        self.assertIsInstance(MarkAllPolicy(), Policy)
        self.assertIsInstance(SilencePolicy(), Policy)
        self.assertTrue(OraclePolicy(self.ep).evaluator_only)
        self.assertFalse(ReconciliationPolicy().evaluator_only)


if __name__ == "__main__":
    unittest.main()
