import unittest

from finaudit.env import AuditEnv
from finaudit.generator import generate_episode
from finaudit.matcher import oracle_submission, score_submission, terminal_reward
from finaudit.transform import remap_episode
from finaudit.trajectory import run_named_policy


class AttackTests(unittest.TestCase):
    def setUp(self):
        self.ep_clean = generate_episode(5001, {"n_issues": 0})
        self.ep_err = generate_episode(5002, {"n_issues": 3})

    def test_correct_answer_beats_silence_on_erroneous_books(self):
        oracle = run_named_policy(self.ep_err, "oracle")
        silence = run_named_policy(self.ep_err, "silence")
        self.assertGreater(oracle["graded_reward"], silence["graded_reward"])
        self.assertEqual(oracle["graded_reward"], 1.0)
        self.assertEqual(silence["graded_reward"], -1.0)

    def test_silence_beats_false_accusations_on_clean_books(self):
        silence = run_named_policy(self.ep_clean, "silence")
        mark_all = run_named_policy(self.ep_clean, "mark_all")
        self.assertGreater(silence["graded_reward"], mark_all["graded_reward"])
        self.assertEqual(silence["graded_reward"], 1.0)

    def test_duplicates_give_no_benefit(self):
        base = score_submission(self.ep_err, oracle_submission(self.ep_err), inspection_steps=0).reward
        payload = oracle_submission(self.ep_err)
        payload["findings"] = payload["findings"] + [dict(payload["findings"][0])]
        dup = score_submission(self.ep_err, payload, inspection_steps=0).reward
        self.assertLess(dup, base)

    def test_duplicates_do_not_add_tp(self):
        payload = oracle_submission(self.ep_err)
        base = score_submission(self.ep_err, payload, inspection_steps=0)
        payload["findings"] = payload["findings"] + [dict(payload["findings"][0])]
        dup = score_submission(self.ep_err, payload, inspection_steps=0)
        self.assertEqual(base.tp, dup.tp)
        self.assertEqual(base.fp + 1, dup.fp)
        self.assertEqual(base.fn, dup.fn)

    def test_mark_all_is_bad(self):
        clean = run_named_policy(self.ep_clean, "mark_all")
        err = run_named_policy(self.ep_err, "mark_all")
        self.assertEqual(clean["graded_reward"], -1.0)
        self.assertGreater(err["grade"]["fp"], 0)
        self.assertLessEqual(err["graded_reward"], 0.0)

    def test_step_economy_only_slightly_differentiates_equal_answers(self):
        payload = oracle_submission(self.ep_err)
        r0 = score_submission(self.ep_err, payload, inspection_steps=0).reward
        r10 = score_submission(self.ep_err, payload, inspection_steps=10).reward
        r50 = score_submission(self.ep_err, payload, inspection_steps=50).reward
        self.assertAlmostEqual(r0 - r10, 0.02, places=9)
        self.assertAlmostEqual(r0 - r50, 0.1, places=9)

    def test_unknown_ids_and_evidence_are_not_tp(self):
        finding = oracle_submission(self.ep_err)["findings"][0]
        finding["affected_ids"] = ["L999999"]
        finding["evidence_ids"] = ["L999999"]
        result = score_submission(self.ep_err, {"episode_id": self.ep_err.episode_id, "findings": [finding]}, inspection_steps=0)
        self.assertEqual(result.tp, 0)
        self.assertEqual(result.fp, 1)
        self.assertEqual(result.reward, -1.0)

    def test_invalid_structure_minus_one(self):
        for payload in (
            {"episode_id": "nope", "findings": []},
            {"episode_id": self.ep_err.episode_id, "findings": [{"type": "duplicate_account"}]},
            {"episode_id": self.ep_err.episode_id},
        ):
            result = score_submission(self.ep_err, payload, inspection_steps=0)
            self.assertFalse(result.valid)
            self.assertEqual(result.reward, -1.0)

    def test_too_many_findings_is_invalid(self):
        finding = oracle_submission(self.ep_err)["findings"][0]
        payload = {"episode_id": self.ep_err.episode_id, "findings": [dict(finding) for _ in range(31)]}
        result = score_submission(self.ep_err, payload, inspection_steps=0)
        self.assertFalse(result.valid)
        self.assertEqual(result.reward, -1.0)

    def test_resubmit_after_done_is_noop(self):
        env = AuditEnv(episode=self.ep_err)
        env.reset()
        env.step({"type": "submit", "findings": oracle_submission(self.ep_err)})
        env.step({"type": "submit", "findings": oracle_submission(self.ep_err)})
        self.assertTrue(env.done)
        result = env.grade()
        self.assertEqual(result.tp, len(self.ep_err.truth["issues"]))

    def test_shuffle_and_id_remap_invariance(self):
        remapped = remap_episode(self.ep_err, seed=5)
        for name in ("oracle", "reconciliation"):
            a = run_named_policy(self.ep_err, name)
            b = run_named_policy(remapped, name)
            self.assertEqual(a["grade"]["tp"], b["grade"]["tp"])
            self.assertEqual(a["grade"]["fp"], b["grade"]["fp"])
            self.assertEqual(a["grade"]["fn"], b["grade"]["fn"])

    def test_reward_formula_is_public_and_stable(self):
        self.assertAlmostEqual(terminal_reward(1.0, 0), 1.0)
        self.assertAlmostEqual(terminal_reward(1.0, 10), 0.98)
        self.assertAlmostEqual(terminal_reward(0.5, 2), -0.004)


if __name__ == "__main__":
    unittest.main()
