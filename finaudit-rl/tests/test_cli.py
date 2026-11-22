import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from finaudit.cli import (
    ROOT,
    cmd_check,
    cmd_demo,
    cmd_evaluate,
    cmd_generate,
    cmd_run,
    cmd_validate,
    main,
)


class CliTests(unittest.TestCase):
    def test_generate_validate_run_evaluate_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "episodes"
            out = Path(tmp) / "run.json"
            traj = Path(tmp) / "traj" / "trajectories.jsonl"

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(cmd_generate(type("A", (), {"split": "train", "out": str(root), "count": 1})()), 0)
            self.assertTrue((root / "index.json").is_file())
            episode_dir = root / "train" / "train-01"
            self.assertTrue((episode_dir / "open" / "manifest.json").is_file())
            self.assertTrue((episode_dir / "private" / "truth.json").is_file())

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_validate(type("A", (), {"paths": [str(root)], "remap": True, "remap_seed": 3})()), 0)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cmd_run(type("A", (), {"episode": str(episode_dir), "policy": "reconciliation", "seed": 0, "mode": "eval", "out": str(out), "full_observations": False})()),
                    0,
                )
            summary = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(summary["policy"], "reconciliation")
            self.assertIn("graded_reward", summary)
            self.assertIn("grade", summary)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    cmd_evaluate(type("A", (), {"episodes": str(root / "train"), "policies": "all", "seed": 0, "out": str(traj)})()),
                    0,
                )
            lines = [json.loads(line) for line in traj.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(lines)
            summary_path = traj.with_name("trajectories_summary.json")
            self.assertTrue(summary_path.is_file())
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("random", summary_data["policies"])
            self.assertIn("oracle", summary_data["policies"])

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_demo(type("A", (), {"seed": 7, "issues": 2})()), 0)

    def test_validate_missing_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                cmd_validate(type("A", (), {"paths": [Path(tmp) / "nope"], "remap": False, "remap_seed": 1})()),
                1,
            )

    def test_check_runs_behavior_invariants(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_check(type("A", (), {"skip_tests": True})()), 0)

    def test_main_parser_dispatches(self):
        # `main` requires a command; missing command exits with SystemExit(2).
        with self.assertRaises(SystemExit):
            main([])
        # `main("validate")` should run; with a nonexistent root it returns 1.
        self.assertEqual(main(["validate", str(ROOT / "no-such-episodes")]), 1)


if __name__ == "__main__":
    unittest.main()
