"""Command-line interface for FinAudit RL.

Commands::

    finaudit generate --split all
    finaudit validate
    finaudit run --episode episodes/dev/dev-05 --policy reconciliation
    finaudit evaluate --episodes episodes/dev
    finaudit demo
    finaudit check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .env import AuditEnv
from .generator import generate_episode
from .matcher import oracle_submission, score_submission
from .models import Episode
from .policies import make_policy
from .splits import ALL_SPECS, specs_for
from .trajectory import run_named_policy, run_policy, trajectory_jsonl
from .transform import remap_episode
from .validate import validate_episode

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPISODES = ROOT / "episodes"
DEFAULT_TRAJECTORIES = ROOT / "trajectories"
POLICY_NAMES = ["random", "reconciliation", "mark_all", "silence", "oracle"]


# ---------------------------------------------------------------------------
# helpers


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _fingerprint(episode: Episode) -> str:
    blob = json.dumps(
        {"open": episode.open_dicts(), "truth": episode.truth},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _episode_dirs(path: Path) -> list[Path]:
    path = path.resolve()
    if (path / "open" / "manifest.json").is_file():
        return [path]
    if (path / "manifest.json").is_file():
        return [path]
    found = sorted({m.parent.parent for m in path.rglob("open/manifest.json")})
    return found


def _load(path: Path) -> Episode:
    return Episode.load(path)


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _read_index(root: Path) -> dict:
    path = root / "index.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------------------
# generate


def cmd_generate(args: argparse.Namespace) -> int:
    root = Path(args.out)
    specs = specs_for(args.split)
    if args.count is not None:
        specs = specs[: args.count]
    index = _read_index(root)
    entries = index.setdefault("episodes", [])
    by_name = {e["name"]: e for e in entries if "name" in e}
    rows = []
    for spec in specs:
        episode = generate_episode(spec.seed, spec.config)
        errors = validate_episode(episode)
        if errors:
            print(f"{spec.name}: validation failed", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        directory = root / spec.split / spec.name
        episode.write(directory)
        entry = {
            **spec.public(),
            "directory": str(directory.relative_to(root)),
            "episode_id": episode.episode_id,
            "n_records": len(episode.ledger),
            "n_documents": len(episode.invoices) + len(episode.payments),
            "n_issues": len(episode.truth.get("issues", [])) if episode.truth else None,
            "fingerprint": _fingerprint(episode),
        }
        by_name[spec.name] = entry
        rows.append(
            [
                spec.name,
                episode.episode_id,
                str(len(episode.ledger)),
                str(len(episode.invoices) + len(episode.payments)),
                str(entry["n_issues"]),
                entry["fingerprint"][:12],
            ]
        )
    index["schema_version"] = "1.0"
    index["episodes"] = [by_name[name] for name in sorted(by_name)]
    _json_dump(root / "index.json", index)
    _print_table(rows, ["name", "episode_id", "rows", "docs", "issues", "sha256"])
    print(f"\nWrote {len(rows)} episode(s) under {root}")
    return 0


# ---------------------------------------------------------------------------
# validate


def cmd_validate(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in (args.paths or [str(DEFAULT_EPISODES)])]
    dirs: list[Path] = []
    for path in paths:
        dirs.extend(_episode_dirs(path))
    if not dirs:
        print(f"no episodes found under {', '.join(map(str, paths))}", file=sys.stderr)
        return 1
    failures = 0
    rows = []
    for directory in dirs:
        try:
            episode = _load(directory)
        except Exception as exc:
            print(f"{directory}: load failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        errors = validate_episode(episode)
        if args.remap:
            remapped = remap_episode(episode, seed=args.remap_seed)
            errors += [f"remap: {e}" for e in validate_episode(remapped)]
        rows.append(
            [
                str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
                episode.episode_id,
                str(len(episode.ledger)),
                str(len(episode.invoices) + len(episode.payments)),
                str(len(episode.truth.get("issues", [])) if episode.truth else "?"),
                "OK" if not errors else f"{len(errors)} error(s)",
            ]
        )
        if errors:
            failures += 1
            for error in errors:
                print(f"{directory}: {error}", file=sys.stderr)
    _print_table(rows, ["episode", "id", "rows", "docs", "issues", "status"])
    if failures:
        print(f"\n{failures} episode(s) failed validation", file=sys.stderr)
        return 1
    print(f"\nAll {len(dirs)} episode(s) valid")
    return 0


# ---------------------------------------------------------------------------
# run


def cmd_run(args: argparse.Namespace) -> int:
    directory = Path(args.episode)
    episode = _load(directory)
    result = run_named_policy(
        episode,
        args.policy,
        seed=args.seed,
        mode=args.mode,
        include_observations=args.full_observations,
    )
    if args.out:
        _json_dump(Path(args.out), result)
    summary = {
        "policy": result["policy"],
        "episode_id": result["episode_id"],
        "reward": result["graded_reward"],
        "steps": result["steps"],
        "grade": result["grade"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# evaluate


def cmd_evaluate(args: argparse.Namespace) -> int:
    dirs = _episode_dirs(Path(args.episodes))
    if not dirs:
        print(f"no episodes found under {args.episodes}", file=sys.stderr)
        return 1
    if args.policies in ("all", None):
        policies = list(POLICY_NAMES)
    else:
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    records = []
    for directory in dirs:
        episode = _load(directory)
        for name in policies:
            record = run_named_policy(
                episode,
                name,
                seed=args.seed,
                mode="eval",
                include_observations=False,
            )
            record["episode_dir"] = str(directory)
            record["n_issues"] = len(episode.truth.get("issues", [])) if episode.truth else None
            records.append(record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(trajectory_jsonl(records), encoding="utf-8")

    summary: dict[str, Any] = {"episodes": len(dirs), "policies": {}, "records": len(records)}
    for name in policies:
        subset = [r for r in records if r["policy"] == name]
        if not subset:
            continue
        grades = [r["grade"] for r in subset if r["grade"]]
        summary["policies"][name] = {
            "runs": len(subset),
            "mean_reward": sum(r["graded_reward"] for r in subset) / len(subset),
            "min_reward": min(r["graded_reward"] for r in subset),
            "max_reward": max(r["graded_reward"] for r in subset),
            "mean_steps": sum(r["steps"] for r in subset) / len(subset),
            "mean_tp": sum(g["tp"] for g in grades) / len(grades) if grades else 0.0,
            "mean_fp": sum(g["fp"] for g in grades) / len(grades) if grades else 0.0,
            "mean_fn": sum(g["fn"] for g in grades) / len(grades) if grades else 0.0,
        }
    summary_path = out.with_name(out.stem + "_summary.json")
    _json_dump(summary_path, summary)

    rows = []
    for name in policies:
        s = summary["policies"].get(name)
        if not s:
            continue
        rows.append(
            [
                name,
                f"{s['mean_reward']:+.3f}",
                f"{s['min_reward']:+.3f}",
                f"{s['max_reward']:+.3f}",
                f"{s['mean_steps']:.1f}",
                f"{s['mean_tp']:.2f}",
                f"{s['mean_fp']:.2f}",
                f"{s['mean_fn']:.2f}",
            ]
        )
    _print_table(rows, ["policy", "mean", "min", "max", "steps", "TP", "FP", "FN"])
    print(f"\nWrote trajectories to {out}")
    print(f"Wrote summary to {summary_path}")
    return 0


# ---------------------------------------------------------------------------
# demo


def cmd_demo(args: argparse.Namespace) -> int:
    episode = generate_episode(args.seed, {"n_issues": args.issues, "difficulty": "demo"})
    errors = validate_episode(episode)
    if errors:
        for error in errors:
            print(f"validation: {error}", file=sys.stderr)
        return 1
    print(f"Episode {episode.episode_id}: {len(episode.ledger)} ledger rows, "
          f"{len(episode.invoices) + len(episode.payments)} documents, "
          f"{len(episode.truth['issues'])} hidden issues")
    for issue in episode.truth["issues"]:
        print(f"  - {issue['issue_id']}: {issue['type']} ({issue['subtype']}) amount={issue['expected_amount_minor']}")
    print()
    rows = []
    for name in POLICY_NAMES:
        result = run_named_policy(episode, name, seed=args.seed)
        grade = result["grade"]
        rows.append(
            [
                name,
                f"{result['graded_reward']:+.3f}",
                str(result["steps"]),
                str(grade["tp"]),
                str(grade["fp"]),
                str(grade["fn"]),
                f"{grade['f1']:.3f}",
            ]
        )
    _print_table(rows, ["policy", "reward", "steps", "TP", "FP", "FN", "F1"])
    return 0


# ---------------------------------------------------------------------------
# check


def _check_behavior() -> list[str]:
    problems: list[str] = []
    clean = generate_episode(1101, {"clean": True, "n_issues": 0})
    err = generate_episode(1105, {"n_issues": 5})
    clean_oracle = run_named_policy(clean, "oracle")
    clean_silence = run_named_policy(clean, "silence")
    err_oracle = run_named_policy(err, "oracle")
    err_silence = run_named_policy(err, "silence")
    mark_all = run_named_policy(clean, "mark_all")

    if clean_oracle["graded_reward"] != 1.0:
        problems.append("oracle on clean episode did not score 1.0")
    if clean_silence["graded_reward"] != 1.0:
        problems.append("silence on clean episode did not score 1.0")
    if err_oracle["graded_reward"] != 1.0:
        problems.append("oracle on erroneous episode did not score 1.0")
    if err_silence["graded_reward"] != -1.0:
        problems.append("silence on erroneous episode did not score -1.0")
    if mark_all["graded_reward"] >= 0.0:
        problems.append("mark-all on clean episode did not score below zero")

    # Duplicate findings must not help.
    duplicate = oracle_submission(err)
    duplicate["findings"] = duplicate["findings"] + duplicate["findings"][:1]
    dup_result = score_submission(err, duplicate, inspection_steps=0)
    if not (dup_result.reward < err_oracle["graded_reward"]):
        problems.append("duplicate finding did not reduce reward")

    # Step economy only slightly differentiates equal answers.
    answer = oracle_submission(err)
    r0 = score_submission(err, answer, inspection_steps=0).reward
    r40 = score_submission(err, answer, inspection_steps=40).reward
    if abs((r0 - r40) - 0.08) > 1e-9:
        problems.append("step penalty is not 0.002 per inspection step")

    # Remapping identifiers must not change the answer.
    remapped = remap_episode(err, seed=7)
    remap_errors = validate_episode(remapped)
    if remap_errors:
        problems.append("remapped episode failed validation")
    else:
        a = run_named_policy(err, "oracle")["grade"]
        b = run_named_policy(remapped, "oracle")["grade"]
        if (a["tp"], a["fp"], a["fn"]) != (b["tp"], b["fp"], b["fn"]):
            problems.append("oracle result changed after ID remap")
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    problems: list[str] = []

    # 1. Every named spec must generate and validate.
    for spec in ALL_SPECS:
        episode = generate_episode(spec.seed, spec.config)
        errors = validate_episode(episode)
        if errors:
            problems.append(f"{spec.name}: {errors[0]}")

    # 2. Behavioral invariants.
    problems.extend(_check_behavior())

    # 3. On-disk episodes, when present.
    on_disk = _episode_dirs(DEFAULT_EPISODES) if DEFAULT_EPISODES.exists() else []
    for directory in on_disk:
        episode = _load(directory)
        errors = validate_episode(episode)
        if errors:
            problems.append(f"{directory.name}: {errors[0]}")

    # 4. Unit tests.
    if not args.skip_tests:
        tests = ROOT / "tests"
        if tests.is_dir():
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-v"],
                cwd=str(ROOT),
            )
            if completed.returncode != 0:
                problems.append("unit tests failed")

    if problems:
        print("CHECK FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"CHECK OK: {len(ALL_SPECS)} named specs valid, {len(on_disk)} on-disk episodes valid, "
        "reward and remap invariants hold"
    )
    return 0


# ---------------------------------------------------------------------------
# parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finaudit", description="FinAudit RL benchmark tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate the named train/dev episodes")
    p.add_argument("--split", default="all", choices=["train", "dev", "all"])
    p.add_argument("--out", default=str(DEFAULT_EPISODES))
    p.add_argument("--count", type=int, default=None, help="only the first N specs")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="validate episodes on disk")
    p.add_argument("paths", nargs="*", help="episode directories or a root to search")
    p.add_argument("--remap", action="store_true", help="also validate an ID-remapped copy")
    p.add_argument("--remap-seed", type=int, default=7)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run one policy on one episode")
    p.add_argument("--episode", required=True)
    p.add_argument("--policy", default="reconciliation", choices=POLICY_NAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="eval", choices=["train", "eval"])
    p.add_argument("--out", default=None)
    p.add_argument("--full-observations", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate", help="run baseline policies over a directory of episodes")
    p.add_argument("--episodes", default=str(DEFAULT_EPISODES / "dev"))
    p.add_argument("--policies", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(DEFAULT_TRAJECTORIES / "trajectories.jsonl"))
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("demo", help="generate one episode and compare baselines")
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--issues", type=int, default=3)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("check", help="single verification command")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
