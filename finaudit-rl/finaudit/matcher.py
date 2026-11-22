"""Public matcher, metrics and terminal reward.

Matching is one-to-one and maximum-cardinality.  Ties are resolved
deterministically: findings are considered in submission order and candidate
issues in ascending issue_id order.  See README for the published rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .models import Episode
from .inject import ALL_TYPES


MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_AFFECTED_IDS = 100


@dataclass
class NormalizedFinding:
    index: int
    type: str
    affected_ids: list[str]
    claimed_amount_minor: int
    currency: str
    evidence_ids: list[str]
    explanation: str
    raw: dict


@dataclass
class MatchResult:
    valid: bool
    error: str | None
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    reward: float
    inspection_steps: int
    matches: list[tuple[int, int]] = field(default_factory=list)
    finding_status: list[str] = field(default_factory=list)
    issue_status: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "error": self.error,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "reward": self.reward,
            "inspection_steps": self.inspection_steps,
            "matches": self.matches,
            "finding_status": self.finding_status,
            "issue_status": self.issue_status,
        }


def tolerance_for(episode: Episode, expected: int) -> int:
    rounding = episode.manifest.get("rounding", {})
    absolute = int(rounding.get("amount_tolerance_minor", 2))
    bps = float(rounding.get("amount_tolerance_bps", 5))
    return max(absolute, int(round(expected * bps / 10000.0)))


def normalize_id(episode: Episode, raw: str) -> str:
    try:
        return episode.resolve_id(raw)
    except KeyError:
        return raw


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_and_normalize(episode: Episode, payload: Any) -> tuple[list[NormalizedFinding] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "submission must be an object"
    if payload.get("episode_id") != episode.episode_id:
        return None, "episode_id mismatch"
    findings_raw = payload.get("findings")
    if not isinstance(findings_raw, list):
        return None, "findings must be a list"
    if len(findings_raw) > MAX_FINDINGS:
        return None, f"too many findings (max {MAX_FINDINGS})"
    out: list[NormalizedFinding] = []
    for i, raw in enumerate(findings_raw):
        if not isinstance(raw, dict):
            return None, f"finding {i} is not an object"
        ftype = raw.get("type")
        if not isinstance(ftype, str) or not ftype:
            return None, f"finding {i}: type must be a non-empty string"
        affected = raw.get("affected_ids")
        if not isinstance(affected, list) or not all(isinstance(x, str) for x in affected):
            return None, f"finding {i}: affected_ids must be a list of strings"
        if len(affected) > MAX_AFFECTED_IDS:
            return None, f"finding {i}: too many affected_ids"
        claimed = raw.get("claimed_amount_minor")
        if not _is_int(claimed):
            return None, f"finding {i}: claimed_amount_minor must be an integer"
        if claimed < 0:
            return None, f"finding {i}: claimed_amount_minor must be non-negative"
        currency = raw.get("currency")
        if not isinstance(currency, str) or not currency:
            return None, f"finding {i}: currency must be a non-empty string"
        evidence = raw.get("evidence_ids")
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            return None, f"finding {i}: evidence_ids must be a list of strings"
        if len(evidence) > MAX_EVIDENCE_IDS:
            return None, f"finding {i}: too many evidence_ids (max {MAX_EVIDENCE_IDS})"
        explanation = raw.get("explanation", "")
        if not isinstance(explanation, str):
            return None, f"finding {i}: explanation must be a string"
        if len(explanation) > MAX_EXPLANATION_CHARS:
            return None, f"finding {i}: explanation too long"
        out.append(
            NormalizedFinding(
                index=i,
                type=ftype,
                affected_ids=sorted({normalize_id(episode, x) for x in affected}),
                claimed_amount_minor=claimed,
                currency=currency,
                evidence_ids=sorted({normalize_id(episode, x) for x in evidence}),
                explanation=explanation,
                raw=raw,
            )
        )
    return out, None


def _finding_matches(episode: Episode, finding: NormalizedFinding, issue: dict, valid_ids: set[str]) -> bool:
    if finding.type != issue["type"]:
        return False
    if finding.currency != issue["currency"]:
        return False
    if set(finding.affected_ids) != set(issue["affected_ids"]):
        return False
    if abs(finding.claimed_amount_minor - int(issue["expected_amount_minor"])) > tolerance_for(
        episode, int(issue["expected_amount_minor"])
    ):
        return False
    # Every cited evidence id must be a real record.  Extra real evidence is
    # allowed; at least one complete allowed set must be present.
    if any(x not in valid_ids for x in finding.evidence_ids):
        return False
    evidence_set = set(finding.evidence_ids)
    if not any(set(allowed) <= evidence_set for allowed in issue.get("evidence_sets", [])):
        return False
    # Every affected id must also exist.  Unknown ids are an unconfirmed claim.
    if any(x not in valid_ids for x in finding.affected_ids):
        return False
    return True


def _maximum_matching(edges: list[list[int]], n_issues: int) -> tuple[list[int], list[int]]:
    """Kuhn's algorithm with deterministic finding/issue order."""
    issue_to_finding = [-1] * n_issues

    def try_assign(fi: int, visited: set[int]) -> bool:
        for ii in edges[fi]:
            if ii in visited:
                continue
            visited.add(ii)
            current = issue_to_finding[ii]
            if current == -1 or try_assign(current, visited):
                issue_to_finding[ii] = fi
                return True
        return False

    for fi in range(len(edges)):
        try_assign(fi, set())
    finding_to_issue = [-1] * len(edges)
    for ii, fi in enumerate(issue_to_finding):
        if fi != -1:
            finding_to_issue[fi] = ii
    return issue_to_finding, finding_to_issue


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def terminal_reward(f1: float, inspection_steps: int) -> float:
    return _clip(2.0 * f1 - 1.0 - 0.002 * max(0, int(inspection_steps)), -1.0, 1.0)


def score_submission(
    episode: Episode,
    payload: Any,
    *,
    inspection_steps: int,
    invalid_reward: float = -1.0,
) -> MatchResult:
    findings, error = validate_and_normalize(episode, payload)
    if findings is None:
        return MatchResult(
            valid=False,
            error=error,
            tp=0,
            fp=0,
            fn=len(episode.truth.get("issues", [])) if episode.truth else 0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            reward=float(invalid_reward),
            inspection_steps=inspection_steps,
        )

    issues = list(episode.truth.get("issues", [])) if episode.truth else []
    valid_ids = episode.all_record_ids()
    edges: list[list[int]] = []
    for finding in findings:
        candidates = [
            ii for ii, issue in enumerate(issues) if _finding_matches(episode, finding, issue, valid_ids)
        ]
        edges.append(candidates)

    issue_to_finding, finding_to_issue = _maximum_matching(edges, len(issues))
    tp = sum(1 for x in issue_to_finding if x != -1)
    fp = len(findings) - tp
    fn = len(issues) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if 2 * tp + fp + fn > 0:
        f1 = (2 * tp) / (2 * tp + fp + fn)
    else:
        f1 = 1.0
    reward = terminal_reward(f1, inspection_steps)

    finding_status = []
    for fi, ii in enumerate(finding_to_issue):
        finding_status.append(f"TP:{issues[ii]['issue_id']}" if ii != -1 else "FP")
    issue_status = []
    for ii, fi in enumerate(issue_to_finding):
        issue_status.append(f"TP:{fi}" if fi != -1 else "FN")

    return MatchResult(
        valid=True,
        error=None,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        reward=reward,
        inspection_steps=inspection_steps,
        matches=[(fi, ii) for fi, ii in enumerate(finding_to_issue) if ii != -1],
        finding_status=finding_status,
        issue_status=issue_status,
    )


def empty_submission(episode: Episode, *, inspection_steps: int) -> dict:
    return {"episode_id": episode.episode_id, "findings": []}


def oracle_submission(episode: Episode) -> dict:
    """Evaluator-only oracle: the truth converted into a valid submission."""
    findings = []
    for issue in episode.truth.get("issues", []):
        evidence = issue["evidence_sets"][0] if issue["evidence_sets"] else []
        findings.append(
            {
                "type": issue["type"],
                "affected_ids": list(issue["affected_ids"]),
                "claimed_amount_minor": int(issue["expected_amount_minor"]),
                "currency": issue["currency"],
                "evidence_ids": list(evidence),
                "explanation": f"Oracle finding for {issue['issue_id']} ({issue['type']}).",
            }
        )
    return {"episode_id": episode.episode_id, "findings": findings}
