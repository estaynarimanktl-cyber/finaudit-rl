"""Episode generator: clean baseline + controlled error plan."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Iterable

from .baseline import Baseline, Builder, Entry, _period_of
from .inject import (
    ALL_TYPES,
    TYPE_ALLOCATION,
    TYPE_DUPLICATE,
    TYPE_FX,
    TYPE_PERIOD,
    inject_duplicate,
    inject_fx,
    inject_payment_allocation,
    inject_period_cutoff,
)
from .models import Episode
from .rng import Rng
from .version import GENERATOR_VERSION


DEFAULT_CONFIG = {
    "difficulty": "medium",
    "n_issues": 0,
    "error_types": list(ALL_TYPES),
    "interacting": False,
    "clean": False,
    "n_invoices": 55,
    "n_payments": 45,
    "margin_invoices": 2,
    "max_steps": 100,
}


def _public_config(config: dict) -> dict:
    return {
        "difficulty": config.get("difficulty", "medium"),
        "max_steps": int(config.get("max_steps", 100)),
        "time_limit_seconds": int(config.get("time_limit_seconds", 60)),
        "mode": config.get("mode", "train"),
    }


def _is_foreign_invoice(baseline: Baseline, entry: Entry) -> bool:
    inv = baseline.metadata["invoice_by_doc"].get(entry.source_doc or "")
    return inv is not None and inv.currency != baseline.episode.base_currency


def _is_foreign_payment(baseline: Baseline, entry: Entry) -> bool:
    pay = baseline.metadata["payment_by_doc"].get(entry.source_doc or "")
    return pay is not None and pay.currency != baseline.episode.base_currency


def _entry_source_doc(entry: Entry) -> str | None:
    return entry.source_doc


def _eligible_duplicates(baseline: Baseline, used_entries: set[str], used_docs: set[str]) -> list[Entry]:
    out = []
    for e in baseline.entries:
        if e.entry_id in used_entries or e.source_doc is None or e.source_doc in used_docs:
            continue
        if e.kind not in ("invoice_accrual", "payment_settlement"):
            continue
        if len(e.rows) <= 4 and e.total_debit() > 0:
            out.append(e)
    return out


def _eligible_fx(baseline: Baseline, used_entries: set[str], used_docs: set[str]) -> list[tuple[Entry, str]]:
    out = []
    for e in baseline.entries:
        if e.entry_id in used_entries or e.source_doc is None or e.source_doc in used_docs:
            continue
        if e.kind == "invoice_accrual" and _is_foreign_invoice(baseline, e) and e.total_debit() >= 50_000:
            out.append((e, "accrual"))
        elif e.kind == "payment_settlement" and _is_foreign_payment(baseline, e):
            cash = next((r for r in e.rows if r.account == "1000"), None)
            if cash is not None and cash.amount_minor >= 50_000:
                out.append((e, "settlement"))
    return out


def _eligible_period(baseline: Baseline, used_entries: set[str], used_docs: set[str]) -> list[Entry]:
    out = []
    for e in baseline.entries:
        if e.entry_id in used_entries or e.source_doc is None or e.source_doc in used_docs:
            continue
        if e.kind in ("invoice_accrual", "payment_settlement") and len(e.rows) <= 8 and e.total_debit() > 0:
            out.append(e)
    return out


def _eligible_allocations(baseline: Baseline, used_entries: set[str], used_docs: set[str]) -> list[str]:
    out = []
    for doc_id, entry in baseline.payment_entries.items():
        if entry.entry_id in used_entries or doc_id in used_docs:
            continue
        pay = baseline.metadata["payment_by_doc"].get(doc_id)
        if pay is None or pay.kind == "refund":
            continue
        src = next((r for r in entry.rows if r.account in ("1100", "2000")), None)
        if src is None or src.amount_minor <= 0:
            continue
        out.append(doc_id)
    return out


def _find_payment_for_invoice(baseline: Baseline, invoice_doc: str) -> str | None:
    inv = baseline.metadata["invoice_by_doc"].get(invoice_doc)
    if inv is None:
        return None
    for doc_id, pay in baseline.metadata["payment_by_doc"].items():
        if inv.invoice_id in (pay.invoice_refs or []) and pay.kind != "refund":
            return doc_id
    return None


def _cross_currency_candidates(baseline: Baseline) -> list[str]:
    out = []
    for doc_id in baseline.metadata.get("cross_currency_payments", []):
        entry = baseline.payment_entries.get(doc_id)
        pay = baseline.metadata["payment_by_doc"].get(doc_id)
        if entry is None or pay is None or pay.kind == "refund":
            continue
        src = next((r for r in entry.rows if r.account in ("1100", "2000")), None)
        cash = next((r for r in entry.rows if r.account == "1000"), None)
        if src is not None and cash is not None and src.amount_minor > 0:
            out.append(doc_id)
    return out


def _choose_unique(rng: Rng, candidates: list, used: set[str], key=lambda x: x):
    pool = [c for c in candidates if key(c) not in used]
    if not pool:
        return None
    return rng.choice(pool)


def _run_interaction_payment_fx_boundary(
    baseline: Baseline,
    rng: Rng,
    issue_counter: int,
    used_entries: set[str],
    used_docs: set[str],
) -> tuple[list[dict], list[dict], int]:
    candidates = _cross_currency_candidates(baseline)
    rng.shuffle(candidates)
    for pay_doc in candidates:
        entry = baseline.payment_entries[pay_doc]
        pay = baseline.metadata["payment_by_doc"][pay_doc]
        src = next((r for r in entry.rows if r.account in ("1100", "2000")), None)
        if src is None:
            continue
        invoice_doc = None
        for inv in baseline.episode.invoices:
            if src.invoice_id == inv.invoice_id:
                invoice_doc = inv.doc_id
                break
        if invoice_doc is None:
            continue
        try:
            issues: list[dict] = []
            mods: list[dict] = []
            alloc, mod = inject_payment_allocation(
                baseline,
                rng,
                f"ISS-{issue_counter:03d}",
                target_payment_doc=pay_doc,
                subtype="partial_allocation",
            )
            alloc["currency"] = baseline.episode.base_currency
            issues.append(alloc)
            mods.append(mod)
            issue_counter += 1

            fx, mod = inject_fx(
                baseline,
                rng,
                f"ISS-{issue_counter:03d}",
                target=entry,
                target_kind="settlement",
            )
            fx["currency"] = baseline.episode.base_currency
            issues.append(fx)
            mods.append(mod)
            issue_counter += 1

            cutoff, mod = inject_period_cutoff(
                baseline,
                rng,
                f"ISS-{issue_counter:03d}",
                target=entry,
            )
            cutoff["currency"] = baseline.episode.base_currency
            issues.append(cutoff)
            mods.append(mod)
            issue_counter += 1

            used_entries.add(entry.entry_id)
            used_docs.add(pay_doc)
            used_docs.add(invoice_doc)
            return issues, mods, issue_counter
        except Exception:
            # Try another candidate; the interaction is optional.
            continue
    return [], [], issue_counter


def _run_interaction_duplicate_fx(
    baseline: Baseline,
    rng: Rng,
    issue_counter: int,
    used_entries: set[str],
    used_docs: set[str],
) -> tuple[list[dict], list[dict], int]:
    # Find a foreign invoice accrual with an associated payment entry.
    pairs: list[tuple[Entry, str]] = []
    for inv_doc, inv_entry in baseline.invoice_entries.items():
        inv = baseline.metadata["invoice_by_doc"].get(inv_doc)
        if inv is None or inv.currency == baseline.episode.base_currency:
            continue
        pay_doc = _find_payment_for_invoice(baseline, inv_doc)
        if pay_doc is None:
            continue
        pay_entry = baseline.payment_entries.get(pay_doc)
        if pay_entry is None or len(pay_entry.rows) > 4:
            continue
        pairs.append((inv_entry, pay_doc))
    rng.shuffle(pairs)
    for inv_entry, pay_doc in pairs:
        pay_entry = baseline.payment_entries[pay_doc]
        inv_doc = inv_entry.source_doc
        if inv_doc is None:
            continue
        try:
            issues: list[dict] = []
            mods: list[dict] = []
            dup, mod = inject_duplicate(
                baseline,
                rng,
                f"ISS-{issue_counter:03d}",
                target=pay_entry,
                max_rows=4,
            )
            dup["currency"] = baseline.episode.base_currency
            issues.append(dup)
            mods.append(mod)
            issue_counter += 1

            fx, mod = inject_fx(
                baseline,
                rng,
                f"ISS-{issue_counter:03d}",
                target=inv_entry,
                target_kind="accrual",
            )
            fx["currency"] = baseline.episode.base_currency
            issues.append(fx)
            mods.append(mod)
            issue_counter += 1

            used_entries.add(pay_entry.entry_id)
            used_entries.add(inv_entry.entry_id)
            used_docs.add(pay_doc)
            used_docs.add(inv_doc)
            return issues, mods, issue_counter
        except Exception:
            continue
    return [], [], issue_counter


def _safe_inject(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _inject_single(
    baseline: Baseline,
    rng: Rng,
    issue_type: str,
    issue_id: str,
    used_entries: set[str],
    used_docs: set[str],
) -> tuple[dict, dict] | None:
    if issue_type == TYPE_DUPLICATE:
        target = _choose_unique(rng, _eligible_duplicates(baseline, used_entries, used_docs), used_entries, key=lambda e: e.entry_id)
        if target is None:
            return None
        result = _safe_inject(inject_duplicate, baseline, rng, issue_id, target=target)
        if result is None:
            return None
        issue, mod = result
        used_entries.add(target.entry_id)
        if target.source_doc:
            used_docs.add(target.source_doc)
        return issue, mod
    if issue_type == TYPE_ALLOCATION:
        candidates = _eligible_allocations(baseline, used_entries, used_docs)
        if not candidates:
            return None
        pay_doc = rng.choice(candidates)
        result = _safe_inject(inject_payment_allocation, baseline, rng, issue_id, target_payment_doc=pay_doc)
        if result is None:
            return None
        issue, mod = result
        used_entries.add(baseline.payment_entries[pay_doc].entry_id)
        used_docs.add(pay_doc)
        for inv in baseline.episode.invoices:
            if inv.invoice_id in (baseline.metadata["payment_by_doc"][pay_doc].invoice_refs or []):
                used_docs.add(inv.doc_id)
        return issue, mod
    if issue_type == TYPE_FX:
        candidates = _eligible_fx(baseline, used_entries, used_docs)
        if not candidates:
            return None
        entry, kind = rng.choice(candidates)
        result = _safe_inject(inject_fx, baseline, rng, issue_id, target=entry, target_kind=kind)
        if result is None:
            return None
        issue, mod = result
        used_entries.add(entry.entry_id)
        if entry.source_doc:
            used_docs.add(entry.source_doc)
        return issue, mod
    if issue_type == TYPE_PERIOD:
        target = _choose_unique(rng, _eligible_period(baseline, used_entries, used_docs), used_entries, key=lambda e: e.entry_id)
        if target is None:
            return None
        result = _safe_inject(inject_period_cutoff, baseline, rng, issue_id, target=target)
        if result is None:
            return None
        issue, mod = result
        used_entries.add(target.entry_id)
        if target.source_doc:
            used_docs.add(target.source_doc)
        return issue, mod
    raise ValueError(issue_type)


def _plan_types(rng: Rng, allowed: list[str], n_issues: int) -> list[str]:
    allowed = [t for t in allowed if t in ALL_TYPES]
    if not allowed:
        raise ValueError("no supported error types requested")
    if n_issues <= 0:
        return []
    if n_issues >= len(allowed):
        plan = list(allowed)
        while len(plan) < n_issues:
            plan.append(rng.choice(allowed))
        rng.shuffle(plan)
        return plan[:n_issues]
    # At least one of each allowed type when possible.
    plan = list(allowed)
    rng.shuffle(plan)
    return plan[:n_issues]


def generate_episode(seed: int, config: dict | None = None) -> Episode:
    config = {**DEFAULT_CONFIG, **(config or {})}
    if config.get("clean"):
        config["n_issues"] = 0
        config["interacting"] = False
    n_issues = int(config.get("n_issues", 0))
    if n_issues < 0 or n_issues > 12:
        raise ValueError("n_issues must be between 0 and 12")
    allowed = list(config.get("error_types") or ALL_TYPES)

    baseline = Builder(seed, config).build()
    rng = Rng(seed).fork("errors")
    used_entries: set[str] = set()
    used_docs: set[str] = set()
    issues: list[dict] = []
    mods: list[dict] = []
    issue_counter = 1

    # Interacting groups first; they are optional if targets are unavailable.
    remaining_types: list[str] = []
    if n_issues > 0 and config.get("interacting"):
        group_size = 3 if n_issues >= 3 else (2 if n_issues >= 2 else 1)
        group_issues: list[dict] = []
        group_mods: list[dict] = []
        if group_size == 3 and TYPE_ALLOCATION in allowed and TYPE_FX in allowed and TYPE_PERIOD in allowed:
            group_issues, group_mods, issue_counter = _run_interaction_payment_fx_boundary(
                baseline, rng, issue_counter, used_entries, used_docs
            )
        if not group_issues and group_size >= 2 and TYPE_DUPLICATE in allowed and TYPE_FX in allowed:
            group_issues, group_mods, issue_counter = _run_interaction_duplicate_fx(
                baseline, rng, issue_counter, used_entries, used_docs
            )
        issues.extend(group_issues)
        mods.extend(group_mods)
        n_issues -= len(group_issues)

    # Independent issues.
    plan = _plan_types(rng, allowed, max(0, n_issues))
    attempts = 0
    while len(plan) < n_issues and attempts < 50:
        plan.append(rng.choice(allowed))
        attempts += 1
    pending = list(plan)
    guard = 0
    while pending and guard < 200:
        guard += 1
        issue_type = pending.pop(0)
        result = _inject_single(
            baseline,
            rng,
            issue_type,
            f"ISS-{issue_counter:03d}",
            used_entries,
            used_docs,
        )
        if result is None:
            # Try another allowed type before giving up on this slot.
            alternatives = [t for t in allowed if t != issue_type]
            rng.shuffle(alternatives)
            for alt in alternatives:
                result = _inject_single(
                    baseline,
                    rng,
                    alt,
                    f"ISS-{issue_counter:03d}",
                    used_entries,
                    used_docs,
                )
                if result is not None:
                    break
        if result is None:
            raise RuntimeError(f"could not inject enough issues; missing {len(pending) + 1}")
        issue, mod = result
        issue["currency"] = baseline.episode.base_currency
        issues.append(issue)
        mods.append(mod)
        issue_counter += 1

    if len(issues) != int(config.get("n_issues", 0)):
        # A requested interaction may have degraded; fill the remainder.
        target_total = int(config.get("n_issues", 0))
        guard = 0
        while len(issues) < target_total and guard < 100:
            guard += 1
            for issue_type in allowed:
                result = _inject_single(
                    baseline,
                    rng,
                    issue_type,
                    f"ISS-{issue_counter:03d}",
                    used_entries,
                    used_docs,
                )
                if result is not None:
                    issue, mod = result
                    issue["currency"] = baseline.episode.base_currency
                    issues.append(issue)
                    mods.append(mod)
                    issue_counter += 1
                    break
            else:
                break
        if len(issues) != target_total:
            raise RuntimeError(f"generated {len(issues)} issues, expected {target_total}")

    # Stable ids and truth.
    issues.sort(key=lambda x: x["issue_id"])
    for issue in issues:
        issue["currency"] = baseline.episode.base_currency

    baseline.rebuild_ledger()
    # Shuffle final file order deterministically.
    final_rng = Rng(seed).fork("final_order")
    final_rng.shuffle(baseline.episode.ledger)
    baseline.episode.manifest["n_records"] = len(baseline.episode.ledger)
    baseline.episode.truth = {
        "schema_version": "1.0",
        "episode_id": baseline.episode.episode_id,
        "base_currency": baseline.episode.base_currency,
        "period_start": baseline.episode.period_start,
        "period_end": baseline.episode.period_end,
        "n_issues": len(issues),
        "clean": len(issues) == 0,
        "issues": issues,
    }
    baseline.episode.provenance["modifications"] = mods
    baseline.episode.provenance["baseline"]["final_ledger_rows"] = len(baseline.episode.ledger)
    baseline.episode.provenance["baseline"]["balanced_after_errors"] = baseline.episode.is_balanced()
    return baseline.episode
