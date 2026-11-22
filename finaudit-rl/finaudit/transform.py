"""Episode transformations for paired invariance checks.

``remap_episode`` produces an isomorphic episode with shuffled file order and
fresh row/document identifiers.  The economic content, truth types, amounts and
evidence relationships are unchanged, so any ID-agnostic auditor should score
the same on both copies.  The utility is used by the test suite and by
``finaudit check``; it is not exposed to policies.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .models import Episode


def remap_episode(episode: Episode, seed: int = 0) -> Episode:
    rng = random.Random(int(seed))
    ep = episode.clone()

    # Snapshot the old identifiers before mutation.
    old_invoice_docs = [i.doc_id for i in ep.invoices]
    old_invoice_biz = [i.invoice_id for i in ep.invoices]
    old_payment_docs = [p.doc_id for p in ep.payments]
    old_payment_biz = [p.payment_id for p in ep.payments]
    old_row_ids = [r.row_id for r in ep.ledger]

    rng.shuffle(ep.ledger)
    rng.shuffle(ep.invoices)
    rng.shuffle(ep.payments)

    # FX rows are kept in canonical date order so write/load round-trips are
    # fingerprint-stable; economic permutations are covered by the ledger,
    # invoice and payment shuffles.

    row_map: dict[str, str] = {}
    for i, row in enumerate(ep.ledger, start=1):
        new_id = f"R{i:06d}"
        row_map[row.row_id] = new_id
        row.row_id = new_id

    invoice_doc_map: dict[str, str] = {}
    invoice_biz_map: dict[str, str] = {}
    for i, inv in enumerate(ep.invoices, start=1):
        new_doc = f"DOC-I-RM{i:04d}"
        new_biz = f"INV-RM{i:04d}"
        invoice_doc_map[inv.doc_id] = new_doc
        invoice_biz_map[inv.invoice_id] = new_biz
        inv.doc_id = new_doc
        inv.invoice_id = new_biz

    payment_doc_map: dict[str, str] = {}
    payment_biz_map: dict[str, str] = {}
    for i, pay in enumerate(ep.payments, start=1):
        new_doc = f"DOC-P-RM{i:04d}"
        new_biz = f"PAY-RM{i:04d}"
        payment_doc_map[pay.doc_id] = new_doc
        payment_biz_map[pay.payment_id] = new_biz
        pay.doc_id = new_doc
        pay.payment_id = new_biz

    # Ledger references use business invoice/payment ids.
    for row in ep.ledger:
        if row.invoice_id is not None:
            row.invoice_id = invoice_biz_map.get(row.invoice_id, row.invoice_id)
        if row.payment_id is not None:
            row.payment_id = payment_biz_map.get(row.payment_id, row.payment_id)

    for pay in ep.payments:
        pay.invoice_refs = [invoice_biz_map.get(ref, ref) for ref in pay.invoice_refs]

    # Remap truth identifiers (canonical ids, with business aliases as a
    # defensive fallback).
    full_map: dict[str, str] = {}
    full_map.update(row_map)
    full_map.update(invoice_doc_map)
    full_map.update(invoice_biz_map)
    full_map.update(payment_doc_map)
    full_map.update(payment_biz_map)

    def mapped(value: str) -> str:
        return full_map.get(value, value)

    truth = ep.truth or {}
    for issue in truth.get("issues", []):
        issue["affected_ids"] = sorted({mapped(x) for x in issue.get("affected_ids", [])})
        issue["evidence_sets"] = [
            sorted({mapped(x) for x in evidence}) for evidence in issue.get("evidence_sets", [])
        ]

    new_episode_id = "ep-" + hashlib.sha256(
        f"{episode.episode_id}|{seed}|remap".encode("utf-8")
    ).hexdigest()[:16]
    ep.manifest["episode_id"] = new_episode_id
    ep.manifest["n_records"] = len(ep.ledger)
    ep.manifest["n_documents"] = len(ep.invoices) + len(ep.payments)
    ep.manifest["n_invoices"] = len(ep.invoices)
    ep.manifest["n_payments"] = len(ep.payments)
    if ep.truth is not None:
        ep.truth["episode_id"] = new_episode_id
    if ep.provenance is not None:
        ep.provenance["episode_id"] = new_episode_id
        ep.provenance["transform"] = {
            "operation": "remap_and_shuffle",
            "seed": int(seed),
            "source_episode_id": episode.episode_id,
            "row_map_size": len(row_map),
            "invoice_map_size": len(invoice_doc_map),
            "payment_map_size": len(payment_doc_map),
        }
    return ep
