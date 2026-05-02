#!/usr/bin/env python3
"""Paired comparison helper for small validation sets.

Input CSV columns:
  record_id, gt, baseline_pred, candidate_pred

The output JSON contains accuracy, macro-F1, per-class recall, bootstrap CIs
for paired deltas, and an exact binomial sign test over discordant correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

CLASSES = ["insulator_ok", "defect_flashover", "defect_broken"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"record_id", "gt", "baseline_pred", "candidate_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        return list(reader)


def recall_by_class(gts: Sequence[str], preds: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in CLASSES:
        denom = sum(1 for g in gts if g == cls)
        num = sum(1 for g, p in zip(gts, preds) if g == cls and p == cls)
        out[cls] = num / denom if denom else float("nan")
    return out


def macro_f1(gts: Sequence[str], preds: Sequence[str]) -> float:
    f1s: List[float] = []
    for cls in CLASSES:
        tp = sum(1 for g, p in zip(gts, preds) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gts, preds) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gts, preds) if g == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / len(f1s)


def accuracy(gts: Sequence[str], preds: Sequence[str]) -> float:
    return sum(1 for g, p in zip(gts, preds) if g == p) / len(gts)


def confusion(gts: Sequence[str], preds: Sequence[str]) -> Dict[str, Dict[str, int]]:
    pred_labels = CLASSES + ["unknown", "other"]
    matrix: Dict[str, Dict[str, int]] = {g: {p: 0 for p in pred_labels} for g in CLASSES}
    for g, p in zip(gts, preds):
        if g not in matrix:
            continue
        pp = p if p in matrix[g] else "other"
        matrix[g][pp] += 1
    return matrix


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--changed-csv", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    gts = [r["gt"] for r in rows]
    base = [r["baseline_pred"] for r in rows]
    cand = [r["candidate_pred"] for r in rows]

    base_correct = [g == p for g, p in zip(gts, base)]
    cand_correct = [g == p for g, p in zip(gts, cand)]
    base_only = sum(b and not c for b, c in zip(base_correct, cand_correct))
    cand_only = sum(c and not b for b, c in zip(base_correct, cand_correct))
    discordant = base_only + cand_only

    rng = random.Random(args.seed)
    acc_deltas: List[float] = []
    f1_deltas: List[float] = []
    n = len(rows)
    for _ in range(args.n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        bg = [gts[i] for i in idxs]
        bb = [base[i] for i in idxs]
        cc = [cand[i] for i in idxs]
        acc_deltas.append(accuracy(bg, cc) - accuracy(bg, bb))
        f1_deltas.append(macro_f1(bg, cc) - macro_f1(bg, bb))

    result = {
        "n": n,
        "baseline": {
            "accuracy": accuracy(gts, base),
            "macro_f1": macro_f1(gts, base),
            "recall_by_class": recall_by_class(gts, base),
            "confusion": confusion(gts, base),
        },
        "candidate": {
            "accuracy": accuracy(gts, cand),
            "macro_f1": macro_f1(gts, cand),
            "recall_by_class": recall_by_class(gts, cand),
            "confusion": confusion(gts, cand),
        },
        "delta": {
            "accuracy": accuracy(gts, cand) - accuracy(gts, base),
            "accuracy_ci95": [percentile(acc_deltas, 0.025), percentile(acc_deltas, 0.975)],
            "macro_f1": macro_f1(gts, cand) - macro_f1(gts, base),
            "macro_f1_ci95": [percentile(f1_deltas, 0.025), percentile(f1_deltas, 0.975)],
        },
        "paired_correctness": {
            "baseline_only_correct": base_only,
            "candidate_only_correct": cand_only,
            "discordant_pairs": discordant,
            "binomial_sign_test_p_two_sided": binom_two_sided_p(cand_only, discordant),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.changed_csv:
        changed = [
            r
            for r, b, c in zip(rows, base_correct, cand_correct)
            if r["baseline_pred"] != r["candidate_pred"] or b != c
        ]
        args.changed_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.changed_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["record_id", "gt", "baseline_pred", "candidate_pred"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fieldnames} for row in changed)

    print(json.dumps(result["delta"], indent=2))


if __name__ == "__main__":
    main()
