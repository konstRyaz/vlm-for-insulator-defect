#!/usr/bin/env python3
"""Build paired Stage 4 comparison tables from two case-table CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired Stage 4 case tables.")
    parser.add_argument("--baseline-case-table", required=True, type=Path)
    parser.add_argument("--candidate-case-table", required=True, type=Path)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_case_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if "record_id" not in rows[0]:
        raise ValueError(f"{path} has no record_id column")
    return {row["record_id"]: row for row in rows}


def is_correct(row: dict[str, str]) -> bool:
    value = row.get("vlm_correct_on_good_crop")
    if value is None or value == "":
        value = row.get("pipeline_correct")
    return str(value).strip().lower() == "true"


def pred_class(row: dict[str, str]) -> str:
    return row.get("pred_vlm_coarse_class") or row.get("coarse_class") or ""


def gt_class(row: dict[str, str]) -> str:
    return row.get("gt_coarse_class") or ""


def exact_two_sided_sign_p(helped: int, hurt: int) -> float:
    n = helped + hurt
    if n == 0:
        return 1.0
    k = min(helped, hurt)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def bootstrap_delta(records: list[dict[str, object]], iters: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(records)
    deltas: list[float] = []
    for _ in range(iters):
        cand = 0
        base = 0
        for _ in range(n):
            row = records[rng.randrange(n)]
            cand += int(bool(row["candidate_correct"]))
            base += int(bool(row["baseline_correct"]))
        deltas.append((cand - base) / n)
    return {
        "delta_rate_mean": sum(deltas) / len(deltas),
        "delta_rate_ci95_low": percentile(deltas, 0.025),
        "delta_rate_ci95_high": percentile(deltas, 0.975),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    baseline = read_case_table(args.baseline_case_table)
    candidate = read_case_table(args.candidate_case_table)
    ids = sorted(set(baseline).intersection(candidate))
    if not ids:
        raise ValueError("No overlapping record_id values")

    paired_rows: list[dict[str, object]] = []
    for rid in ids:
        b = baseline[rid]
        c = candidate[rid]
        b_ok = is_correct(b)
        c_ok = is_correct(c)
        if (not b_ok) and c_ok:
            category = "helped"
        elif b_ok and (not c_ok):
            category = "hurt"
        elif b_ok and c_ok:
            category = "both_correct"
        else:
            category = "both_wrong"
        paired_rows.append(
            {
                "record_id": rid,
                "gt_coarse_class": gt_class(c) or gt_class(b),
                "baseline_pred": pred_class(b),
                "candidate_pred": pred_class(c),
                "baseline_correct": b_ok,
                "candidate_correct": c_ok,
                "category": category,
                "candidate_matched_pred": c.get("matched_pred_record_id", ""),
                "candidate_match_iou": c.get("match_iou", ""),
                "candidate_good_crop": c.get("is_good_crop", ""),
                "candidate_error_bucket": c.get("error_bucket", ""),
            }
        )

    helped = sum(1 for r in paired_rows if r["category"] == "helped")
    hurt = sum(1 for r in paired_rows if r["category"] == "hurt")
    both_correct = sum(1 for r in paired_rows if r["category"] == "both_correct")
    both_wrong = sum(1 for r in paired_rows if r["category"] == "both_wrong")
    baseline_correct = helped * 0 + hurt + both_correct
    candidate_correct = helped + both_correct
    total = len(paired_rows)
    boot = bootstrap_delta(paired_rows, args.bootstrap_iters, args.seed)
    summary = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "total": total,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "delta_correct": candidate_correct - baseline_correct,
        "baseline_rate": baseline_correct / total,
        "candidate_rate": candidate_correct / total,
        "delta_rate": (candidate_correct - baseline_correct) / total,
        "helped": helped,
        "hurt": hurt,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "changed_total": helped + hurt,
        "sign_test_two_sided_p": exact_two_sided_sign_p(helped, hurt),
        **boot,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "paired_cases.csv", paired_rows)
    write_csv(args.out_dir / "helped_cases.csv", [r for r in paired_rows if r["category"] == "helped"])
    write_csv(args.out_dir / "hurt_cases.csv", [r for r in paired_rows if r["category"] == "hurt"])
    (args.out_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Stage 4 Paired Analysis",
        "",
        f"Baseline: `{args.baseline_name}`",
        f"Candidate: `{args.candidate_name}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total | {total} |",
        f"| baseline correct | {baseline_correct} |",
        f"| candidate correct | {candidate_correct} |",
        f"| delta correct | {candidate_correct - baseline_correct} |",
        f"| helped | {helped} |",
        f"| hurt | {hurt} |",
        f"| sign-test p | {summary['sign_test_two_sided_p']:.4f} |",
        f"| bootstrap delta 95% CI | [{summary['delta_rate_ci95_low']:.4f}, {summary['delta_rate_ci95_high']:.4f}] |",
        "",
        "Interpretation: the sign test uses only changed cases. The bootstrap interval is over paired GT objects and should be read cautiously on a 58-object validation slice.",
    ]
    (args.out_dir / "paired_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
