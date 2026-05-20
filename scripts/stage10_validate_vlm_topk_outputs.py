#!/usr/bin/env python3
"""Validate Stage 10 VLM top-k JSONL outputs without using labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_CLASSES = {"insulator_ok", "defect_flashover", "defect_broken", "unknown"}
ALLOWED_RANKS = {"top1", "top2", "top3", "uncertain"}
ALLOWED_SUPPORT = {"yes", "no", "uncertain"}
ALLOWED_EVIDENCE = {"strong", "medium", "weak", "none"}
ALLOWED_QUALITY = {"clear", "partial", "ambiguous", "bad"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                row = {
                    "record_id": f"__invalid_json_line_{line_no}",
                    "parse_ok": False,
                    "parse_error": f"jsonl_decode_error: {exc}",
                }
            rows.append(row)
    return rows


def count_invalid(rows: list[dict[str, Any]], field: str, allowed: set[str]) -> int:
    return sum(1 for r in rows if str(r.get(field, "")).strip() not in allowed)


def distribution(rows: list[dict[str, Any]], field: str) -> pd.DataFrame:
    counts = Counter(str(r.get(field, "")).strip() for r in rows)
    total = sum(counts.values())
    data = [
        {"field": field, "value": value, "count": count, "fraction": count / total if total else 0.0}
        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return pd.DataFrame(data)


def validate(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.outputs_jsonl))
    manifest_rows = read_jsonl(Path(args.manifest_jsonl)) if args.manifest_jsonl else []
    manifest_ids = [str(r.get("record_id", "")) for r in manifest_rows]
    output_ids = [str(r.get("record_id", "")) for r in rows]

    duplicate_ids = sorted([rid for rid, count in Counter(output_ids).items() if rid and count > 1])
    missing_ids = sorted(set(manifest_ids) - set(output_ids)) if manifest_ids and args.expect_complete_manifest else []
    unexpected_ids = sorted(set(output_ids) - set(manifest_ids)) if manifest_ids else []

    parse_ok_count = sum(1 for r in rows if r.get("parse_ok") is True)
    summary = {
        "n": len(rows),
        "parse_ok_count": parse_ok_count,
        "parse_ok_rate": parse_ok_count / len(rows) if rows else 0.0,
        "invalid_selected_rank_count": count_invalid(rows, "selected_rank", ALLOWED_RANKS),
        "invalid_selected_class_count": count_invalid(rows, "selected_class", ALLOWED_CLASSES),
        "invalid_supports_top1_count": count_invalid(rows, "supports_top1", ALLOWED_SUPPORT),
        "invalid_supports_top2_count": count_invalid(rows, "supports_top2", ALLOWED_SUPPORT),
        "invalid_supports_top3_count": count_invalid(rows, "supports_top3", ALLOWED_SUPPORT),
        "invalid_evidence_strength_count": count_invalid(rows, "evidence_strength", ALLOWED_EVIDENCE),
        "invalid_crop_quality_count": count_invalid(rows, "crop_quality", ALLOWED_QUALITY),
        "missing_record_ids": len(missing_ids),
        "duplicate_record_ids": len(duplicate_ids),
        "unexpected_record_ids": len(unexpected_ids),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / args.summary_csv_name
    summary_md = out_dir / args.summary_md_name
    distributions_csv = out_dir / "vlm_topk_output_distributions.csv"
    issues_json = out_dir / "vlm_topk_output_validation_issues.json"

    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    dist = pd.concat(
        [
            distribution(rows, "selected_rank"),
            distribution(rows, "selected_class"),
            distribution(rows, "evidence_strength"),
            distribution(rows, "needs_review"),
            distribution(rows, "crop_quality"),
        ],
        ignore_index=True,
    )
    dist.to_csv(distributions_csv, index=False)
    issues_json.write_text(
        json.dumps(
            {
                "missing_record_ids": missing_ids,
                "duplicate_record_ids": duplicate_ids,
                "unexpected_record_ids": unexpected_ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Stage 10 VLM Top-k Output Validation",
        "",
        "This validation checks schema and output distributions only. It does not use ground-truth labels and does not compute accuracy.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"| {key} | {rendered} |")
    lines.extend(["", "## Distributions", ""])
    for field in ["selected_rank", "selected_class", "evidence_strength", "needs_review", "crop_quality"]:
        lines.extend([f"### {field}", "", "| value | count | fraction |", "| --- | ---: | ---: |"])
        for row in dist[dist["field"] == field].to_dict(orient="records"):
            lines.append(f"| {row['value']} | {int(row['count'])} | {float(row['fraction']):.4f} |")
        lines.append("")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {distributions_csv}")
    print(f"Wrote: {issues_json}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-jsonl", required=True)
    parser.add_argument("--manifest-jsonl", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-csv-name", default="pilot_train_30_validation.csv")
    parser.add_argument("--summary-md-name", default="pilot_train_30_validation.md")
    parser.add_argument(
        "--expect-complete-manifest",
        action="store_true",
        help="Report manifest rows missing from outputs. Leave unset for pilot/limit runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    validate(args)


if __name__ == "__main__":
    main()
