#!/usr/bin/env python3
"""Build a compact Markdown report from VLM comparison CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VLM comparison report.")
    parser.add_argument("--stage3-csv", default="reports/vlm_comparison/stage3_vlm_backbone_comparison.csv")
    parser.add_argument("--stage4-csv", default="reports/vlm_comparison/stage4_vlm_backbone_comparison.csv")
    parser.add_argument("--domain-status-csv", default="reports/vlm_comparison/domain_specific_models_status.csv")
    parser.add_argument("--out", default="reports/vlm_comparison/final_vlm_comparison_summary.md")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return value or ""


def stage3_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| model | parse | schema | acc | macro-F1 | visibility macro-F1 | tag Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model}` | {parse} | {schema} | {acc} | {macro} | {vis_macro} | {tag} |".format(
                model=row.get("model_key", ""),
                parse=fmt_float(row.get("parse_success", "")),
                schema=fmt_float(row.get("schema_valid", "")),
                acc=fmt_float(row.get("coarse_acc", "")),
                macro=fmt_float(row.get("coarse_macro_f1", "")),
                vis_macro=fmt_float(row.get("visibility_macro_f1", "")),
                tag=fmt_float(row.get("tag_mean_jaccard", "")),
            )
        )
    return lines


def domain_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| candidate | status | eval mode | blocker |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{candidate}` | {status} | {mode} | {blocker} |".format(
                candidate=row.get("candidate", ""),
                status=row.get("runnable_status", ""),
                mode=row.get("expected_eval_mode", ""),
                blocker=row.get("blocker", ""),
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    stage3 = read_rows(Path(args.stage3_csv))
    stage4 = read_rows(Path(args.stage4_csv))
    domain = read_rows(Path(args.domain_status_csv))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# VLM Backbone Comparison Report",
        "",
        "This report is generated from the frozen VLM comparison tables. It keeps the clean Stage 3 protocol fixed and summarizes which models are safe to promote.",
        "",
        "## Stage 3 Structured Reporter Runs",
        "",
    ]
    lines.extend(stage3_table(stage3))
    lines.extend(
        [
            "",
            "## Domain-Specific and Coarse-Only Candidates",
            "",
        ]
    )
    lines.extend(domain_table(domain))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No new frozen VLM is promoted to Stage 4 from this pass. InternVL3-2B improves raw accuracy, but it does not improve macro-F1 and loses visibility/tag quality. The next branch should be domain adaptation or a hybrid discriminative coarse classifier plus structured Qwen reporter.",
            "",
            f"Stage 3 rows: {len(stage3)}",
            f"Stage 4 rows: {len(stage4)}",
            f"Domain status rows: {len(domain)}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
