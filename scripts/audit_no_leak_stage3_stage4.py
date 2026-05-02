#!/usr/bin/env python3
"""Audit Stage 3/4 artifacts for model-visible path/class leakage markers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_PATTERNS = [
    "crop_path",
    "crops/val",
    "crops/train",
    "insulator_ok/",
    "defect_flashover/",
    "defect_broken/",
    "\\insulator_ok\\",
    "\\defect_flashover\\",
    "\\defect_broken\\",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prompt/raw artifacts for leakage markers.")
    parser.add_argument("--paths", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS)
    return parser.parse_args()


def iter_text_files(paths: list[Path]):
    allowed_suffixes = {".txt", ".md", ".json", ".jsonl", ".csv", ".yaml", ".yml"}
    for root in paths:
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                yield path


def scan_file(path: Path, patterns: list[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        for pattern in patterns:
            if pattern.lower() in lower:
                hits.append(
                    {
                        "path": str(path),
                        "line": str(lineno),
                        "pattern": pattern,
                        "snippet": line[:240],
                    }
                )
    return hits


def classify_hit(hit: dict[str, str]) -> str:
    path = hit["path"].replace("\\", "/").lower()
    snippet = hit["snippet"].lower()
    if "/prompts/" in path or path.endswith("system.txt") or path.endswith("user_template.txt"):
        return "prompt_visible"
    if "raw_responses" in path or "parsed_predictions" in path:
        return "model_output_or_parser"
    if "predictions_vlm_labels_v1" in path or "case_table" in path or "review_table" in path:
        return "metadata_or_eval_artifact"
    if "manifest" in path or "summary" in path:
        return "metadata_or_summary"
    if "crop_path" in snippet:
        return "metadata_or_eval_artifact"
    return "unknown"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hits: list[dict[str, str]] = []
    scanned = 0
    for path in iter_text_files(args.paths):
        scanned += 1
        for hit in scan_file(path, args.patterns):
            hit["category"] = classify_hit(hit)
            hits.append(hit)

    with (args.out_dir / "leakage_audit_summary.json").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["path", "line", "pattern", "category", "snippet"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hits)

    prompt_visible = [h for h in hits if h["category"] == "prompt_visible"]
    summary = {
        "scanned_files": scanned,
        "total_hits": len(hits),
        "prompt_visible_hits": len(prompt_visible),
        "status": "PASS" if not prompt_visible else "REVIEW_PROMPT_VISIBLE_HITS",
    }
    (args.out_dir / "leakage_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# No-Leak Audit",
        "",
        f"Scanned files: {scanned}",
        f"Total marker hits: {len(hits)}",
        f"Prompt-visible hits: {len(prompt_visible)}",
        f"Status: `{summary['status']}`",
        "",
        "This audit flags class/path marker strings. Hits in predictions, manifests, or case tables can be legitimate metadata. Prompt-visible hits require review.",
    ]
    if prompt_visible:
        md.extend(["", "## Prompt-Visible Hits", ""])
        for hit in prompt_visible[:50]:
            md.append(f"- `{hit['path']}:{hit['line']}` pattern `{hit['pattern']}`")
    (args.out_dir / "no_leak_audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
