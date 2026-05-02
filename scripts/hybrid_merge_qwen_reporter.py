#!/usr/bin/env python3
"""Merge discriminative coarse-class predictions into Qwen reporter JSONL.

The VLM output remains the source for `visibility`, evidence tags and text fields.
Only `coarse_class` is overridden, plus `needs_review` can optionally be marked
when the override disagrees with Qwen.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

ALLOWED_CLASSES = {"insulator_ok", "defect_flashover", "defect_broken", "unknown"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_classifier_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "record_id" not in fields:
            raise ValueError("Classifier CSV missing column: record_id")
        if "pred_coarse_class" not in fields and "pred" not in fields and "hybrid_pred" not in fields:
            raise ValueError("Classifier CSV needs one of: pred_coarse_class, pred, hybrid_pred")
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            rid = row["record_id"]
            if rid in out:
                raise ValueError(f"Duplicate record_id in classifier CSV: {rid}")
            out[rid] = row
        return out


def classifier_class(row: Dict[str, str]) -> str:
    for key in ("pred_coarse_class", "pred", "hybrid_pred"):
        value = row.get(key)
        if value:
            return value if value in ALLOWED_CLASSES else "unknown"
    return "unknown"


def classifier_confidence(row: Dict[str, str]) -> float:
    for key in ("confidence", "clip_linear_probe_confidence", "max_probability"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                return 0.0
    # If the classifier produced a hard class with no probability, treat it as selected.
    return 1.0 if classifier_class(row) != "unknown" else 0.0


def choose_class(qwen_class: str, clf_row: Dict[str, str] | None, mode: str, threshold: float) -> tuple[str, str, float, str]:
    if clf_row is None:
        return qwen_class, "missing_classifier_keep_qwen", 0.0, "missing"
    clf_class = classifier_class(clf_row)
    conf = classifier_confidence(clf_row)
    if mode == "hard":
        if clf_class != "unknown":
            return clf_class, "classifier_hard_override", conf, clf_class
        return qwen_class, "classifier_unknown_keep_qwen", conf, clf_class
    if mode == "confidence_gate":
        if clf_class != "unknown" and conf >= threshold:
            return clf_class, "classifier_confident_override", conf, clf_class
        return qwen_class, "classifier_low_confidence_keep_qwen", conf, clf_class
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-jsonl", required=True, type=Path)
    parser.add_argument("--classifier-csv", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--decision-csv", required=True, type=Path)
    parser.add_argument("--mode", choices=["hard", "confidence_gate"], default="hard")
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--mark-needs-review-on-change", action="store_true")
    args = parser.parse_args()

    qwen_rows = read_jsonl(args.qwen_jsonl)
    clf_rows = read_classifier_csv(args.classifier_csv)

    merged: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for row in qwen_rows:
        rid = str(row.get("record_id"))
        if not rid or rid == "None":
            raise KeyError(f"Missing record_id in Qwen row: {row.keys()}")
        qwen_class = str(row.get("coarse_class", "unknown"))
        new_class, reason, conf, clf_class = choose_class(qwen_class, clf_rows.get(rid), args.mode, args.confidence_threshold)

        out = json.loads(json.dumps(row))
        out["coarse_class"] = new_class
        if args.mark_needs_review_on_change and qwen_class != new_class:
            out["needs_review"] = True
        merged.append(out)
        decisions.append(
            {
                "record_id": rid,
                "qwen_class": qwen_class,
                "classifier_class": clf_class,
                "hybrid_class": new_class,
                "confidence": f"{conf:.6f}",
                "decision_reason": reason,
                "changed": str(qwen_class != new_class).lower(),
            }
        )

    write_jsonl(args.out_jsonl, merged)
    args.decision_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.decision_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["record_id", "qwen_class", "classifier_class", "hybrid_class", "confidence", "decision_reason", "changed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)

    print(f"Wrote {len(merged)} merged predictions to {args.out_jsonl}")
    print(f"Wrote decisions to {args.decision_csv}")


if __name__ == "__main__":
    main()
