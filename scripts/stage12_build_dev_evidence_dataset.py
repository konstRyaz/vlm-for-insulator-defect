#!/usr/bin/env python3
"""Build Stage 12 development evidence dataset and pilot ID splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def margin_bucket(x: float) -> str:
    if x < 0.10:
        return "[0.00,0.10)"
    if x < 0.20:
        return "[0.10,0.20)"
    if x < 0.30:
        return "[0.20,0.30)"
    if x < 0.50:
        return "[0.30,0.50)"
    return "[0.50,+)"


def sanitize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str) and value.strip():
        raw = [x.strip() for x in value.split(",") if x.strip()]
    else:
        raw = []
    out: list[str] = []
    seen = set()
    for tag in raw:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-reference-csv", required=True)
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.eval_reference_csv)
    man = pd.DataFrame(read_jsonl(Path(args.manifest_jsonl)))

    if "record_id" not in ref.columns or "record_id" not in man.columns:
        raise ValueError("Both inputs must include record_id.")

    # Dev-only (historical train).
    ref = ref[ref["split"].astype(str) == "train"].copy()
    man = man[man["split"].astype(str) == "train"].copy()

    df = ref.merge(
        man[
            [
                "record_id",
                "resolved_image_path",
                "dino_top1",
                "dino_top2",
                "dino_top3",
                "dino_top1_score",
                "dino_top2_score",
                "dino_top3_score",
                "dino_margin",
            ]
        ],
        on="record_id",
        how="inner",
        suffixes=("", "_m"),
    )

    if len(df) != 182:
        raise ValueError(f"Expected 182 development rows, got {len(df)}.")

    # Prompt-safe path checks.
    bad_paths = 0
    resolved_paths: list[str] = []
    for raw in df["resolved_image_path"].astype(str):
        norm = raw.replace("\\", "/")
        name = norm.split("/")[-1]
        if any(k in norm.lower() for k in ["insulator_ok", "defect_flashover", "defect_broken"]):
            bad_paths += 1
        pth = Path(norm)
        if not pth.exists():
            # Try relative to repo root.
            pth = Path.cwd() / norm
        if not pth.exists():
            raise FileNotFoundError(f"Missing image: {raw}")
        resolved_paths.append(str(pth))
    if bad_paths > 0:
        raise ValueError(f"Found {bad_paths} class-leaking image paths.")

    out_rows: list[dict[str, Any]] = []
    for row, abs_path in zip(df.to_dict(orient="records"), resolved_paths):
        out_rows.append(
            {
                "record_id": str(row["record_id"]),
                "split": "train",
                "image_path": abs_path,
                "coarse_class": str(row.get("label_coarse_class", "")),
                "visual_evidence_tags": sanitize_tags(row.get("label_visual_evidence_tags", [])),
                "visibility": str(row.get("label_visibility", "")),
                "needs_review": bool(row.get("label_needs_review", False)),
                "short_canonical_description": str(row.get("label_short_canonical_description", "")),
                "report_snippet": str(row.get("label_report_snippet", "")),
            }
        )

    # Strata features for pilot selection.
    strata = df.copy()
    strata["margin_bucket"] = strata["dino_margin"].astype(float).map(margin_bucket)
    strata["pair"] = strata["dino_top1"].astype(str) + "->" + strata["dino_top2"].astype(str)
    strata["is_error"] = ~strata["dino_top1_correct"].astype(bool)
    strata["is_nonclear"] = strata["label_visibility"].astype(str).isin(["partial", "ambiguous"])
    strata["is_review"] = strata["label_needs_review"].astype(bool)
    strata["is_non_ok"] = strata["label_coarse_class"].astype(str) != "insulator_ok"

    # Pilot30: hard-first with stratified controls.
    selected30: list[str] = []
    hard = strata[strata["is_error"] | strata["is_nonclear"] | strata["is_review"] | strata["is_non_ok"]]
    for rid in hard["record_id"].astype(str).tolist():
        if rid not in selected30:
            selected30.append(rid)
        if len(selected30) >= 24:
            break
    controls = strata[~strata["record_id"].astype(str).isin(selected30)].copy()
    controls = controls.sort_values(["margin_bucket", "dino_margin", "record_id"])
    control_pool = controls["record_id"].astype(str).tolist()
    rng.shuffle(control_pool)
    for rid in control_pool:
        if len(selected30) >= 30:
            break
        selected30.append(rid)

    # Pilot60: pilot30 + extra hard/minority + stratified fill.
    selected60 = list(selected30)
    extra_hard = strata[
        (~strata["record_id"].astype(str).isin(selected60))
        & (strata["is_error"] | strata["is_nonclear"] | strata["is_review"] | strata["is_non_ok"])
    ]
    for rid in extra_hard["record_id"].astype(str).tolist():
        if len(selected60) >= 48:
            break
        selected60.append(rid)

    remain = strata[~strata["record_id"].astype(str).isin(selected60)].copy()
    by_bucket = {}
    for rid, bucket in zip(remain["record_id"].astype(str), remain["margin_bucket"].astype(str)):
        by_bucket.setdefault(bucket, []).append(rid)
    for k in sorted(by_bucket):
        rng.shuffle(by_bucket[k])
    while len(selected60) < 60:
        progressed = False
        for k in sorted(by_bucket):
            if by_bucket[k]:
                selected60.append(by_bucket[k].pop())
                progressed = True
                if len(selected60) >= 60:
                    break
        if not progressed:
            break

    # Save outputs.
    ds_jsonl = out_dir / "stage12_dev_evidence_dataset.jsonl"
    eval_csv = out_dir / "stage12_dev_eval_reference.csv"
    pilot30_txt = out_dir / "stage12_dev_pilot30_ids.txt"
    pilot60_txt = out_dir / "stage12_dev_pilot60_ids.txt"
    strata_csv = out_dir / "stage12_dev_strata_table.csv"
    readme = out_dir / "README.md"

    write_jsonl(ds_jsonl, out_rows)
    pd.DataFrame(out_rows).to_csv(eval_csv, index=False)
    pilot30_txt.write_text("\n".join(selected30) + "\n", encoding="utf-8")
    pilot60_txt.write_text("\n".join(selected60) + "\n", encoding="utf-8")
    strata[
        [
            "record_id",
            "label_coarse_class",
            "label_visibility",
            "label_needs_review",
            "dino_top1",
            "dino_top2",
            "dino_margin",
            "dino_top1_correct",
            "margin_bucket",
            "pair",
            "is_error",
            "is_nonclear",
            "is_review",
            "is_non_ok",
        ]
    ].to_csv(strata_csv, index=False)

    readme.write_text(
        "\n".join(
            [
                "# Stage12 Dev Evidence Dataset",
                "",
                "- Historical `train` split is used as development split.",
                "- Historical `val` split is excluded from this dataset.",
                "- Image paths are prompt-safe and verified to exist.",
                "- Pilot30 selection: hard-first (errors/non-clear/review/non-OK) + stratified controls.",
                "- Pilot60 selection: Pilot30 + additional hard/minority + margin-bucket stratified fill.",
                "",
                f"- n_dev_rows: {len(out_rows)}",
                f"- n_pilot30: {len(selected30)}",
                f"- n_pilot60: {len(selected60)}",
                "",
                "## Files",
                "- stage12_dev_evidence_dataset.jsonl",
                "- stage12_dev_eval_reference.csv",
                "- stage12_dev_pilot30_ids.txt",
                "- stage12_dev_pilot60_ids.txt",
                "- stage12_dev_strata_table.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {ds_jsonl}")
    print(f"Wrote: {eval_csv}")
    print(f"Wrote: {pilot30_txt}")
    print(f"Wrote: {pilot60_txt}")
    print(f"Wrote: {strata_csv}")
    print(f"Wrote: {readme}")


if __name__ == "__main__":
    main()

