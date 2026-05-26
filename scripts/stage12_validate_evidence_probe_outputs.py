#!/usr/bin/env python3
"""Validate Stage 12 evidence probe outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_VISIBLE = {"yes", "no", "uncertain"}
ALLOWED_VISIBILITY = {"clear", "partial", "ambiguous", "bad"}
ALLOWED_TAGS = {
    "intact_structure",
    "regular_disc_shape",
    "missing_fragment",
    "edge_discontinuity",
    "burn_like_mark",
    "surface_stain",
    "ambiguous_evidence",
    "surface_damage_mark",
    "blurred_region",
    "partial_view",
    "dark_surface_trace",
    "unclear_boundary",
    "no_visible_break",
    "occluded_region",
    "low_contrast",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dist(df: pd.DataFrame, field: str) -> pd.DataFrame:
    vc = df[field].astype(str).value_counts(dropna=False)
    total = max(int(vc.sum()), 1)
    return pd.DataFrame(
        {"field": field, "value": vc.index.astype(str), "count": vc.values, "fraction": vc.values / total}
    )


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.outputs_jsonl))
    if not rows:
        raise ValueError("No rows in outputs.")
    df = pd.DataFrame(rows)
    if "record_id" not in df.columns:
        raise ValueError("outputs-jsonl must contain record_id")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(df)
    parse_ok_rate = float(df.get("parse_ok", False).astype(bool).mean())
    schema_ok_rate = float(df.get("schema_ok", False).astype(bool).mean())
    invalid_visible = int((~df["visible_insulator"].astype(str).isin(ALLOWED_VISIBLE)).sum()) if "visible_insulator" in df else n
    invalid_visibility = int((~df["visibility"].astype(str).isin(ALLOWED_VISIBILITY)).sum()) if "visibility" in df else n

    tags_all: list[str] = []
    tags_count = []
    oov_counter: Counter[str] = Counter()
    for v in df.get("evidence_tags", []):
        if isinstance(v, list):
            tags = [str(x).strip() for x in v if str(x).strip()]
        elif isinstance(v, str) and v.strip():
            tags = [v.strip()]
        else:
            tags = []
        tags_count.append(len(tags))
        tags_all.extend(tags)
        for t in tags:
            if t not in ALLOWED_TAGS:
                oov_counter[t] += 1

    oov_count = sum(oov_counter.values())
    oov_rate = float(oov_count / max(len(tags_all), 1))
    record_ids = df["record_id"].astype(str)
    dup_count = int(record_ids.duplicated().sum())
    missing_record_ids = int(record_ids.isna().sum() + (record_ids == "").sum())

    desc = df.get("short_canonical_description", pd.Series([""] * n)).astype(str).str.strip()
    non_empty_rate = float((desc != "").mean())
    non_empty_desc = desc[desc != ""]
    duplicate_desc_rate = float(non_empty_desc.duplicated().mean()) if len(non_empty_desc) else 0.0

    summary = pd.DataFrame(
        [
            {
                "n": n,
                "parse_ok_rate": parse_ok_rate,
                "schema_ok_rate": schema_ok_rate,
                "invalid_visible_insulator_count": invalid_visible,
                "invalid_visibility_count": invalid_visibility,
                "oov_tag_count": oov_count,
                "oov_tag_rate": oov_rate,
                "missing_record_ids": missing_record_ids,
                "duplicate_record_ids": dup_count,
                "avg_tags_per_sample": float(sum(tags_count) / max(len(tags_count), 1)),
                "description_non_empty_rate": non_empty_rate,
                "description_duplicate_rate": duplicate_desc_rate,
            }
        ]
    )

    dists = pd.concat(
        [
            dist(df.assign(visible_insulator=df.get("visible_insulator", "missing")), "visible_insulator"),
            dist(df.assign(visibility=df.get("visibility", "missing")), "visibility"),
            dist(df.assign(needs_review=df.get("needs_review", "missing")), "needs_review"),
        ],
        ignore_index=True,
    )

    tag_dist = pd.DataFrame(Counter(tags_all).most_common(), columns=["tag", "count"])
    if len(tag_dist):
        tag_dist["fraction"] = tag_dist["count"] / max(int(tag_dist["count"].sum()), 1)
    else:
        tag_dist = pd.DataFrame(columns=["tag", "count", "fraction"])
    tag_dist.insert(0, "field", "evidence_tags")
    dists = pd.concat(
        [dists, tag_dist.rename(columns={"tag": "value"})[["field", "value", "count", "fraction"]]], ignore_index=True
    )

    oov_df = pd.DataFrame(oov_counter.most_common(), columns=["oov_tag", "count"])
    if len(oov_df):
        oov_df["fraction_among_all_tags"] = oov_df["count"] / max(len(tags_all), 1)

    # Optional join for needs_review by dino_top1 class.
    by_dino = pd.DataFrame()
    if args.eval_reference_csv and Path(args.eval_reference_csv).exists():
        ref = pd.read_csv(args.eval_reference_csv)
        cols = ["record_id", "dino_top1"]
        if set(cols).issubset(set(ref.columns)):
            merged = df[["record_id", "needs_review"]].merge(ref[cols], on="record_id", how="left")
            by_dino = (
                merged.groupby("dino_top1", dropna=False)["needs_review"]
                .agg(n="count", needs_review_rate="mean")
                .reset_index()
            )
            by_dino["field"] = "needs_review_by_dino_top1"
            by_dino = by_dino.rename(columns={"dino_top1": "value", "n": "count", "needs_review_rate": "fraction"})[
                ["field", "value", "count", "fraction"]
            ]
            dists = pd.concat([dists, by_dino], ignore_index=True)

    summary_csv = out_dir / "stage12_vlm_evidence_probe_validation.csv"
    summary_md = out_dir / "stage12_vlm_evidence_probe_validation.md"
    oov_csv = out_dir / "stage12_vlm_evidence_probe_oov_tags.csv"
    dist_csv = out_dir / "stage12_vlm_evidence_probe_distribution.csv"

    summary.to_csv(summary_csv, index=False)
    oov_df.to_csv(oov_csv, index=False)
    dists.to_csv(dist_csv, index=False)

    lines = [
        "# Stage 12 Evidence Probe Validation",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for k, v in summary.iloc[0].to_dict().items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Distributions", "", "| field | value | count | fraction |", "| --- | --- | ---: | ---: |"])
    for r in dists.to_dict(orient="records"):
        lines.append(f"| {r['field']} | {r['value']} | {int(r['count'])} | {float(r['fraction']):.6f} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {oov_csv}")
    print(f"Wrote: {dist_csv}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-jsonl", required=True)
    p.add_argument("--eval-reference-csv", default=None)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()

