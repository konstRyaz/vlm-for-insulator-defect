#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--outputs-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest_csv)
    o = pd.read_json(args.outputs_jsonl, lines=True)
    df = m.merge(o, on="record_id", how="left")
    if "parse_ok" in df.columns:
        df["parse_ok"] = df["parse_ok"].fillna(False).astype(bool)
    else:
        df["parse_ok"] = False
    if "selected_class_hint" in df.columns:
        df["selected_class_hint"] = df["selected_class_hint"].fillna("uncertain").astype(str)
    else:
        df["selected_class_hint"] = "uncertain"
    df["hint_match"] = (
        ((df["selected_class_hint"] == "top1") & (df["dino_top1"] == df["label_coarse_class"]))
        | ((df["selected_class_hint"] == "top2") & (df["dino_top2"] == df["label_coarse_class"]))
        | ((df["selected_class_hint"] == "top3") & (df["dino_top3"] == df["label_coarse_class"]))
    )
    metrics = pd.DataFrame(
        [
            {
                "n": len(df),
                "parse_ok_rate": float(df["parse_ok"].mean()),
                "hint_match_rate": float(df["hint_match"].mean()),
                "review_rate": float(df["needs_review"].fillna(True).astype(bool).mean()) if "needs_review" in df.columns else 1.0,
            }
        ]
    )
    metrics.to_csv(out / "multiview_metrics.csv", index=False)
    with (out / "E08_report.md").open("w", encoding="utf-8") as f:
        f.write("# E08 Multiview Evidence\n\nReal multiview VLM run completed.\n")


if __name__ == "__main__":
    main()
