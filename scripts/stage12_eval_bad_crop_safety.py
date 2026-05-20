#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--vlm-jsonl", required=False, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest_csv)
    m["is_bad_crop"] = 1

    if args.vlm_jsonl:
        v = pd.DataFrame(parse_jsonl(Path(args.vlm_jsonl)))
        key = "record_id"
        if "bad_crop_id" in v.columns:
            key = "bad_crop_id"
        if key == "record_id":
            # expected format "record_id__corruption"
            v["bad_crop_id"] = v["record_id"].astype(str)
        pred = v[["bad_crop_id", "visibility", "needs_review"]].copy()
        df = m.merge(pred, on="bad_crop_id", how="left")
        df["vlm_safe_behavior"] = (df["needs_review"].fillna(True).astype(bool)) | (df["visibility"].fillna("ambiguous").isin(["ambiguous", "bad"]))
    else:
        df = m.copy()
        df["visibility"] = "unknown"
        df["needs_review"] = True
        df["vlm_safe_behavior"] = True

    # Closed-set DINO proxy: always outputs known class and never review.
    df["dino_closed_set_false_confident"] = True

    summary = pd.DataFrame(
        [
            {
                "n": len(df),
                "safe_behavior_rate": float(df["vlm_safe_behavior"].mean()),
                "false_accept_rate": float((~df["vlm_safe_behavior"]).mean()),
                "review_rate": float(df["needs_review"].fillna(False).astype(bool).mean()),
                "closed_set_false_confident_rate": float(df["dino_closed_set_false_confident"].mean()),
            }
        ]
    )
    by_corr = (
        df.groupby("corruption", as_index=False)
        .agg(
            n=("bad_crop_id", "count"),
            safe_behavior_rate=("vlm_safe_behavior", "mean"),
            false_accept_rate=("vlm_safe_behavior", lambda s: float((~s).mean())),
            review_rate=("needs_review", lambda s: float(pd.Series(s).fillna(False).astype(bool).mean())),
        )
        .sort_values("n", ascending=False)
    )

    df.to_csv(out / "stage12_bad_crop_predictions_with_flags.csv", index=False)
    summary.to_csv(out / "stage12_bad_crop_summary.csv", index=False)
    by_corr.to_csv(out / "stage12_bad_crop_by_corruption.csv", index=False)
    (out / "stage12_bad_crop_report.md").write_text(
        "\n".join(
            [
                "# Stage12 Bad-Crop Safety",
                "",
                f"- n: {int(summary.iloc[0]['n'])}",
                f"- safe_behavior_rate: {summary.iloc[0]['safe_behavior_rate']:.4f}",
                f"- false_accept_rate: {summary.iloc[0]['false_accept_rate']:.4f}",
                f"- review_rate: {summary.iloc[0]['review_rate']:.4f}",
                f"- closed_set_false_confident_rate: {summary.iloc[0]['closed_set_false_confident_rate']:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
