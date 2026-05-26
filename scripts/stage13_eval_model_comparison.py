#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def eval_e03(dirp: Path, model_name: str) -> pd.DataFrame:
    m = dirp / "flashover_claim_manifest.csv"
    o = dirp / "flashover_claim_outputs.jsonl"
    if not (m.exists() and o.exists()):
        return pd.DataFrame()
    man = pd.read_csv(m)
    out = pd.read_json(o, lines=True)
    df = man.merge(out, on="record_id", how="left")
    if "parse_ok" not in df.columns:
        df["parse_ok"] = False
    else:
        df["parse_ok"] = df["parse_ok"].fillna(False).astype(bool)
    if "claim_supported" not in df.columns:
        df["claim_supported"] = "uncertain"
    else:
        df["claim_supported"] = df["claim_supported"].fillna("uncertain").astype(str)
    if "recommend_review" not in df.columns:
        df["recommend_review"] = True
    else:
        df["recommend_review"] = df["recommend_review"].fillna(True).astype(bool)
    cs = df["claim_supported"]
    rr = df["recommend_review"]
    is_fa = (df["label_coarse_class"] == "insulator_ok") & (df["dino_top1"] == "defect_flashover")
    is_tf = (df["label_coarse_class"] == "defect_flashover") & (df["dino_top1"] == "defect_flashover")
    caught = is_fa & ((cs != "yes") | rr)
    hurt = is_tf & ((cs != "yes") | rr)
    return pd.DataFrame([{
        "model": model_name,
        "task": "E03_flashover_checker",
        "parse_ok_rate": float(df["parse_ok"].mean()),
        "false_alarm_capture_rate": float(caught.sum() / max(1, is_fa.sum())),
        "true_flashover_hurt_rate": float(hurt.sum() / max(1, is_tf.sum())),
        "review_rate": float(rr.mean()),
    }])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="Directory with subfolders per model run")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E07_model_comparison")
    args = ap.parse_args()

    runs = Path(args.runs_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        rows.append(eval_e03(d, d.name))
    comp = pd.concat([r for r in rows if len(r)], ignore_index=True) if rows else pd.DataFrame()
    comp.to_csv(out / "model_comparison_metrics.csv", index=False)
    (out / "model_comparison_report.md").write_text(
        "# E07 Model Comparison\n\nReal runs comparison generated from provided model run folders.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
