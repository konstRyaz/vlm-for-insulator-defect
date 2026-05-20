#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-table", required=True)
    ap.add_argument("--stage12-dir", required=True)
    ap.add_argument("--stage13-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.full_table)
    s12 = Path(args.stage12_dir)
    s13 = Path(args.stage13_dir)

    # Primary feature source (already merged labels + dino + vlm probe features)
    risk_features_path = s12 / "risk_models" / "dev_risk_features.csv"
    if risk_features_path.exists():
        rf = pd.read_csv(risk_features_path)
        key_cols = ["record_id"]
        merge_cols = [c for c in rf.columns if c not in full.columns or c == "record_id"]
        full = full.merge(rf[merge_cols], on=key_cols, how="left", suffixes=("", "_rf"))

    # Add claim-checker signal if present
    e03_out = s13 / "E03_flashover_overclaim_checker" / "flashover_claim_outputs.jsonl"
    if e03_out.exists():
        e03 = pd.read_json(e03_out, lines=True)
        if "record_id" in e03.columns:
            keep = [
                "record_id",
                "claim_supported",
                "recommend_review",
                "direct_flashover_evidence",
                "parse_ok",
            ]
            keep = [c for c in keep if c in e03.columns]
            full = full.merge(e03[keep], on="record_id", how="left")

    # Add claim verification signal if present
    e05_out = s13 / "E05_claim_verification" / "claim_verification_outputs.jsonl"
    if e05_out.exists():
        e05 = pd.read_json(e05_out, lines=True)
        if "record_id" in e05.columns:
            tmp = e05.copy()
            if "verification" in tmp.columns:
                grp = (
                    tmp.groupby("record_id")["verification"]
                    .agg(lambda s: int((s == "contradicted").any()))
                    .rename("claim_has_contradicted")
                    .reset_index()
                )
                full = full.merge(grp, on="record_id", how="left")

    # Add crop failure features if present
    e04_feat = s13 / "E04_detector_vlm_crop_failure_guard" / "crop_failure_features.csv"
    if e04_feat.exists():
        cf = pd.read_csv(e04_feat)
        if "record_id" in cf.columns:
            keep = [c for c in cf.columns if c in {"record_id", "target_crop_failure", "src_is_pred", "visibility", "needs_review"}]
            full = full.merge(cf[keep].drop_duplicates("record_id"), on="record_id", how="left")

    # Normalize key booleans/targets
    full["general_error"] = (full["dino_top1"].astype(str) != full["label_coarse_class"].astype(str)).astype(int)
    full["false_alarm"] = (
        (full["label_coarse_class"].astype(str) == "insulator_ok")
        & (full["dino_top1"].astype(str) != "insulator_ok")
    ).astype(int)
    full["ok_to_flashover_false_alarm"] = (
        (full["label_coarse_class"].astype(str) == "insulator_ok")
        & (full["dino_top1"].astype(str) == "defect_flashover")
    ).astype(int)
    full["dangerous_miss"] = (
        (full["label_coarse_class"].astype(str) != "insulator_ok")
        & (full["dino_top1"].astype(str) == "insulator_ok")
    ).astype(int)
    full["defect_vs_ok"] = (full["label_coarse_class"].astype(str) != "insulator_ok").astype(int)
    full["defect_type_confusion"] = (
        (full["label_coarse_class"].astype(str).isin(["defect_broken", "defect_flashover"]))
        & (full["dino_top1"].astype(str).isin(["defect_broken", "defect_flashover"]))
        & (full["label_coarse_class"].astype(str) != full["dino_top1"].astype(str))
    ).astype(int)

    # Fill obvious missing fields
    for col, val in [
        ("vlm_needs_review", 0),
        ("vlm_has_broken_structure", 0),
        ("vlm_has_flashover_surface", 0),
        ("vlm_has_quality_or_confounder", 0),
        ("class_evidence_consistency", 0),
        ("entropy", 0.0),
        ("dino_margin", 0.0),
        ("dino_top1_score", 0.0),
        ("dino_top2_score", 0.0),
        ("dino_top3_score", 0.0),
        ("claim_has_contradicted", 0),
    ]:
        if col not in full.columns:
            full[col] = val
        full[col] = full[col].fillna(val)

    out_master = out_dir / "stage14_resplit_master_table.csv"
    full.to_csv(out_master, index=False)

    inv = pd.DataFrame(
        [{"column": c, "dtype": str(full[c].dtype), "non_null": int(full[c].notna().sum())} for c in full.columns]
    )
    inv.to_csv(out_dir / "stage14_feature_inventory.csv", index=False)

    miss = pd.DataFrame(
        [{"column": c, "missing_count": int(full[c].isna().sum()), "missing_rate": float(full[c].isna().mean())} for c in full.columns]
    ).sort_values("missing_count", ascending=False)
    miss.to_csv(out_dir / "stage14_missing_features.csv", index=False)


if __name__ == "__main__":
    main()
