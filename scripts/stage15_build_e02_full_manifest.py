#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _first_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str:
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise KeyError(f"None of columns found: {names}")
    return ""


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "record_id" not in out.columns:
        out["record_id"] = out.index.astype(str)

    split_col = _first_col(out, ["split", "dataset_split", "subset"], required=False)
    if split_col:
        out["split"] = out[split_col].astype(str).str.lower()
    else:
        out["split"] = "unknown"

    out["split"] = out["split"].replace(
        {"development": "train", "dev": "train", "test": "val", "validation": "val"}
    )

    path_col = _first_col(out, ["resolved_image_path", "crop_path", "image_path", "path"])
    label_col = _first_col(out, ["label_coarse_class", "coarse_class", "label", "gt_class", "true_class"])
    d1_col = _first_col(out, ["dino_top1", "top1_class", "pred_class", "dino_pred"])
    d2_col = _first_col(out, ["dino_top2", "top2_class"], required=False)
    d3_col = _first_col(out, ["dino_top3", "top3_class"], required=False)
    s1_col = _first_col(out, ["dino_top1_score", "top1_score", "confidence"], required=False)
    s2_col = _first_col(out, ["dino_top2_score", "top2_score"], required=False)
    margin_col = _first_col(out, ["dino_margin", "margin"], required=False)

    out["resolved_image_path"] = out[path_col].astype(str).str.replace("\\", "/", regex=False)
    out["label_coarse_class"] = out[label_col].astype(str)
    out["dino_top1"] = out[d1_col].astype(str)
    out["dino_top2"] = out[d2_col].astype(str) if d2_col else ""
    out["dino_top3"] = out[d3_col].astype(str) if d3_col else ""
    out["dino_top1_score"] = pd.to_numeric(out[s1_col], errors="coerce") if s1_col else 0.0
    out["dino_top2_score"] = pd.to_numeric(out[s2_col], errors="coerce") if s2_col else 0.0
    out["dino_margin"] = (
        pd.to_numeric(out[margin_col], errors="coerce")
        if margin_col
        else (out["dino_top1_score"] - out["dino_top2_score"])
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument(
        "--out-csv",
        default="outputs/stage15_development_value/E02_flashover_overclaim_checker/full_manifest.csv",
    )
    ap.add_argument(
        "--preflight-csv",
        default="outputs/stage15_development_value/E02_flashover_overclaim_checker/run_full/path_preflight.csv",
    )
    ap.add_argument(
        "--preflight-summary-json",
        default="outputs/stage15_development_value/E02_flashover_overclaim_checker/run_full/path_preflight_summary.json",
    )
    ap.add_argument("--low-margin-threshold", type=float, default=0.2)
    ap.add_argument("--image-root", default="", help="Optional root to resolve relative image paths.")
    args = ap.parse_args()

    ref = pd.read_csv(args.reference_csv)
    ref = _normalize(ref)

    is_top1_flash = ref["dino_top1"] == "defect_flashover"
    is_false_alarm = (ref["label_coarse_class"] == "insulator_ok") & is_top1_flash
    is_true_flash = (ref["label_coarse_class"] == "defect_flashover") & is_top1_flash
    is_hard_control = (
        (ref["label_coarse_class"] == "insulator_ok")
        & (~is_top1_flash)
        & ((ref["dino_top2"] == "defect_flashover") | (ref["dino_top3"] == "defect_flashover"))
    )
    is_low_margin_flash = is_top1_flash & (ref["dino_margin"].fillna(1.0) <= args.low_margin_threshold)
    is_flash_like_confounder = is_top1_flash & ref["label_coarse_class"].isin(["insulator_ok", "defect_broken"])

    cand = ref[
        is_top1_flash | is_hard_control | is_low_margin_flash | is_flash_like_confounder
    ].copy()
    cand = cand.drop_duplicates("record_id")

    cand["e02_bucket"] = "top1_flashover"
    cand.loc[is_false_alarm.reindex(cand.index, fill_value=False), "e02_bucket"] = "ok_to_flashover_false_alarm"
    cand.loc[is_true_flash.reindex(cand.index, fill_value=False), "e02_bucket"] = "true_flashover_control"
    cand.loc[is_hard_control.reindex(cand.index, fill_value=False), "e02_bucket"] = "top2_top3_flashover_control"
    cand.loc[is_low_margin_flash.reindex(cand.index, fill_value=False), "e02_bucket"] = "low_margin_flashover"

    keep_cols = [
        "record_id",
        "split",
        "source",
        "image_id",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_margin",
        "e02_bucket",
    ]
    present = [c for c in keep_cols if c in cand.columns]
    out = cand[present].copy()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    image_root = Path(args.image_root) if args.image_root else None

    def _resolve(p: str, rid: str) -> Path:
        pp = Path(str(p))
        if pp.exists():
            return pp
        if image_root is not None:
            candidate = image_root / pp
            if candidate.exists():
                return candidate
            # common manifest bundle layout: <image_root>/images/<record_id>.jpg
            rid_name = Path(str(rid)).stem
            if rid_name:
                by_id = image_root / "images" / f"{rid_name}.jpg"
                if by_id.exists():
                    return by_id
            # fallback for paths like outputs/stage10/.../images/xxx.jpg when kaggle input root already points there
            try_name = image_root / pp.name
            if try_name.exists():
                return try_name
        return pp

    resolved = [
        _resolve(p, rid)
        for p, rid in zip(out["resolved_image_path"].astype(str), out["record_id"].astype(str))
    ]
    resolved = pd.Series(resolved, index=out.index)
    out["resolved_image_path"] = resolved.map(lambda p: str(p).replace("\\", "/"))
    exists = resolved.map(lambda p: p.exists())
    pf = out[["record_id", "resolved_image_path"]].copy()
    pf["exists"] = exists
    pf["error"] = pf["exists"].map(lambda x: "" if x else "missing_path")
    pf_path = Path(args.preflight_csv)
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(pf_path, index=False)

    summary = {
        "n_candidates": int(len(out)),
        "missing": int((~exists).sum()),
        "exists_rate": float(exists.mean()) if len(out) else 0.0,
        "false_alarm_count": int(((out["label_coarse_class"] == "insulator_ok") & (out["dino_top1"] == "defect_flashover")).sum()),
        "true_flashover_count": int(((out["label_coarse_class"] == "defect_flashover") & (out["dino_top1"] == "defect_flashover")).sum()),
        "bucket_counts": {k: int(v) for k, v in out["e02_bucket"].value_counts().to_dict().items()},
    }
    s_path = Path(args.preflight_summary_json)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    if summary["missing"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
