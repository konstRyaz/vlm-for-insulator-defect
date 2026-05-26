#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts.stage15_manifest_path_utils import apply_resolve, build_image_index
except Exception:
    from stage15_manifest_path_utils import apply_resolve, build_image_index


def claim_for_row(dino_top1: str) -> tuple[str, str]:
    if dino_top1 == "defect_flashover":
        return "flashover_surface_evidence", "Visible burn/arc/dark trace is attached to insulator surface."
    if dino_top1 == "defect_broken":
        return "broken_structure_evidence", "Visible missing fragment/crack/edge discontinuity exists."
    return "ok_intact_evidence", "Visible insulator appears intact with no direct defect evidence."


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}. Available: {list(df.columns)}")
    return ""


def _ensure_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    split_col = _pick_col(out, ["split", "dataset_split", "subset"])
    path_col = _pick_col(out, ["resolved_image_path", "image_path", "crop_path", "path"])
    label_col = _pick_col(out, ["label_coarse_class", "coarse_class", "label", "gt_class", "true_class"])
    top1_col = _pick_col(out, ["dino_top1", "top1_class", "pred_class", "dino_pred"])
    top2_col = _pick_col(out, ["dino_top2", "top2_class"], required=False)
    top3_col = _pick_col(out, ["dino_top3", "top3_class"], required=False)
    s1_col = _pick_col(out, ["dino_top1_score", "top1_score", "confidence"], required=False)
    s2_col = _pick_col(out, ["dino_top2_score", "top2_score"], required=False)
    s3_col = _pick_col(out, ["dino_top3_score", "top3_score"], required=False)
    margin_col = _pick_col(out, ["dino_margin", "margin"], required=False)
    if "record_id" not in out.columns:
        out["record_id"] = out.index.astype(str)

    out["split"] = out[split_col].astype(str).str.lower().replace({"dev": "train", "development": "train", "test": "val"})
    out["resolved_image_path"] = out[path_col].astype(str)
    out["label_coarse_class"] = out[label_col].astype(str)
    out["dino_top1"] = out[top1_col].astype(str)
    out["dino_top2"] = out[top2_col].astype(str) if top2_col else ""
    out["dino_top3"] = out[top3_col].astype(str) if top3_col else ""
    out["dino_top1_score"] = pd.to_numeric(out[s1_col], errors="coerce") if s1_col else 0.0
    out["dino_top2_score"] = pd.to_numeric(out[s2_col], errors="coerce") if s2_col else 0.0
    out["dino_top3_score"] = pd.to_numeric(out[s3_col], errors="coerce") if s3_col else 0.0
    out["dino_margin"] = pd.to_numeric(out[margin_col], errors="coerce") if margin_col else (out["dino_top1_score"] - out["dino_top2_score"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-root", default="outputs/stage15_development_value")
    ap.add_argument("--limit-e02", type=int, default=120)
    ap.add_argument("--limit-e05", type=int, default=180)
    ap.add_argument("--limit-e06", type=int, default=80)
    args = ap.parse_args()

    ref = pd.read_csv(args.reference_csv)
    ref = _ensure_common_schema(ref)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    idx = build_image_index([Path(args.images_root)])

    # Normalize resolved paths
    ref = apply_resolve(ref, "resolved_image_path", idx)

    # E02
    e02 = out_root / "E02_flashover_overclaim_checker"
    e02.mkdir(parents=True, exist_ok=True)
    test = ref[ref["split"] == "val"].copy()
    fa = test[(test["dino_top1"] == "defect_flashover") & (test["label_coarse_class"] == "insulator_ok")]
    tp = test[(test["dino_top1"] == "defect_flashover") & (test["label_coarse_class"] == "defect_flashover")]
    ctrl = test[(test["dino_top2"] == "defect_flashover") & (test["dino_top1"] != "defect_flashover")]
    n = args.limit_e02
    n_fa = min(len(fa), int(n * 0.45))
    n_tp = min(len(tp), int(n * 0.35))
    n_ctrl = max(0, n - n_fa - n_tp)
    m2 = pd.concat([fa.head(n_fa), tp.head(n_tp), ctrl.head(n_ctrl)], ignore_index=True).drop_duplicates("record_id").head(n).copy()
    m2["task"] = "flashover_claim_check"
    keep2 = [
        "record_id",
        "split",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
        "task",
    ]
    m2[keep2].to_csv(e02 / "flashover_checker_manifest.csv", index=False)

    # E05
    e05 = out_root / "E05_claim_verification"
    e05.mkdir(parents=True, exist_ok=True)
    dev = ref[ref["split"] == "train"].copy()
    hard = dev[(dev["dino_top1"] != dev["label_coarse_class"]) | (dev["dino_margin"] < 0.2)]
    ctrl = dev[(dev["dino_top1"] == dev["label_coarse_class"]) & (dev["dino_margin"] > 0.5)]
    n = args.limit_e05
    nh = min(len(hard), int(n * 0.65))
    nc = max(0, n - nh)
    m5 = pd.concat([hard.head(nh), ctrl.head(nc)], ignore_index=True).drop_duplicates("record_id").head(n).copy()
    m5["claim_id"] = m5["dino_top1"].map(lambda x: claim_for_row(str(x))[0])
    m5["claim_text"] = m5["dino_top1"].map(lambda x: claim_for_row(str(x))[1])
    keep5 = [
        "record_id",
        "split",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_margin",
        "claim_id",
        "claim_text",
    ]
    m5[keep5].to_csv(e05 / "claim_manifest.csv", index=False)

    # E06
    e06 = out_root / "E06_multiview_asset_packet"
    e06.mkdir(parents=True, exist_ok=True)
    m6 = ref[ref["split"] == "val"].copy().head(args.limit_e06)
    zdir = e06 / "zoom_views"
    rows6 = []
    from PIL import Image  # local import

    for _, r in m6.iterrows():
        p = Path(str(r["resolved_image_path"]))
        zp = zdir / f"{r['record_id']}_zoom.jpg"
        if p.exists():
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                x0, y0, x1, y1 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
                crop = im.crop((x0, y0, x1, y1)).resize((w, h), Image.Resampling.BICUBIC)
                zp.parent.mkdir(parents=True, exist_ok=True)
                crop.save(zp)
        rows6.append(
            {
                "record_id": r["record_id"],
                "split": r["split"],
                "image_main": str(p),
                "image_zoom": str(zp),
                "label_coarse_class": r["label_coarse_class"],
                "dino_top1": r["dino_top1"],
                "dino_top2": r["dino_top2"],
                "dino_top3": r["dino_top3"],
            }
        )
    pd.DataFrame(rows6).to_csv(e06 / "multiview_manifest.csv", index=False)

    # preflight summary
    pre = []
    for name, col, d in [
        ("E02", "resolved_image_path", m2),
        ("E05", "resolved_image_path", m5),
        ("E06", "image_main", pd.DataFrame(rows6)),
    ]:
        s = d[col].astype(str).map(lambda x: Path(x).exists())
        pre.append({"experiment": name, "n": int(len(s)), "path_exists_rate": float(s.mean()) if len(s) else 0.0})
    pd.DataFrame(pre).to_csv(out_root / "path_preflight_summary.csv", index=False)
    print("manifests built")


if __name__ == "__main__":
    main()
