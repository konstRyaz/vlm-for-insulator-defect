import argparse
import math
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ID_CANDIDATES = ["id", "sample_id", "record_id", "image_id", "crop_id"]
IMAGE_CANDIDATES = ["image_path", "crop_path", "path", "filename"]
TRUE_CANDIDATES = ["gt", "y_true", "true_class", "label", "ground_truth"]
TOP1_CANDIDATES = ["pred", "y_pred", "pred_class", "dino_pred", "predicted_class", "pred_coarse_class", "top1_class"]
TOP2_CANDIDATES = ["top2_class"]
TOP3_CANDIDATES = ["top3_class"]
TOP1_SCORE_CANDIDATES = ["confidence", "top1_score"]
TOP2_SCORE_CANDIDATES = ["top2_confidence", "top2_score"]
TOP3_SCORE_CANDIDATES = ["top3_confidence", "top3_score"]
MARGIN_CANDIDATES = ["margin"]

BUCKET_DESCRIPTIONS = {
    "recoverable_top2_error": "Top-1 is wrong and the ground-truth class is top-2.",
    "recoverable_top3_error": "Top-1 is wrong and the ground-truth class is top-3 but not top-2.",
    "hard_correct_low_margin": "Top-1 is correct but the classifier margin is low.",
    "hard_wrong_low_margin": "Top-1 is wrong and the classifier margin is low.",
    "confident_wrong": "Top-1 is wrong despite a high classifier margin.",
    "flashover_risk": "Flashover appears in top candidates, especially as a false positive risk.",
    "broken_risk": "Broken-defect appears in top candidates, especially as a false positive risk.",
    "ok_vs_defect_confusion": "Top candidates include both normal-insulator and defect classes.",
    "easy_control_correct": "Top-1 is correct and the classifier margin is high.",
}

BUCKET_ORDER = list(BUCKET_DESCRIPTIONS)


def infer_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        cl = str(col).lower()
        if any(cand.lower() in cl for cand in candidates):
            return col
    return None


def optional_str(df, col):
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def optional_float(df, col):
    if col is None:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def add_topk_from_probability_columns(df):
    prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
    if not prob_cols:
        return df
    out = df.copy()
    probs = out[prob_cols].apply(pd.to_numeric, errors="coerce")
    for i, row in probs.iterrows():
        ranked = sorted(
            [(str(c)[5:], float(v)) for c, v in row.items() if pd.notna(v)],
            key=lambda x: x[1],
            reverse=True,
        )
        for rank in range(3):
            if len(ranked) > rank:
                out.loc[i, f"top{rank + 1}_class"] = ranked[rank][0]
                out.loc[i, f"top{rank + 1}_score"] = ranked[rank][1]
    if "pred_coarse_class" in out.columns and "top1_class" not in df.columns:
        out["top1_class"] = out["pred_coarse_class"]
    return out


def resolve_image_path(raw, images_root):
    if not raw:
        return ""
    p = Path(str(raw))
    if p.exists():
        return str(p)
    root = Path(images_root) if images_root else None
    if root is not None:
        direct = root / p
        if direct.exists():
            return str(direct)
        by_name = root / p.name
        if by_name.exists():
            return str(by_name)
    return str(p if p.is_absolute() else (root / p if root else p))


def has_defect(classes):
    return any(str(c).startswith("defect_") for c in classes if str(c))


def has_class(classes, target):
    return any(str(c) == target for c in classes)


def build_normalized_frame(df, images_root):
    cols = {
        "id": infer_col(df, ID_CANDIDATES),
        "image": infer_col(df, IMAGE_CANDIDATES),
        "gt": infer_col(df, TRUE_CANDIDATES),
        "top1": infer_col(df, TOP1_CANDIDATES),
        "top2": infer_col(df, TOP2_CANDIDATES),
        "top3": infer_col(df, TOP3_CANDIDATES),
        "top1_score": infer_col(df, TOP1_SCORE_CANDIDATES),
        "top2_score": infer_col(df, TOP2_SCORE_CANDIDATES),
        "top3_score": infer_col(df, TOP3_SCORE_CANDIDATES),
        "margin": infer_col(df, MARGIN_CANDIDATES),
    }
    if cols["top1"] is None:
        raise ValueError(f"Could not infer top-1 prediction column from {list(df.columns)}")

    norm = pd.DataFrame(index=df.index)
    norm["source_id"] = optional_str(df, cols["id"])
    norm.loc[norm["source_id"] == "", "source_id"] = [f"row_{i}" for i in norm.index[norm["source_id"] == ""]]
    raw_image = optional_str(df, cols["image"])
    norm["image_path"] = [resolve_image_path(x, images_root) for x in raw_image]
    norm["gt_class"] = optional_str(df, cols["gt"])
    norm["dino_top1"] = optional_str(df, cols["top1"])
    norm["dino_top2"] = optional_str(df, cols["top2"])
    norm["dino_top3"] = optional_str(df, cols["top3"])
    norm["top1_score"] = optional_float(df, cols["top1_score"])
    norm["top2_score"] = optional_float(df, cols["top2_score"])
    norm["top3_score"] = optional_float(df, cols["top3_score"])
    margin = optional_float(df, cols["margin"])
    if margin.isna().all() and not norm["top1_score"].isna().all() and not norm["top2_score"].isna().all():
        margin = norm["top1_score"] - norm["top2_score"]
    norm["margin"] = margin

    norm["dino_top1_correct"] = (norm["gt_class"] != "") & (norm["gt_class"] == norm["dino_top1"])
    norm["gt_in_top2"] = (norm["gt_class"] != "") & (
        (norm["gt_class"] == norm["dino_top1"]) | (norm["gt_class"] == norm["dino_top2"])
    )
    norm["gt_in_top3"] = norm["gt_in_top2"] | (
        (norm["gt_class"] != "") & (norm["gt_class"] == norm["dino_top3"])
    )
    return norm, cols


def margin_thresholds(norm):
    margins = norm["margin"].dropna()
    if margins.empty:
        return math.nan, math.nan
    return float(margins.quantile(0.25)), float(margins.quantile(0.75))


def matching_buckets(row, low_thr, high_thr):
    buckets = []
    top1_wrong = bool(row["gt_class"]) and not row["dino_top1_correct"]
    candidates = [row["dino_top1"], row["dino_top2"], row["dino_top3"]]
    margin = row["margin"]
    low_margin = pd.notna(margin) and (margin <= low_thr)
    high_margin = pd.notna(margin) and (margin >= high_thr)

    if top1_wrong and row["gt_class"] == row["dino_top2"]:
        buckets.append("recoverable_top2_error")
    if top1_wrong and row["gt_class"] == row["dino_top3"] and row["gt_class"] != row["dino_top2"]:
        buckets.append("recoverable_top3_error")
    if row["dino_top1_correct"] and low_margin:
        buckets.append("hard_correct_low_margin")
    if top1_wrong and low_margin:
        buckets.append("hard_wrong_low_margin")
    if top1_wrong and high_margin:
        buckets.append("confident_wrong")
    if has_class(candidates[:2], "defect_flashover"):
        buckets.append("flashover_risk")
    if has_class(candidates[:2], "defect_broken"):
        buckets.append("broken_risk")
    if has_class(candidates, "insulator_ok") and has_defect(candidates):
        buckets.append("ok_vs_defect_confusion")
    if row["dino_top1_correct"] and high_margin:
        buckets.append("easy_control_correct")
    return buckets


def priority(row, bucket):
    margin = row["margin"]
    margin = float(margin) if pd.notna(margin) else 0.0
    top1_wrong = bool(row["gt_class"]) and not row["dino_top1_correct"]
    false_risk = top1_wrong or (
        bucket == "flashover_risk" and row["gt_class"] != "defect_flashover"
    ) or (
        bucket == "broken_risk" and row["gt_class"] != "defect_broken"
    )
    if bucket in {"recoverable_top2_error", "recoverable_top3_error", "hard_correct_low_margin", "hard_wrong_low_margin"}:
        return -margin
    if bucket == "confident_wrong":
        return margin
    if bucket in {"flashover_risk", "broken_risk", "ok_vs_defect_confusion"}:
        return (100.0 if false_risk else 0.0) - margin
    if bucket == "easy_control_correct":
        return margin
    return 0.0


def make_contact_sheet(rows, out_path, warnings, thumb_size=(180, 180), cols=5):
    images = []
    for _, row in rows.iterrows():
        path = Path(str(row["image_path"]))
        if not path.exists():
            warnings.append(f"missing image for {row['candidate_id']}: {path}")
            continue
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail(thumb_size)
            tile = Image.new("RGB", (thumb_size[0], thumb_size[1] + 70), "white")
            x = (thumb_size[0] - im.width) // 2
            tile.paste(im, (x, 0))
            draw = ImageDraw.Draw(tile)
            label = [
                str(row["candidate_id"]),
                f"gt: {row['gt_class']}",
                f"t1/t2: {row['dino_top1']} / {row['dino_top2']}",
                f"m: {row['margin']:.3f}" if pd.notna(row["margin"]) else "m: NA",
            ]
            y = thumb_size[1] + 2
            for line in label:
                draw.text((4, y), line[:32], fill="black", font=ImageFont.load_default())
                y += 15
            images.append(tile)
        except Exception as exc:
            warnings.append(f"could not render {row['candidate_id']}: {path} ({exc})")
    if not images:
        return False
    rows_n = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows_n * (thumb_size[1] + 70)), "white")
    for i, im in enumerate(images):
        x = (i % cols) * thumb_size[0]
        y = (i // cols) * (thumb_size[1] + 70)
        sheet.paste(im, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return True


def write_readme(out, input_path, selected, bucket_summary, warnings):
    warning_block = "\n".join(f"- {w}" for w in warnings[:100]) if warnings else "- none"
    bucket_lines = ["| bucket | description |", "|---|---|"]
    for _, row in bucket_summary[["bucket", "description"]].drop_duplicates().iterrows():
        bucket_lines.append(f"| {row['bucket']} | {row['description']} |")
    bucket_table = "\n".join(bucket_lines)
    text = f"""# Stage 10 Hard-review Candidate Mining

This directory contains candidate crops for a hard-review benchmark for VLM-like
checkers and constrained top-k rerankers.

This is not a natural test split. It is a mined review set designed to stress
cases where a VLM may help as a top-k checker/reranker or as a safety/review
checker. Depending on the source predictions, this mined set may be used as
`hard_dev` for method development or `hard_test` for a held-out evaluation.

Input predictions: `{input_path}`

Selected candidates: {len(selected)}

## Annotation fields

The `annotation_*` columns are intentionally left blank for manual review.
Recommended usage:

- `annotation_gt_class`: final human class label.
- `annotation_crop_quality`: one of good, partial, bad, background, uncertain.
- `annotation_visible_insulator`: whether the crop contains a visible insulator.
- `annotation_ambiguous`: whether the class is visually ambiguous.
- `annotation_review_required`: whether the case should be escalated.
- `annotation_notes`: short free-text note.

The initial `suggested_annotation_status` is `needs_manual_annotation`.

## Buckets

{bucket_table}

## Later evaluation

After manual annotation, the set can evaluate:

- constrained top-2/top-3 VLM reranking;
- VLM keep/switch behavior against DINOv2 top-k candidates;
- safety/review behavior on low-margin and bad-crop-like cases;
- false positive risk around flashover and broken defects.

Do not use hard-test labels for prompt tuning. If prompts or policies are tuned
on this mined set, call it hard-dev and reserve a separate hard-test set.

## Contact sheet warnings

{warning_block}
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-bucket", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_topk_from_probability_columns(pd.read_csv(args.predictions_csv))
    norm, inferred_cols = build_normalized_frame(df, args.images_root)
    low_thr, high_thr = margin_thresholds(norm)

    all_buckets = []
    for idx, row in norm.iterrows():
        buckets = matching_buckets(row, low_thr, high_thr)
        all_buckets.append(buckets)
    norm["all_buckets"] = [";".join(x) for x in all_buckets]

    selected_source_ids = set()
    selected_records = []
    summary_rows = []
    for bucket in BUCKET_ORDER:
        candidates = [i for i, bs in zip(norm.index, all_buckets) if bucket in bs]
        ranked = sorted(candidates, key=lambda i: (priority(norm.loc[i], bucket), random.random()), reverse=True)
        chosen = []
        for idx in ranked:
            source_id = str(norm.loc[idx, "source_id"])
            if source_id in selected_source_ids:
                continue
            chosen.append(idx)
            selected_source_ids.add(source_id)
            if len(chosen) >= args.max_per_bucket:
                break
        for idx in chosen:
            row = norm.loc[idx].copy()
            row["bucket"] = bucket
            row["priority_score"] = priority(row, bucket)
            selected_records.append(row)
        summary_rows.append(
            {
                "bucket": bucket,
                "n_candidates": len(candidates),
                "n_selected": len(chosen),
                "description": BUCKET_DESCRIPTIONS[bucket],
            }
        )

    selected = pd.DataFrame(selected_records)
    if selected.empty:
        selected = pd.DataFrame(columns=list(norm.columns) + ["bucket", "priority_score"])
    selected = selected.reset_index(drop=True)
    selected["candidate_id"] = [f"stage10_{i:04d}" for i in range(1, len(selected) + 1)]
    selected["additional_buckets"] = [
        ";".join([b for b in str(all_buckets).split(";") if b and b != bucket])
        for all_buckets, bucket in zip(selected["all_buckets"].fillna(""), selected["bucket"].fillna(""))
    ]
    for col in [
        "suggested_annotation_status",
        "annotation_gt_class",
        "annotation_crop_quality",
        "annotation_visible_insulator",
        "annotation_ambiguous",
        "annotation_review_required",
        "annotation_notes",
    ]:
        selected[col] = ""
    selected["suggested_annotation_status"] = "needs_manual_annotation"

    out_cols = [
        "candidate_id",
        "source_id",
        "image_path",
        "gt_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "top1_score",
        "top2_score",
        "top3_score",
        "margin",
        "dino_top1_correct",
        "gt_in_top2",
        "gt_in_top3",
        "bucket",
        "additional_buckets",
        "priority_score",
        "suggested_annotation_status",
        "annotation_gt_class",
        "annotation_crop_quality",
        "annotation_visible_insulator",
        "annotation_ambiguous",
        "annotation_review_required",
        "annotation_notes",
    ]
    selected[out_cols].to_csv(out / "hard_review_candidates.csv", index=False)

    bucket_summary = pd.DataFrame(summary_rows)
    bucket_summary.to_csv(out / "bucket_summary.csv", index=False)

    warnings = []
    contact_dir = out / "contact_sheets"
    for bucket in BUCKET_ORDER:
        rows = selected[selected["bucket"] == bucket]
        if rows.empty:
            continue
        make_contact_sheet(rows, contact_dir / f"{bucket}.png", warnings)

    metadata = pd.DataFrame(
        [{"field": k, "inferred_column": v or ""} for k, v in inferred_cols.items()]
        + [{"field": "low_margin_threshold", "inferred_column": low_thr}, {"field": "high_margin_threshold", "inferred_column": high_thr}]
    )
    metadata.to_csv(out / "column_inference_and_thresholds.csv", index=False)
    write_readme(out, args.predictions_csv, selected, bucket_summary, warnings)

    print("[saved]", out)
    print("selected:", len(selected))
    print("bucket summary:", out / "bucket_summary.csv")


if __name__ == "__main__":
    main()
