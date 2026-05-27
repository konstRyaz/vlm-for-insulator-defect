import argparse
import json
import math
from pathlib import Path

import pandas as pd


LABEL_FIELDS = [
    "record_id",
    "split",
    "source",
    "image_id",
    "box_id",
    "crop_path",
    "image_path",
    "bbox_xywh",
    "coarse_class",
    "visual_evidence_tags",
    "visibility",
    "needs_review",
    "short_canonical_description",
    "report_snippet",
    "category_name",
    "annotator_notes",
    "label_version",
]

ID_CANDIDATES = ["record_id", "id", "sample_id", "crop_id", "image_id"]
GT_CANDIDATES = ["gt", "y_true", "true_class", "label", "ground_truth", "gold"]
TOP1_CANDIDATES = ["dino_top1", "top1_class", "pred_coarse_class", "pred_class", "dino_pred", "predicted_class", "pred", "y_pred"]
TOP2_CANDIDATES = ["dino_top2", "top2_class"]
TOP3_CANDIDATES = ["dino_top3", "top3_class"]
TOP1_SCORE_CANDIDATES = ["dino_top1_score", "top1_score", "confidence", "top1_confidence"]
TOP2_SCORE_CANDIDATES = ["dino_top2_score", "top2_score", "top2_confidence"]
TOP3_SCORE_CANDIDATES = ["dino_top3_score", "top3_score", "top3_confidence"]
MARGIN_CANDIDATES = ["dino_margin", "margin"]


def infer_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        cl = str(col).lower()
        if any(cand.lower() in cl for cand in candidates):
            return col
    return None


def jsonish(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def read_labels(paths: list[str]) -> pd.DataFrame:
    rows = []
    for path_s in paths:
        path = Path(path_s)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                row = {field: jsonish(rec.get(field, "")) for field in LABEL_FIELDS}
                row["label_jsonl_path"] = str(path)
                row["label_jsonl_line"] = line_no
                rows.append(row)
    if not rows:
        raise ValueError("No label rows read from labels-jsonl inputs.")
    labels = pd.DataFrame(rows)
    labels["record_id"] = labels["record_id"].astype(str)
    return labels


def add_topk_from_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    return out


def as_str(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def as_float(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def read_predictions(path_s: str) -> pd.DataFrame:
    raw = add_topk_from_probability_columns(pd.read_csv(path_s))
    cols = {
        "record_id": infer_col(raw, ID_CANDIDATES),
        "top1_class": infer_col(raw, TOP1_CANDIDATES),
        "top2_class": infer_col(raw, TOP2_CANDIDATES),
        "top3_class": infer_col(raw, TOP3_CANDIDATES),
        "top1_score": infer_col(raw, TOP1_SCORE_CANDIDATES),
        "top2_score": infer_col(raw, TOP2_SCORE_CANDIDATES),
        "top3_score": infer_col(raw, TOP3_SCORE_CANDIDATES),
        "margin": infer_col(raw, MARGIN_CANDIDATES),
    }
    if cols["record_id"] is None or cols["top1_class"] is None:
        raise ValueError(f"Could not infer record_id/top1 columns from {list(raw.columns)}")

    pred = pd.DataFrame(
        {
            "record_id": as_str(raw, cols["record_id"]),
            "top1_class": as_str(raw, cols["top1_class"]),
            "top2_class": as_str(raw, cols["top2_class"]),
            "top3_class": as_str(raw, cols["top3_class"]),
            "top1_score": as_float(raw, cols["top1_score"]),
            "top2_score": as_float(raw, cols["top2_score"]),
            "top3_score": as_float(raw, cols["top3_score"]),
            "margin": as_float(raw, cols["margin"]),
        }
    )
    if pred["margin"].isna().all() and not pred["top1_score"].isna().all() and not pred["top2_score"].isna().all():
        pred["margin"] = pred["top1_score"] - pred["top2_score"]
    pred["prediction_csv_path"] = str(path_s)
    return pred


def value_counts_df(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if col not in df:
        return pd.DataFrame(columns=["section", "value", "count"])
    out = df[col].fillna("").astype(str).value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    out.insert(0, "section", prefix)
    return out


def write_readme(out: Path, args, summary: dict):
    lines = [
        "# Stage 10 Full-dataset Table",
        "",
        "This directory joins `vlm_labels_v1` crop-level labels with DINOv2 top-k predictions.",
        "It is a diagnostic base for later VLM checker/reranker experiments, not a VLM inference result.",
        "",
        "Methodology note: frozen VLM configurations should be selected on the development split and reported on the test split.",
        "For historical file names, `val`/`val_v2` usually denotes the current test split.",
        "",
        "## Inputs",
        "",
        *[f"- labels-jsonl: `{p}`" for p in args.labels_jsonl],
        f"- predictions-csv: `{args.predictions_csv}`",
        "",
        "## Join Summary",
        "",
        *[f"- {k}: {v}" for k, v in summary.items()],
        "",
        "## Outputs",
        "",
        "- `stage10_full_dataset_table.csv`",
        "- `stage10_full_dataset_join_summary.csv`",
        "- `stage10_full_dataset_missing_predictions.csv`",
        "- `stage10_full_dataset_missing_labels.csv`",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-jsonl", action="append", required=True)
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.labels_jsonl)
    preds = read_predictions(args.predictions_csv)

    labels_unique = labels.drop_duplicates("record_id", keep="first")
    preds_unique = preds.drop_duplicates("record_id", keep="first")

    joined = labels_unique.merge(preds_unique, on="record_id", how="outer", indicator=True)
    joined["label_matched"] = joined["_merge"].isin(["both", "left_only"])
    joined["dino_prediction_matched"] = joined["_merge"].isin(["both", "right_only"])

    joined["label_coarse_class"] = joined["coarse_class"].fillna("").astype(str)
    joined["label_visual_evidence_tags"] = joined["visual_evidence_tags"].fillna("").astype(str)
    joined["label_visibility"] = joined["visibility"].fillna("").astype(str)
    joined["label_needs_review"] = joined["needs_review"]
    joined["label_short_canonical_description"] = joined["short_canonical_description"].fillna("").astype(str)
    joined["label_report_snippet"] = joined["report_snippet"].fillna("").astype(str)

    joined["dino_top1"] = joined["top1_class"].fillna("").astype(str)
    joined["dino_top2"] = joined["top2_class"].fillna("").astype(str)
    joined["dino_top3"] = joined["top3_class"].fillna("").astype(str)
    joined["dino_top1_score"] = joined["top1_score"]
    joined["dino_top2_score"] = joined["top2_score"]
    joined["dino_top3_score"] = joined["top3_score"]
    joined["dino_margin"] = joined["margin"]

    has_label = joined["label_coarse_class"] != ""
    joined["dino_top1_correct"] = has_label & (joined["label_coarse_class"] == joined["dino_top1"])
    joined["gt_in_top2"] = has_label & (
        (joined["label_coarse_class"] == joined["dino_top1"]) | (joined["label_coarse_class"] == joined["dino_top2"])
    )
    joined["gt_in_top3"] = joined["gt_in_top2"] | (has_label & (joined["label_coarse_class"] == joined["dino_top3"]))

    output_cols = [
        "record_id",
        "split",
        "source",
        "image_id",
        "box_id",
        "crop_path",
        "image_path",
        "bbox_xywh",
        "label_coarse_class",
        "label_visual_evidence_tags",
        "label_visibility",
        "label_needs_review",
        "label_short_canonical_description",
        "label_report_snippet",
        "category_name",
        "annotator_notes",
        "label_version",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
        "dino_top1_correct",
        "gt_in_top2",
        "gt_in_top3",
        "label_matched",
        "dino_prediction_matched",
        "label_jsonl_path",
        "prediction_csv_path",
    ]
    for col in output_cols:
        if col not in joined.columns:
            joined[col] = ""
    table = joined[output_cols].sort_values(["split", "record_id"], na_position="last")
    table.to_csv(out / "stage10_full_dataset_table.csv", index=False)

    missing_predictions = table[table["label_matched"] & ~table["dino_prediction_matched"]]
    missing_labels = table[~table["label_matched"] & table["dino_prediction_matched"]]
    missing_predictions.to_csv(out / "stage10_full_dataset_missing_predictions.csv", index=False)
    missing_labels.to_csv(out / "stage10_full_dataset_missing_labels.csv", index=False)

    summary = {
        "n_label_rows_raw": len(labels),
        "n_prediction_rows_raw": len(preds),
        "n_unique_label_record_id": labels_unique["record_id"].nunique(),
        "n_unique_prediction_record_id": preds_unique["record_id"].nunique(),
        "n_joined_rows": len(table),
        "n_matched_labels_and_predictions": int((table["label_matched"] & table["dino_prediction_matched"]).sum()),
        "n_missing_predictions": int(len(missing_predictions)),
        "n_missing_labels": int(len(missing_labels)),
    }
    summary_rows = [{"metric": k, "value": v} for k, v in summary.items()]
    summary_df = pd.DataFrame(summary_rows)
    distributions = pd.concat(
        [
            value_counts_df(table, "label_coarse_class", "label_coarse_class"),
            value_counts_df(table, "label_visibility", "label_visibility"),
            value_counts_df(table, "label_needs_review", "label_needs_review"),
            value_counts_df(table, "split", "split"),
            value_counts_df(table, "source", "source"),
        ],
        ignore_index=True,
    )
    summary_df.to_csv(out / "stage10_full_dataset_join_summary.csv", index=False)
    distributions.to_csv(out / "stage10_full_dataset_distributions.csv", index=False)
    write_readme(out, args, summary)
    print(summary)
    print("[saved]", out)


if __name__ == "__main__":
    main()
