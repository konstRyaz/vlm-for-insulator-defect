import argparse
from pathlib import Path

import pandas as pd


def boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def safe_div(a: float, b: float):
    return float(a / b) if b else 0.0


def metrics_for(df: pd.DataFrame, group_name: str = "all", group_value: str = "all") -> dict:
    has_label_and_pred = (df["label_coarse_class"].fillna("") != "") & (df["dino_top1"].fillna("") != "")
    sub = df[has_label_and_pred].copy()
    n = len(sub)
    if n == 0:
        return {
            "group": group_name,
            "value": group_value,
            "n": 0,
            "top1_accuracy": 0.0,
            "top2_oracle_accuracy": 0.0,
            "top3_oracle_accuracy": 0.0,
            "n_top1_wrong": 0,
            "n_recoverable_top2": 0,
            "n_recoverable_top3": 0,
            "recoverable_top2_rate_among_errors": 0.0,
            "recoverable_top3_rate_among_errors": 0.0,
        }
    top1_correct = boolish(sub["dino_top1_correct"])
    gt_in_top2 = boolish(sub["gt_in_top2"])
    gt_in_top3 = boolish(sub["gt_in_top3"])
    top1_wrong = ~top1_correct
    recoverable_top2 = top1_wrong & (sub["label_coarse_class"] == sub["dino_top2"])
    recoverable_top3 = top1_wrong & gt_in_top3
    n_wrong = int(top1_wrong.sum())
    return {
        "group": group_name,
        "value": group_value,
        "n": int(n),
        "top1_accuracy": float(top1_correct.mean()),
        "top2_oracle_accuracy": float(gt_in_top2.mean()),
        "top3_oracle_accuracy": float(gt_in_top3.mean()),
        "n_top1_wrong": n_wrong,
        "n_recoverable_top2": int(recoverable_top2.sum()),
        "n_recoverable_top3": int(recoverable_top3.sum()),
        "recoverable_top2_rate_among_errors": safe_div(int(recoverable_top2.sum()), n_wrong),
        "recoverable_top3_rate_among_errors": safe_div(int(recoverable_top3.sum()), n_wrong),
    }


def grouped_metrics(df: pd.DataFrame, col: str, group_name: str) -> pd.DataFrame:
    rows = []
    for value, sub in df.groupby(col, dropna=False):
        rows.append(metrics_for(sub, group_name, str(value)))
    return pd.DataFrame(rows)


def write_summary_md(out: Path, overall: pd.DataFrame, split_df: pd.DataFrame):
    row = overall.iloc[0].to_dict()
    lines = [
        "# Stage 10 Full-dataset Oracle Analysis",
        "",
        "This report estimates the upper bound for a constrained VLM top-k reranker.",
        "",
        "## Overall",
        "",
        f"- n: {int(row['n'])}",
        f"- top1_accuracy: {row['top1_accuracy']:.4f}",
        f"- top2_oracle_accuracy: {row['top2_oracle_accuracy']:.4f}",
        f"- top3_oracle_accuracy: {row['top3_oracle_accuracy']:.4f}",
        f"- n_top1_wrong: {int(row['n_top1_wrong'])}",
        f"- n_recoverable_top2: {int(row['n_recoverable_top2'])}",
        f"- n_recoverable_top3: {int(row['n_recoverable_top3'])}",
        f"- recoverable_top2_rate_among_errors: {row['recoverable_top2_rate_among_errors']:.4f}",
        f"- recoverable_top3_rate_among_errors: {row['recoverable_top3_rate_among_errors']:.4f}",
        "",
        "## By Split",
        "",
        "| split | n | top1 | top2 oracle | top3 oracle | recoverable top2 errors | recoverable top3 errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in split_df.iterrows():
        lines.append(
            f"| {r['value']} | {int(r['n'])} | {r['top1_accuracy']:.4f} | "
            f"{r['top2_oracle_accuracy']:.4f} | {r['top3_oracle_accuracy']:.4f} | "
            f"{int(r['n_recoverable_top2'])} | {int(r['n_recoverable_top3'])} |"
        )
    lines += [
        "",
        "Interpretation rule: if top-2/top-3 oracle is much higher than top-1, a VLM reranker has potential.",
        "Policy selection should use the development split; final claims should be reported on the test split.",
    ]
    (out / "stage10_full_oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table_csv)

    overall = pd.DataFrame([metrics_for(df)])
    by_split = grouped_metrics(df, "split", "split") if "split" in df.columns else pd.DataFrame()
    by_class = grouped_metrics(df, "label_coarse_class", "label_coarse_class")
    by_visibility = grouped_metrics(df, "label_visibility", "label_visibility")
    by_review = grouped_metrics(df, "label_needs_review", "label_needs_review")
    by_source = grouped_metrics(df, "source", "source") if "source" in df.columns else pd.DataFrame()

    overall.to_csv(out / "stage10_full_oracle_summary.csv", index=False)
    by_split.to_csv(out / "stage10_full_oracle_by_split.csv", index=False)
    by_class.to_csv(out / "stage10_full_oracle_by_class.csv", index=False)
    by_visibility.to_csv(out / "stage10_full_oracle_by_visibility.csv", index=False)
    by_review.to_csv(out / "stage10_full_oracle_by_needs_review.csv", index=False)
    by_source.to_csv(out / "stage10_full_oracle_by_source.csv", index=False)

    matched = df[(df["label_coarse_class"].fillna("") != "") & (df["dino_top1"].fillna("") != "")]
    confusion = pd.crosstab(matched["label_coarse_class"], matched["dino_top1"])
    confusion.to_csv(out / "stage10_full_confusion_top1.csv")

    wrong = matched[matched["label_coarse_class"] != matched["dino_top1"]].copy()
    if len(wrong):
        wrong["recoverable_top2"] = wrong["label_coarse_class"] == wrong["dino_top2"]
        wrong["recoverable_top3"] = boolish(wrong["gt_in_top3"])
        pairs = (
            wrong.groupby(["label_coarse_class", "dino_top1", "dino_top2"], dropna=False)
            .agg(
                n=("record_id", "size"),
                n_recoverable_top2=("recoverable_top2", "sum"),
                n_recoverable_top3=("recoverable_top3", "sum"),
                mean_margin=("dino_margin", "mean"),
            )
            .reset_index()
            .sort_values(["n", "n_recoverable_top2"], ascending=[False, False])
        )
    else:
        pairs = pd.DataFrame(
            columns=["label_coarse_class", "dino_top1", "dino_top2", "n", "n_recoverable_top2", "n_recoverable_top3", "mean_margin"]
        )
    pairs.to_csv(out / "stage10_full_recoverable_pairs.csv", index=False)

    if "label_visual_evidence_tags" in matched.columns:
        tag_rows = []
        for _, row in matched.iterrows():
            tags = str(row.get("label_visual_evidence_tags", "") or "")
            for tag in tags.replace("[", "").replace("]", "").replace('"', "").replace("'", "").split(","):
                tag = tag.strip()
                if tag:
                    tag_rows.append({"tag": tag, "top1_correct": bool(row["label_coarse_class"] == row["dino_top1"])})
        tag_df = pd.DataFrame(tag_rows)
        if len(tag_df):
            tag_df.groupby("tag").agg(n=("top1_correct", "size"), top1_accuracy=("top1_correct", "mean")).reset_index().sort_values(
                "n", ascending=False
            ).to_csv(out / "stage10_full_oracle_by_evidence_tag.csv", index=False)

    write_summary_md(out, overall, by_split)
    print(overall.to_string(index=False))
    print("[saved]", out)


if __name__ == "__main__":
    main()
