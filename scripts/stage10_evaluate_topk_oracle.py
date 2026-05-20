import argparse
from pathlib import Path

import pandas as pd


ID_CANDIDATES = ["id", "sample_id", "record_id", "image_id", "crop_id"]
TRUE_CANDIDATES = ["gt", "y_true", "true_class", "label", "ground_truth"]
TOP1_CANDIDATES = ["dino_top1", "pred", "y_pred", "pred_class", "dino_pred", "predicted_class", "pred_coarse_class", "top1_class"]
TOP2_CANDIDATES = ["dino_top2", "top2_class"]
TOP3_CANDIDATES = ["dino_top3", "top3_class"]


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


def as_str_series(df, col):
    if col is None:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def add_topk_from_probability_columns(df):
    prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
    if not prob_cols:
        return df
    out = df.copy()
    for i, row in out[prob_cols].apply(pd.to_numeric, errors="coerce").iterrows():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = add_topk_from_probability_columns(pd.read_csv(args.predictions_csv))
    gt_col = infer_col(df, TRUE_CANDIDATES)
    top1_col = infer_col(df, TOP1_CANDIDATES)
    top2_col = infer_col(df, TOP2_CANDIDATES)
    top3_col = infer_col(df, TOP3_CANDIDATES)

    if gt_col is None or top1_col is None:
        raise ValueError(
            "Could not infer required ground-truth and top-1 prediction columns. "
            f"Columns: {list(df.columns)}"
        )

    gt = as_str_series(df, gt_col)
    top1 = as_str_series(df, top1_col)
    top2 = as_str_series(df, top2_col)
    top3 = as_str_series(df, top3_col)

    top1_correct = gt == top1
    gt_in_top2 = (gt == top1) | (gt == top2)
    gt_in_top3 = gt_in_top2 | (gt == top3)
    top1_wrong = ~top1_correct

    n = int(len(df))
    n_wrong = int(top1_wrong.sum())
    n_recoverable_top2 = int((top1_wrong & (gt == top2)).sum()) if top2_col else 0
    n_recoverable_top3 = int((top1_wrong & gt_in_top3).sum()) if top3_col else 0

    summary = {
        "n": n,
        "top1_accuracy": float(top1_correct.mean()) if n else 0.0,
        "top2_oracle_accuracy": float(gt_in_top2.mean()) if n and top2_col else None,
        "top3_oracle_accuracy": float(gt_in_top3.mean()) if n and top3_col else None,
        "n_top1_wrong": n_wrong,
        "n_recoverable_top2": n_recoverable_top2,
        "n_recoverable_top3": n_recoverable_top3,
        "recoverable_top2_rate_among_errors": float(n_recoverable_top2 / n_wrong) if n_wrong else 0.0,
        "recoverable_top3_rate_among_errors": float(n_recoverable_top3 / n_wrong) if n_wrong else 0.0,
    }

    pd.DataFrame([summary]).to_csv(out / "stage10_topk_oracle_summary.csv", index=False)

    lines = [
        "# Stage 10 Top-k Oracle Summary",
        "",
        f"- input: `{Path(args.predictions_csv)}`",
        f"- n: {summary['n']}",
        f"- top1_accuracy: {summary['top1_accuracy']:.4f}",
        f"- top2_oracle_accuracy: {summary['top2_oracle_accuracy'] if summary['top2_oracle_accuracy'] is not None else 'NA'}",
        f"- top3_oracle_accuracy: {summary['top3_oracle_accuracy'] if summary['top3_oracle_accuracy'] is not None else 'NA'}",
        f"- n_top1_wrong: {summary['n_top1_wrong']}",
        f"- n_recoverable_top2: {summary['n_recoverable_top2']}",
        f"- n_recoverable_top3: {summary['n_recoverable_top3']}",
        f"- recoverable_top2_rate_among_errors: {summary['recoverable_top2_rate_among_errors']:.4f}",
        f"- recoverable_top3_rate_among_errors: {summary['recoverable_top3_rate_among_errors']:.4f}",
        "",
        "Top-k oracle accuracy is an upper bound for a constrained checker/reranker.",
    ]
    (out / "stage10_topk_oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)
    print("[saved]", out)


if __name__ == "__main__":
    main()
