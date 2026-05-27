import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASSES = ["defect_broken", "defect_flashover", "insulator_ok"]


def boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["label_coarse_class"].fillna("") != "") & (df["dino_top1"].fillna("") != "")].copy()
    sub["label_coarse_class"] = sub["label_coarse_class"].astype(str)
    for col in ["dino_top1", "dino_top2", "dino_top3"]:
        sub[col] = sub[col].fillna("").astype(str)
    sub["dino_margin"] = pd.to_numeric(sub["dino_margin"], errors="coerce")
    sub["baseline_correct"] = sub["label_coarse_class"] == sub["dino_top1"]
    return sub


def metric_row(df: pd.DataFrame, policy_name: str, policy_family: str, split_name: str, final_pred: pd.Series, reviewed: pd.Series, params: dict):
    n = len(df)
    reviewed = reviewed.fillna(False).astype(bool)
    accepted = ~reviewed
    final_pred = final_pred.fillna("").astype(str)
    correct_final = accepted & (final_pred == df["label_coarse_class"])
    baseline_correct = df["baseline_correct"].astype(bool)
    baseline_wrong = ~baseline_correct
    switched = accepted & (final_pred != df["dino_top1"])

    helped = baseline_wrong & (reviewed | correct_final)
    hurt = baseline_correct & (reviewed | (accepted & (final_pred != df["label_coarse_class"])))
    captured_error = baseline_wrong & (reviewed | correct_final)
    correct_rejected = baseline_correct & reviewed

    accepted_labels = df.loc[accepted, "label_coarse_class"]
    accepted_preds = final_pred[accepted]
    if accepted.sum():
        accepted_accuracy = float((accepted_preds.to_numpy() == accepted_labels.to_numpy()).mean())
        macro_f1 = float(f1_score(accepted_labels, accepted_preds, labels=CLASSES, average="macro", zero_division=0))
    else:
        accepted_accuracy = 0.0
        macro_f1 = 0.0

    row = {
        "policy_name": policy_name,
        "policy_family": policy_family,
        "split": split_name,
        "n": int(n),
        "accuracy": safe_div(int(correct_final.sum()), n),
        "macro_f1": macro_f1,
        "coverage": safe_div(int(accepted.sum()), n),
        "accepted_accuracy": accepted_accuracy,
        "review_rate": safe_div(int(reviewed.sum()), n),
        "helped": int(helped.sum()),
        "hurt": int(hurt.sum()),
        "net_gain": int(helped.sum() - hurt.sum()),
        "switch_rate": safe_div(int(switched.sum()), n),
        "error_capture_rate": safe_div(int(captured_error.sum()), int(baseline_wrong.sum())),
        "correct_rejection_rate": safe_div(int(correct_rejected.sum()), int(baseline_correct.sum())),
    }
    row.update(params)
    return row


def threshold_grid(margins: pd.Series) -> list[float]:
    finite = pd.to_numeric(margins, errors="coerce").dropna()
    fixed = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]
    if finite.empty:
        return fixed
    qs = finite.quantile([0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90]).tolist()
    vals = sorted({round(float(x), 6) for x in fixed + qs if pd.notna(x)})
    return vals


def apply_policy(df: pd.DataFrame, family: str, threshold: float | None = None, target_class: str | None = None, rng_seed: int = 42):
    pred = df["dino_top1"].copy()
    reviewed = pd.Series(False, index=df.index)
    margin = pd.to_numeric(df["dino_margin"], errors="coerce")
    low_margin = margin <= threshold if threshold is not None else pd.Series(False, index=df.index)
    if target_class:
        low_margin = low_margin & (df["dino_top1"] == target_class)

    if family == "dino_top1":
        return pred, reviewed
    if family == "top2_oracle":
        pred = np.where((df["label_coarse_class"] == df["dino_top2"]) & (df["dino_top2"] != ""), df["dino_top2"], df["dino_top1"])
        return pd.Series(pred, index=df.index), reviewed
    if family == "top3_oracle":
        pred = np.where(
            (df["label_coarse_class"] == df["dino_top2"]) & (df["dino_top2"] != ""),
            df["dino_top2"],
            np.where((df["label_coarse_class"] == df["dino_top3"]) & (df["dino_top3"] != ""), df["dino_top3"], df["dino_top1"]),
        )
        return pd.Series(pred, index=df.index), reviewed
    if family == "margin_review":
        reviewed = low_margin.fillna(False)
        return pred, reviewed
    if family == "margin_switch_top2":
        can_switch = low_margin.fillna(False) & (df["dino_top2"] != "")
        pred = pred.mask(can_switch, df["dino_top2"])
        return pred, reviewed
    if family == "random_review":
        rng = np.random.default_rng(rng_seed)
        rate = float(threshold or 0.0)
        reviewed = pd.Series(rng.random(len(df)) < rate, index=df.index)
        return pred, reviewed
    if family == "random_switch_top2":
        rng = np.random.default_rng(rng_seed)
        rate = float(threshold or 0.0)
        can_switch = (pd.Series(rng.random(len(df)) < rate, index=df.index)) & (df["dino_top2"] != "")
        pred = pred.mask(can_switch, df["dino_top2"])
        return pred, reviewed
    raise ValueError(f"Unknown policy family: {family}")


def evaluate_policies(df: pd.DataFrame, thresholds: list[float], random_rates: list[float]) -> pd.DataFrame:
    rows = []
    splits = [("all", df)]
    if "split" in df.columns:
        splits += [(str(k), v.copy()) for k, v in df.groupby("split", dropna=False)]

    policy_specs = [
        ("dino_top1", "dino_top1", None, None),
        ("top2_oracle", "top2_oracle", None, None),
        ("top3_oracle", "top3_oracle", None, None),
    ]
    for t in thresholds:
        policy_specs.append((f"margin_review_t{t:.3f}", "margin_review", t, None))
        policy_specs.append((f"margin_switch_top2_t{t:.3f}", "margin_switch_top2", t, None))
        for cls in CLASSES:
            policy_specs.append((f"classwise_{cls}_margin_review_t{t:.3f}", "margin_review", t, cls))
            policy_specs.append((f"classwise_{cls}_margin_switch_top2_t{t:.3f}", "margin_switch_top2", t, cls))
    for r in random_rates:
        policy_specs.append((f"random_review_r{r:.3f}", "random_review", r, None))
        policy_specs.append((f"random_switch_top2_r{r:.3f}", "random_switch_top2", r, None))

    for policy_name, family, threshold, target_class in policy_specs:
        for split_name, sub in splits:
            pred, reviewed = apply_policy(sub, family, threshold, target_class)
            rows.append(
                metric_row(
                    sub,
                    policy_name,
                    family,
                    split_name,
                    pred,
                    reviewed,
                    {
                        "threshold": "" if threshold is None else threshold,
                        "target_class": target_class or "",
                    },
                )
            )
    return pd.DataFrame(rows)


def write_readme(out: Path, table_csv: str, best: pd.DataFrame):
    def simple_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "(none)"
        view = df.copy()
        for col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in view.astype(str).values.tolist()]
        return "\n".join([header, sep] + body)

    lines = [
        "# Stage 10 Non-VLM Policy Baselines",
        "",
        "This directory evaluates non-VLM policies on the full Stage 10 table.",
        "Historical `train` is the development split for policy selection; historical `val` is the test split for evaluation.",
        "",
        "No VLM inference is used here. Oracle rows are upper bounds that use labels and must not be treated as deployable policies.",
        "",
        "## Metrics",
        "",
        "- `accuracy`: correct accepted predictions divided by all rows; reviewed rows count as not automatically correct.",
        "- `macro_f1`: macro-F1 on accepted/non-reviewed rows.",
        "- `coverage`: share of rows with an automatic prediction.",
        "- `accepted_accuracy`: accuracy among accepted/non-reviewed rows.",
        "- `error_capture_rate`: share of DINOv2 top1 errors that the policy either reviews or corrects by switching.",
        "- `correct_rejection_rate`: share of DINOv2 top1-correct rows sent to review.",
        "",
        "## Input",
        "",
        f"- table: `{table_csv}`",
        "",
        "## Best Development Policies",
        "",
    ]
    if len(best):
        lines.append(simple_markdown(best.head(10)))
    else:
        lines.append("No development rows were found.")
    lines.append("")
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = usable_rows(pd.read_csv(args.table_csv))
    thresholds = threshold_grid(df["dino_margin"])
    random_rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    results = evaluate_policies(df, thresholds, random_rates)

    results.to_csv(out / "policy_results_all.csv", index=False)
    results[results["split"] != "all"].to_csv(out / "policy_results_by_split.csv", index=False)

    dev = results[results["split"].isin(["train", "development", "dev"])].copy()
    if len(dev):
        dev = dev[~dev["policy_family"].str.contains("oracle", na=False)].copy()
        dev = dev.sort_values(["net_gain", "accuracy", "macro_f1", "review_rate"], ascending=[False, False, False, True])
        ranked = dev.drop_duplicates("policy_name").head(25)[["policy_name"]].copy()
        ranked["dev_rank"] = range(1, len(ranked) + 1)
        best = results[results["policy_name"].isin(ranked["policy_name"])].merge(ranked, on="policy_name", how="left")
        split_order = {"all": 0, "train": 1, "development": 1, "dev": 1, "val": 2, "test": 2}
        best["_split_order"] = best["split"].map(split_order).fillna(9)
        best = best.sort_values(["dev_rank", "_split_order", "split"]).drop(columns=["_split_order"])
    else:
        best = pd.DataFrame()
    best.to_csv(out / "best_dev_policies.csv", index=False)
    write_readme(out, args.table_csv, best)
    print({"rows": len(results), "out_dir": str(out), "best_dev_policies": 0 if best is None else len(best)})


if __name__ == "__main__":
    main()
