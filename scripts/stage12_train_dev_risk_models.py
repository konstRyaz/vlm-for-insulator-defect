#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DEFECT = {"defect_flashover", "defect_broken"}
TAG_FAMILY = {
    "broken_structure": {"missing_fragment", "edge_discontinuity"},
    "flashover_surface": {"burn_like_mark", "dark_surface_trace", "surface_damage_mark", "surface_stain"},
    "quality_or_confounder": {"blurred_region", "partial_view", "occluded_region", "low_contrast", "ambiguous_evidence"},
}


def parse_tags(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip().strip('"') for x in v if str(x).strip()]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip().strip('"').strip("'").strip("[]") for x in obj if str(x).strip()]
    except Exception:
        pass
    s = s.replace("[", " ").replace("]", " ").replace('"', " ").replace("'", " ")
    return [x.strip() for x in s.split(",") if x.strip()]


def entropy3(a: float, b: float, c: float) -> float:
    p = np.clip(np.array([a, b, c], dtype=float), 1e-9, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dino_pair"] = out["dino_top1"].astype(str) + "->" + out["dino_top2"].astype(str)
    out["entropy"] = [entropy3(a, b, c) for a, b, c in zip(out["dino_top1_score"], out["dino_top2_score"], out["dino_top3_score"])]
    out["vlm_tag_count"] = out["vlm_tags"].apply(len)
    for fam, members in TAG_FAMILY.items():
        out[f"vlm_has_{fam}"] = out["vlm_tags"].apply(lambda x: int(any(t in members for t in x)))
    out["vlm_visibility"] = out["vlm_visibility"].fillna("ambiguous")
    out["vlm_visible_insulator"] = out["vlm_visible_insulator"].fillna("uncertain")
    out["vlm_needs_review"] = out["vlm_needs_review"].fillna(True).astype(int)
    out["class_evidence_consistency"] = (
        ((out["label_coarse_class"] == "insulator_ok") & (out["vlm_has_broken_structure"] == 0) & (out["vlm_has_flashover_surface"] == 0))
        | ((out["label_coarse_class"] == "defect_broken") & (out["vlm_has_broken_structure"] == 1))
        | ((out["label_coarse_class"] == "defect_flashover") & (out["vlm_has_flashover_surface"] == 1))
    ).astype(int)
    return out


def model_oof_probs(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_set: str,
    seed: int,
    model_kind: str = "logreg",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if feature_set == "dino":
        cat = ["dino_top1", "dino_top2", "dino_pair"]
        num = ["dino_top1_score", "dino_top2_score", "dino_top3_score", "dino_margin", "entropy"]
    elif feature_set == "vlm":
        cat = ["vlm_visibility", "vlm_visible_insulator"]
        num = ["vlm_needs_review", "vlm_tag_count", "vlm_has_broken_structure", "vlm_has_flashover_surface", "vlm_has_quality_or_confounder", "class_evidence_consistency"]
    else:
        cat = ["dino_top1", "dino_top2", "dino_pair", "vlm_visibility", "vlm_visible_insulator"]
        num = [
            "dino_top1_score",
            "dino_top2_score",
            "dino_top3_score",
            "dino_margin",
            "entropy",
            "vlm_needs_review",
            "vlm_tag_count",
            "vlm_has_broken_structure",
            "vlm_has_flashover_surface",
            "vlm_has_quality_or_confounder",
            "class_evidence_consistency",
        ]
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ]
    )
    if model_kind == "tree":
        clf = DecisionTreeClassifier(max_depth=3, random_state=seed, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    n_pos = int(y.sum())
    n_splits = max(2, min(5, n_pos, len(y) - n_pos))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold, (tr, va) in enumerate(skf.split(df, y), 1):
        pipe.fit(df.iloc[tr], y[tr])
        p = pipe.predict_proba(df.iloc[va])[:, 1]
        oof[va] = p
        fold_rows.append({"feature_set": feature_set, "model": model_kind, "fold": fold, "n_val": len(va), "pos_rate_val": float(y[va].mean())})
    return oof, fold_rows


def metrics_from_probs(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, p))
    except Exception:
        out["auprc"] = float("nan")
    out["brier"] = float(brier_score_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    return out


def review_budget_metrics(df: pd.DataFrame, score_col: str, y_col: str, budgets=(0.05, 0.1, 0.15, 0.2, 0.3)) -> pd.DataFrame:
    rows = []
    n = len(df)
    base_acc = (df["dino_top1"] == df["label_coarse_class"]).mean()
    for b in budgets:
        k = max(1, int(round(n * b)))
        top = df.nlargest(k, score_col).copy()
        keep = df.drop(top.index)
        accepted_acc = float((keep["dino_top1"] == keep["label_coarse_class"]).mean()) if len(keep) else 1.0
        err_capture = float(top[y_col].mean()) if len(top) else 0.0
        dangerous_capture = float(top["dangerous_miss"].mean()) if len(top) else 0.0
        rows.append(
            {
                "score": score_col,
                "target": y_col,
                "review_rate": b,
                "base_accuracy": float(base_acc),
                "accepted_accuracy": accepted_acc,
                "accuracy_gain_vs_base": accepted_acc - base_acc,
                "error_capture_rate": err_capture,
                "dangerous_miss_capture_rate": dangerous_capture,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage10-table", required=True)
    ap.add_argument("--vlm-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(args.stage10_table)
    t = t[t["split"].astype(str) == "train"].copy()
    v = pd.DataFrame([json.loads(x) for x in Path(args.vlm_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()])
    v = v.rename(columns={"visibility": "vlm_visibility", "visible_insulator": "vlm_visible_insulator", "needs_review": "vlm_needs_review", "evidence_tags": "vlm_tags"})
    v["vlm_tags"] = v["vlm_tags"].apply(parse_tags)
    df = t.merge(v[["record_id", "vlm_visibility", "vlm_visible_insulator", "vlm_needs_review", "vlm_tags"]], on="record_id", how="left")
    df = build_features(df)

    df["general_error"] = (df["dino_top1"] != df["label_coarse_class"]).astype(int)
    df["dangerous_miss"] = ((df["label_coarse_class"].isin(list(DEFECT))) & (df["dino_top1"] == "insulator_ok")).astype(int)
    df["false_alarm"] = ((df["label_coarse_class"] == "insulator_ok") & (df["dino_top1"].isin(list(DEFECT)))).astype(int)
    df["non_clear_or_review"] = ((df["label_visibility"] != "clear") | (df["label_needs_review"].astype(bool))).astype(int)

    df.to_csv(out / "dev_risk_features.csv", index=False)

    all_fold_rows = []
    pred_rows = pd.DataFrame({"record_id": df["record_id"]})
    metrics_rows = []
    for target in ["general_error", "dangerous_miss"]:
        y = df[target].to_numpy()
        for feat in ["dino", "vlm", "dino_vlm"]:
            for model_kind in ["logreg", "tree"]:
                p, fold_rows = model_oof_probs(df, y, feat, args.seed, model_kind=model_kind)
                col = f"risk_{target}_{feat}_{model_kind}"
                pred_rows[col] = p
                all_fold_rows.extend(fold_rows)
                m = {"target": target, "feature_set": feat, "model": model_kind}
                m.update(metrics_from_probs(y, p))
                metrics_rows.append(m)

    pred_rows.to_csv(out / "dev_oof_risk_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(out / "dev_risk_metrics.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(out / "dev_oof_fold_meta.csv", index=False)

    # Coverage/policy outputs using best model per target by AUPRC.
    mdf = pd.DataFrame(metrics_rows)
    policy_rows = []
    acc_rows = []
    for target in ["general_error", "dangerous_miss"]:
        sub = mdf[mdf["target"] == target].sort_values(["auprc", "auroc"], ascending=False)
        best = sub.iloc[0]
        score_col = f"risk_{target}_{best['feature_set']}_{best['model']}"
        work = df.join(pred_rows.set_index("record_id"), on="record_id")
        rb = review_budget_metrics(work, score_col, target)
        rb["best_feature_set"] = best["feature_set"]
        rb["best_model"] = best["model"]
        policy_rows.append(rb)
        acc_rows.append(rb[["target", "review_rate", "accepted_accuracy", "accuracy_gain_vs_base", "dangerous_miss_capture_rate"]])
    pol = pd.concat(policy_rows, ignore_index=True)
    pol.to_csv(out / "dev_policy_sweep.csv", index=False)
    pd.concat(acc_rows, ignore_index=True).to_csv(out / "dev_accepted_accuracy_at_review_rates.csv", index=False)
    pol.to_csv(out / "dev_risk_coverage_curve.csv", index=False)

    # PR/ROC curve points for best general_error model.
    from sklearn.metrics import precision_recall_curve, roc_curve

    best_ge = mdf[mdf["target"] == "general_error"].sort_values(["auprc", "auroc"], ascending=False).iloc[0]
    ge_col = f"risk_general_error_{best_ge['feature_set']}_{best_ge['model']}"
    ge_pred = pred_rows[ge_col].to_numpy()
    y_ge = df["general_error"].to_numpy()
    p, r, thr = precision_recall_curve(y_ge, ge_pred)
    pd.DataFrame({"precision": p, "recall": r, "threshold": np.append(thr, np.nan)}).to_csv(out / "dev_error_pr_curve.csv", index=False)
    fpr, tpr, thr2 = roc_curve(y_ge, ge_pred)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr2}).to_csv(out / "dev_error_roc_curve.csv", index=False)

    report = out / "dev_policy_sweep.md"
    lines = ["# Stage12 EXP-D Risk/Review (Dev)", "", "## Best models by AUPRC", ""]
    lines.append(mdf.sort_values(["target", "auprc"], ascending=[True, False]).groupby("target").head(1).to_string(index=False))
    lines.append("")
    lines.append("## Review-budget summary (best models)")
    lines.append(pol.to_string(index=False))
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
