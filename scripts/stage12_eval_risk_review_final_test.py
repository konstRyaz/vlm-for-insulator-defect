#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def load_vlm_features(jsonl_path: Path) -> pd.DataFrame:
    df = pd.read_json(jsonl_path, lines=True)
    for c in ["needs_review", "evidence_strength", "crop_quality", "selected_rank"]:
        if c not in df.columns:
            df[c] = np.nan
    out = pd.DataFrame({"record_id": df["record_id"].astype(str)})
    out["vlm_needs_review"] = df["needs_review"].astype(str).str.lower().isin(["true", "1", "yes"])
    out["vlm_selected_not_top1"] = df["selected_rank"].astype(str).isin(["top2", "top3"])
    out["vlm_uncertain"] = df["selected_rank"].astype(str).eq("uncertain")
    out["vlm_evidence_strong"] = df["evidence_strength"].astype(str).eq("strong")
    out["vlm_quality_bad"] = df["crop_quality"].astype(str).isin(["bad", "ambiguous", "partial"])
    return out


def eval_at_rate(scores: np.ndarray, y: np.ndarray, dangerous: np.ndarray, review_rate: float):
    n = len(scores)
    k = int(round(review_rate * n))
    order = np.argsort(-scores)
    reviewed = np.zeros(n, dtype=bool)
    reviewed[order[:k]] = True
    accepted = ~reviewed
    accepted_acc = float((1 - y[accepted]).mean()) if accepted.any() else 0.0
    base_acc = float((1 - y).mean())
    err_cap = float(y[reviewed].sum() / y.sum()) if y.sum() else 0.0
    dm_cap = float(dangerous[reviewed].sum() / dangerous.sum()) if dangerous.sum() else 0.0
    corr_rej = float((y[reviewed] == 1).mean()) if reviewed.any() else 0.0
    return {
        "review_rate": review_rate,
        "accepted_accuracy": accepted_acc,
        "base_accuracy": base_acc,
        "accuracy_gain_vs_base": accepted_acc - base_acc,
        "error_capture_rate": err_cap,
        "dangerous_miss_capture_rate": dm_cap,
        "correct_rejection_rate": corr_rej,
        "coverage": float(accepted.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", default="outputs/stage10/full_dataset_all_splits_dinov2_oof_plus_test/stage10_full_dataset_table.csv")
    ap.add_argument("--dev-vlm-jsonl", default="outputs/_kaggle_runs/stage10_vlm_topk_dev_evidence_t4/stage10_vlm_topk_outputs/train_dev_182_evidence_compare.jsonl")
    ap.add_argument("--test-vlm-jsonl", default="outputs/_kaggle_runs/stage10_vlm_topk_test_evidence_t4/stage10_vlm_topk_outputs/test_val_58_evidence_compare.jsonl")
    ap.add_argument("--out-dir", default="outputs/stage12/risk_review_final_test")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.table_csv)
    table["general_error"] = (~table["dino_top1_correct"].astype(bool)).astype(int)
    table["dangerous_miss"] = (
        table["label_coarse_class"].astype(str).str.startswith("defect_")
        & (table["dino_top1"].astype(str) == "insulator_ok")
    ).astype(int)
    table["false_alarm"] = (
        (table["label_coarse_class"].astype(str) == "insulator_ok")
        & table["dino_top1"].astype(str).str.startswith("defect_")
    ).astype(int)

    dev_vlm = load_vlm_features(Path(args.dev_vlm_jsonl))
    test_vlm = load_vlm_features(Path(args.test_vlm_jsonl))
    vf = pd.concat([dev_vlm, test_vlm], ignore_index=True).drop_duplicates("record_id", keep="first")
    table = table.merge(vf, on="record_id", how="left")
    for c in ["vlm_needs_review", "vlm_selected_not_top1", "vlm_uncertain", "vlm_evidence_strong", "vlm_quality_bad"]:
        table[c] = table[c].fillna(False).astype(int)

    table["dino_top1_score"] = table["dino_top1_score"].fillna(0.0)
    table["dino_top2_score"] = table["dino_top2_score"].fillna(0.0)
    table["dino_margin"] = table["dino_margin"].fillna(0.0)
    table["pair_is_ok_defect"] = (
        ((table["dino_top1"] == "insulator_ok") & table["dino_top2"].astype(str).str.startswith("defect_"))
        | ((table["dino_top2"] == "insulator_ok") & table["dino_top1"].astype(str).str.startswith("defect_"))
    ).astype(int)

    dev = table[table["split"] == "train"].copy()
    test = table[table["split"] == "val"].copy()

    feats_dino = ["dino_top1_score", "dino_top2_score", "dino_margin", "pair_is_ok_defect"]
    feats_vlm = ["vlm_needs_review", "vlm_selected_not_top1", "vlm_uncertain", "vlm_evidence_strong", "vlm_quality_bad"]
    feats_dv = feats_dino + feats_vlm

    y_dev = dev["general_error"].astype(int).to_numpy()
    y_test = test["general_error"].astype(int).to_numpy()
    dm_dev = dev["dangerous_miss"].astype(int).to_numpy()
    dm_test = test["dangerous_miss"].astype(int).to_numpy()

    models = {
        "R0_dino_only": (feats_dino, LogisticRegression(max_iter=2000, random_state=42)),
        "R1_vlm_only": (feats_vlm, LogisticRegression(max_iter=2000, random_state=42)),
        "R2_dino_vlm": (feats_dv, LogisticRegression(max_iter=2000, random_state=42)),
    }

    all_metrics = []
    pred_dev = pd.DataFrame({"record_id": dev["record_id"]})
    pred_test = pd.DataFrame({"record_id": test["record_id"]})
    thresholds = []
    acc_rows_dev = []
    acc_rows_test = []
    curve_dev = []
    curve_test = []

    for name, (cols, model) in models.items():
        model.fit(dev[cols], y_dev)
        s_dev = model.predict_proba(dev[cols])[:, 1]
        s_test = model.predict_proba(test[cols])[:, 1]
        pred_dev[name] = s_dev
        pred_test[name] = s_test
        for split_name, y, s, dm in [("dev", y_dev, s_dev, dm_dev), ("test", y_test, s_test, dm_test)]:
            auroc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
            auprc = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan
            all_metrics.append(
                {"split": split_name, "model": name, "target": "general_error", "auroc": auroc, "auprc": auprc}
            )
            for rr in [0.05, 0.10, 0.15, 0.20, 0.30]:
                row = eval_at_rate(s, y, dm, rr)
                row.update({"split": split_name, "model": name})
                if split_name == "dev":
                    acc_rows_dev.append(row)
                    curve_dev.append(row)
                else:
                    acc_rows_test.append(row)
                    curve_test.append(row)

        # thresholds selected on dev
        for rr in [0.05, 0.10, 0.15, 0.20, 0.30]:
            thr = float(np.quantile(s_dev, 1 - rr))
            thresholds.append({"model": name, "review_rate": rr, "threshold_selected_on_dev": thr})

    pd.DataFrame(all_metrics).to_csv(out / "risk_review_final_metrics_dev_test.csv", index=False)
    pd.DataFrame([x for x in all_metrics if x["split"] == "dev"]).to_csv(out / "risk_review_final_metrics_dev.csv", index=False)
    pd.DataFrame([x for x in all_metrics if x["split"] == "test"]).to_csv(out / "risk_review_final_metrics_test.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "risk_review_thresholds_selected_on_dev.csv", index=False)
    pred_dev.to_csv(out / "risk_review_predictions_dev.csv", index=False)
    pred_test.to_csv(out / "risk_review_predictions_test.csv", index=False)
    pd.DataFrame(acc_rows_dev).to_csv(out / "accepted_accuracy_at_review_rates_dev.csv", index=False)
    pd.DataFrame(acc_rows_test).to_csv(out / "accepted_accuracy_at_review_rates_test.csv", index=False)
    pd.DataFrame(curve_dev).to_csv(out / "risk_coverage_curve_dev.csv", index=False)
    pd.DataFrame(curve_test).to_csv(out / "risk_coverage_curve_test.csv", index=False)

    # case tables from best model R2_dino_vlm at 10%
    s_test = pred_test["R2_dino_vlm"].to_numpy()
    n = len(s_test)
    k = int(round(0.10 * n))
    ord_idx = np.argsort(-s_test)
    reviewed = np.zeros(n, dtype=bool)
    reviewed[ord_idx[:k]] = True
    test_case = test.reset_index(drop=True).copy()
    test_case["reviewed_10"] = reviewed
    test_case["dino_error"] = (test_case["dino_top1"] != test_case["label_coarse_class"]).astype(int)
    test_case[(test_case["dino_error"] == 1) & (test_case["reviewed_10"])].to_csv(
        out / "dino_only_missed_vlm_caught_test.csv", index=False
    )
    test_case[(test_case["dino_error"] == 0) & (test_case["reviewed_10"])].to_csv(out / "vlm_hurt_cases_test.csv", index=False)
    test_case[test_case["dangerous_miss"] == 1].to_csv(out / "dangerous_miss_cases_test.csv", index=False)

    report = [
        "# Risk/Review Final Test",
        "",
        "Thresholds selected on development split, then applied frozen on test split.",
        "See CSV metrics for model-level AUROC/AUPRC and coverage-accuracy trade-offs.",
    ]
    (out / "risk_review_final_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
