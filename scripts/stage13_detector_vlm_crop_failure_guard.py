#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="outputs/stage10/full_dataset_all_splits_dinov2_oof_plus_test/stage10_full_dataset_table.csv")
    ap.add_argument("--stage12-dir", default="outputs/stage12")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E04_detector_vlm_crop_failure_guard")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table)
    probe = pd.read_json(Path(args.stage12_dir) / "vlm_evidence_probe" / "dev_train_182_candidate_support_v1.jsonl", lines=True)
    m = df.merge(probe[["record_id", "visibility", "needs_review"]], on="record_id", how="left")

    # proxy target for crop failure
    m["target_crop_failure"] = (
        m["label_needs_review"].astype(bool)
        | m["label_visibility"].isin(["poor", "ambiguous"])
        | (~m["dino_top1_correct"].astype(bool))
    ).astype(int)
    m["dino_margin"] = pd.to_numeric(m["dino_margin"], errors="coerce").fillna(0.0)
    m["bbox_area"] = m["bbox_xywh"].astype(str).str.extract(r"\[(.*)\]").fillna("")
    m["src_is_pred"] = (m["source"] == "pred").astype(int)

    # Feature sets
    fe = m[["record_id", "split", "target_crop_failure", "dino_margin", "src_is_pred", "dino_top1", "visibility", "needs_review"]].copy()
    fe["visibility"] = fe["visibility"].fillna("unknown")
    fe["needs_review"] = fe["needs_review"].fillna(True).astype(int)
    fe["dino_top1"] = fe["dino_top1"].fillna("unknown")
    fe.to_csv(out / "crop_failure_features.csv", index=False)

    dev = fe[fe["split"] == "train"].copy()
    test = fe[fe["split"] != "train"].copy()

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Xdev_cat = ohe.fit_transform(dev[["dino_top1", "visibility"]])
    Xtest_cat = ohe.transform(test[["dino_top1", "visibility"]])
    Xdev_num = dev[["dino_margin", "src_is_pred", "needs_review"]].to_numpy()
    Xtest_num = test[["dino_margin", "src_is_pred", "needs_review"]].to_numpy()

    Xdev = pd.DataFrame(
        data=list(map(list, Xdev_num)),
    )
    Xtest = pd.DataFrame(
        data=list(map(list, Xtest_num)),
    )
    # concat ndarray quickly
    import numpy as np

    Xdev_all = np.hstack([Xdev_num, Xdev_cat])
    Xtest_all = np.hstack([Xtest_num, Xtest_cat])
    ydev = dev["target_crop_failure"].to_numpy()
    ytest = test["target_crop_failure"].to_numpy()

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(Xdev_all, ydev)
    ptest = model.predict_proba(Xtest_all)[:, 1]
    pdev = model.predict_proba(Xdev_all)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "split": "dev",
                "auprc": float(average_precision_score(ydev, pdev)),
                "auroc": float(roc_auc_score(ydev, pdev)),
            },
            {
                "split": "test",
                "auprc": float(average_precision_score(ytest, ptest)),
                "auroc": float(roc_auc_score(ytest, ptest)),
            },
        ]
    )
    metrics.to_csv(out / "crop_failure_model_metrics.csv", index=False)

    sw = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        review = ptest >= thr
        sw.append(
            {
                "threshold": thr,
                "review_rate": float(review.mean()),
                "failure_capture_rate": float(((review == 1) & (ytest == 1)).sum() / max(1, (ytest == 1).sum())),
                "clean_false_review_rate": float(((review == 1) & (ytest == 0)).sum() / max(1, (ytest == 0).sum())),
            }
        )
    pd.DataFrame(sw).to_csv(out / "crop_failure_policy_sweep.csv", index=False)

    cases = test[["record_id", "split", "target_crop_failure"]].copy()
    cases["risk_score"] = ptest
    cases["review_at_03"] = cases["risk_score"] >= 0.3
    cases.to_csv(out / "crop_failure_cases.csv", index=False)

    (out / "E04_report.md").write_text(
        "# E04 Detector+VLM Crop Failure Guard\n\n"
        "Implemented as interpretable logistic model on development split and evaluated on test split.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

