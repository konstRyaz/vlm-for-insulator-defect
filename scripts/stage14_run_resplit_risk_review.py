#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


TARGETS = [
    "general_error",
    "false_alarm",
    "ok_to_flashover_false_alarm",
    "dangerous_miss",
    "defect_vs_ok",
    "defect_type_confusion",
]
REVIEW_BUDGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def _prevalence(y: np.ndarray) -> float:
    return float(y.mean()) if len(y) else float("nan")


def _threshold_for_budget(scores: np.ndarray, budget: float) -> float:
    # review if score >= threshold
    if len(scores) == 0:
        return float("inf")
    q = max(0.0, min(1.0, 1.0 - budget))
    return float(np.quantile(scores, q))


def _build_method_scores(dev: pd.DataFrame, test: pd.DataFrame, target: str, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Shared columns
    dino_feats = [
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
        "entropy",
    ]
    vlm_feats = [
        "vlm_needs_review",
        "vlm_has_broken_structure",
        "vlm_has_flashover_surface",
        "vlm_has_quality_or_confounder",
        "class_evidence_consistency",
        "claim_has_contradicted",
    ]
    for c in dino_feats + vlm_feats:
        if c not in dev.columns:
            dev[c] = 0.0
            test[c] = 0.0

    y_dev = dev[target].astype(int).values
    rng = np.random.default_rng(seed)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # dino_only
    lr_dino = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_dino.fit(dev[dino_feats].values, y_dev)
    out["dino_only"] = (
        lr_dino.predict_proba(dev[dino_feats].values)[:, 1],
        lr_dino.predict_proba(test[dino_feats].values)[:, 1],
    )

    # vlm_only
    lr_vlm = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_vlm.fit(dev[vlm_feats].values, y_dev)
    out["vlm_only"] = (
        lr_vlm.predict_proba(dev[vlm_feats].values)[:, 1],
        lr_vlm.predict_proba(test[vlm_feats].values)[:, 1],
    )

    # dino_plus_vlm
    both = dino_feats + vlm_feats
    lr_both = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_both.fit(dev[both].values, y_dev)
    out["dino_plus_vlm"] = (
        lr_both.predict_proba(dev[both].values)[:, 1],
        lr_both.predict_proba(test[both].values)[:, 1],
    )

    # margin_only (higher risk when margin low)
    out["margin_only"] = (1.0 - dev["dino_margin"].values, 1.0 - test["dino_margin"].values)

    # class proxies
    dev_top1 = dev["dino_top1"].astype(str)
    test_top1 = test["dino_top1"].astype(str)
    out["review_if_top1_is_defect"] = ((dev_top1 != "insulator_ok").astype(float).values, (test_top1 != "insulator_ok").astype(float).values)
    out["review_if_top1_not_ok"] = ((dev_top1 != "insulator_ok").astype(float).values, (test_top1 != "insulator_ok").astype(float).values)
    out["review_if_top1_flashover"] = ((dev_top1 == "defect_flashover").astype(float).values, (test_top1 == "defect_flashover").astype(float).values)

    # random_review_same_budget / oracle_review (scores used for ranking)
    out["random_review_same_budget"] = (rng.random(len(dev)), rng.random(len(test)))
    out["oracle_review"] = (dev[target].astype(float).values, test[target].astype(float).values)

    return out


def _budget_metrics(df_test: pd.DataFrame, reviewed: np.ndarray) -> dict[str, float]:
    n = len(df_test)
    if n == 0:
        return {
            "review_rate": float("nan"),
            "accepted_accuracy": float("nan"),
            "error_capture_rate": float("nan"),
            "false_alarm_capture_rate": float("nan"),
            "ok_to_flashover_capture_rate": float("nan"),
            "dangerous_miss_capture_rate": float("nan"),
            "correct_rejection_rate": float("nan"),
            "unnecessary_review_rate": float("nan"),
        }
    reviewed = reviewed.astype(bool)
    errors = df_test["general_error"].astype(int).values
    fa = df_test["false_alarm"].astype(int).values
    fa_f = df_test["ok_to_flashover_false_alarm"].astype(int).values
    dm = df_test["dangerous_miss"].astype(int).values
    ok = (errors == 0)
    accepted = ~reviewed
    accepted_acc = float(((errors == 0) & accepted).sum() / max(1, accepted.sum()))
    return {
        "review_rate": float(reviewed.mean()),
        "accepted_accuracy": accepted_acc,
        "error_capture_rate": float((reviewed & (errors == 1)).sum() / max(1, (errors == 1).sum())),
        "false_alarm_capture_rate": float((reviewed & (fa == 1)).sum() / max(1, (fa == 1).sum())),
        "ok_to_flashover_capture_rate": float((reviewed & (fa_f == 1)).sum() / max(1, (fa_f == 1).sum())),
        "dangerous_miss_capture_rate": float((reviewed & (dm == 1)).sum() / max(1, (dm == 1).sum())),
        "correct_rejection_rate": float((reviewed & (ok == 1)).sum() / max(1, reviewed.sum())),
        "unnecessary_review_rate": float((reviewed & (ok == 1)).sum() / max(1, n)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config-json", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = Path(args.splits_dir)
    df = pd.read_csv(args.table)

    targets = TARGETS
    budgets = REVIEW_BUDGETS
    if args.config_json:
        cfg = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        targets = cfg.get("targets", targets)
        budgets = cfg.get("review_budgets", budgets)

    metrics_rows = []
    pred_rows = []
    th_rows = []

    dev_files = sorted(splits_dir.glob("split_*_dev_ids.txt"))
    for dev_file in dev_files:
        split_id = dev_file.stem.replace("_dev_ids", "")
        test_file = splits_dir / f"{split_id}_test_ids.txt"
        if not test_file.exists():
            continue
        dev_ids = set(x.strip() for x in dev_file.read_text(encoding="utf-8").splitlines() if x.strip())
        test_ids = set(x.strip() for x in test_file.read_text(encoding="utf-8").splitlines() if x.strip())
        dev = df[df["record_id"].astype(str).isin(dev_ids)].copy()
        test = df[df["record_id"].astype(str).isin(test_ids)].copy()
        if len(dev) == 0 or len(test) == 0:
            continue

        for target in targets:
            if target not in dev.columns:
                continue
            scores = _build_method_scores(dev, test, target, seed=args.seed + int(split_id.split("_")[-1]))
            y_test = test[target].astype(int).values
            prev = _prevalence(y_test)
            n_pos = int(y_test.sum())
            n_total = int(len(y_test))

            for method, (s_dev, s_test) in scores.items():
                auroc = _safe_auroc(y_test, s_test)
                auprc = _safe_auprc(y_test, s_test)
                metrics_rows.append(
                    {
                        "split_id": split_id,
                        "target": target,
                        "method": method,
                        "metric_group": "ranking",
                        "n_total": n_total,
                        "n_positive": n_pos,
                        "positive_prevalence": prev,
                        "auroc": auroc,
                        "auprc": auprc,
                        "auprc_minus_prevalence": (auprc - prev) if pd.notna(auprc) and pd.notna(prev) else np.nan,
                    }
                )
                for rid, sc in zip(test["record_id"].astype(str).tolist(), s_test.tolist()):
                    pred_rows.append(
                        {
                            "split_id": split_id,
                            "subset": "test",
                            "target": target,
                            "method": method,
                            "record_id": rid,
                            "risk_score": float(sc),
                            "label": int(test.loc[test["record_id"].astype(str) == rid, target].iloc[0]),
                        }
                    )

                for b in budgets:
                    th = _threshold_for_budget(s_dev, float(b))
                    reviewed = s_test >= th
                    bm = _budget_metrics(test, reviewed)
                    th_rows.append(
                        {
                            "split_id": split_id,
                            "target": target,
                            "method": method,
                            "review_budget": float(b),
                            "threshold_selected_on_dev": float(th),
                            "dev_score_mean": float(np.mean(s_dev)),
                            "test_score_mean": float(np.mean(s_test)),
                        }
                    )
                    metrics_rows.append(
                        {
                            "split_id": split_id,
                            "target": target,
                            "method": method,
                            "metric_group": "budget",
                            "review_budget": float(b),
                            "n_total": n_total,
                            "n_positive": n_pos,
                            "positive_prevalence": prev,
                            **bm,
                        }
                    )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_by_split.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out_dir / "per_record_predictions_by_split.csv", index=False)
    pd.DataFrame(th_rows).to_csv(out_dir / "thresholds_by_split.csv", index=False)


if __name__ == "__main__":
    main()
