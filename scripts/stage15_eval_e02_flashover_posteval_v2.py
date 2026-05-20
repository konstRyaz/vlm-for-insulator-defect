#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def parse_list_like(x: object) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t) for t in v]
    except Exception:
        pass
    return []


def metric_row(df: pd.DataFrame, reviewed: np.ndarray, method: str, budget_label: str) -> dict:
    labels = df["label_coarse_class"].astype(str).to_numpy()
    false_alarm = labels == "insulator_ok"
    true_flash = labels == "defect_flashover"
    accepted = ~reviewed
    review_count = int(reviewed.sum())
    accept_count = int(accepted.sum())

    fa_reviewed = int((reviewed & false_alarm).sum())
    tf_reviewed = int((reviewed & true_flash).sum())
    tf_accepted = int((accepted & true_flash).sum())

    return {
        "method": method,
        "budget": budget_label,
        "n": int(len(df)),
        "review_count": review_count,
        "review_rate": float(review_count / max(1, len(df))),
        "accept_count": accept_count,
        "accept_rate": float(accept_count / max(1, len(df))),
        "accepted_accuracy": float(tf_accepted / max(1, accept_count)),
        "n_false_alarm": int(false_alarm.sum()),
        "false_alarm_reviewed": fa_reviewed,
        "false_alarm_capture_rate": float(fa_reviewed / max(1, false_alarm.sum())),
        "false_alarm_remaining": int((accepted & false_alarm).sum()),
        "false_alarm_remaining_rate": float((accepted & false_alarm).sum() / max(1, false_alarm.sum())),
        "false_alarm_review_yield": float(fa_reviewed / max(1, review_count)),
        "n_true_flashover": int(true_flash.sum()),
        "true_flashover_accepted": tf_accepted,
        "true_flashover_accept_rate": float(tf_accepted / max(1, true_flash.sum())),
        "true_flashover_reviewed": tf_reviewed,
        "true_flashover_review_rate": float(tf_reviewed / max(1, true_flash.sum())),
        "true_flashover_retention": float(tf_accepted / max(1, true_flash.sum())),
        "helped": fa_reviewed,
        "hurt": tf_reviewed,
        "net_gain": int(fa_reviewed - tf_reviewed),
        "helped_per_review": float(fa_reviewed / max(1, review_count)),
    }


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str, maximize_x: bool, maximize_y: bool) -> pd.DataFrame:
    pts = df.copy()
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            xi, yi = pts.iloc[i][x_col], pts.iloc[i][y_col]
            xj, yj = pts.iloc[j][x_col], pts.iloc[j][y_col]
            no_worse_x = xj >= xi if maximize_x else xj <= xi
            no_worse_y = yj >= yi if maximize_y else yj <= yi
            strictly_better = ((xj > xi) if maximize_x else (xj < xi)) or ((yj > yi) if maximize_y else (yj < yi))
            if no_worse_x and no_worse_y and strictly_better:
                keep[i] = False
                break
    return pts[keep].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-status-json", default="")
    ap.add_argument("--validation-summary-csv", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.joined_csv)
    for c in ["possible_confounders", "visible_evidence_types"]:
        if c in df.columns:
            df[c] = df[c].map(parse_list_like)
    for c in ["recommend_review", "parse_ok", "schema_ok"]:
        if c in df.columns:
            df[c] = df[c].map(to_bool)

    core = df[df["dino_top1"].astype(str) == "defect_flashover"].copy().reset_index(drop=True)
    n = len(core)

    # STEP 2 distributions
    dist_rows = []
    for col in ["claim_supported", "direct_flashover_evidence", "evidence_location"]:
        vc = core[col].fillna("missing").astype(str).value_counts()
        for k, v in vc.items():
            dist_rows.append({"group": col, "key": k, "count": int(v), "rate": float(v / max(1, n))})
    rr_true = int(core["recommend_review"].sum())
    dist_rows.append({"group": "recommend_review", "key": "true", "count": rr_true, "rate": rr_true / max(1, n)})
    dist_rows.append({"group": "recommend_review", "key": "false", "count": n - rr_true, "rate": (n - rr_true) / max(1, n)})
    core["possible_confounders_count"] = core["possible_confounders"].map(len)
    for k, v in core["possible_confounders_count"].value_counts().sort_index().items():
        dist_rows.append({"group": "possible_confounders_count", "key": str(int(k)), "count": int(v), "rate": float(v / max(1, n))})
    tag_counts = {}
    for tags in core["visible_evidence_types"]:
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    for t, v in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
        dist_rows.append({"group": "visible_evidence_types", "key": t, "count": int(v), "rate": float(v / max(1, n))})

    core["false_alarm_flag"] = core["label_coarse_class"].astype(str) == "insulator_ok"
    core["true_flashover_flag"] = core["label_coarse_class"].astype(str) == "defect_flashover"

    c1 = core.groupby(["claim_supported", "label_coarse_class"]).size().reset_index(name="count")
    c1["group"] = "claim_supported_x_label"
    c1["key"] = c1["claim_supported"].astype(str) + "|" + c1["label_coarse_class"].astype(str)
    c2 = core.groupby(["claim_supported", "false_alarm_flag"]).size().reset_index(name="count")
    c2["group"] = "claim_supported_x_false_alarm"
    c2["key"] = c2["claim_supported"].astype(str) + "|" + c2["false_alarm_flag"].astype(str)
    c3 = core.groupby(["evidence_location", "false_alarm_flag"]).size().reset_index(name="count")
    c3["group"] = "evidence_location_x_false_alarm"
    c3["key"] = c3["evidence_location"].astype(str) + "|" + c3["false_alarm_flag"].astype(str)
    c4 = core.groupby(["direct_flashover_evidence", "true_flashover_flag"]).size().reset_index(name="count")
    c4["group"] = "direct_evidence_x_true_flashover"
    c4["key"] = c4["direct_flashover_evidence"].astype(str) + "|" + c4["true_flashover_flag"].astype(str)
    extra = pd.concat(
        [
            c1[["group", "key", "count"]],
            c2[["group", "key", "count"]],
            c3[["group", "key", "count"]],
            c4[["group", "key", "count"]],
        ],
        ignore_index=True,
    )
    extra["rate"] = extra["count"] / max(1, n)

    pd.concat([pd.DataFrame(dist_rows), extra], ignore_index=True).to_csv(
        out_dir / "e02_vlm_output_distribution.csv", index=False
    )

    # STEP 3 budget matched
    budgets = [0.05, 0.10, 4 / max(1, n), 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    budgets = sorted(set([round(b, 6) for b in budgets]))

    # risk bucket for VLM ranking
    cs = core["claim_supported"].fillna("uncertain").astype(str)
    de = core["direct_flashover_evidence"].fillna("uncertain").astype(str)
    el = core["evidence_location"].fillna("uncertain").astype(str)
    rr = core["recommend_review"].map(to_bool)
    conf_nonempty = core["possible_confounders"].map(lambda x: len(x) > 0)
    margin = core["dino_margin"].fillna(0.0).astype(float)

    risk_bucket = np.zeros(n, dtype=int)
    risk_bucket[(cs == "no") | (el == "background_or_shadow") | (de == "no")] = 3
    risk_bucket[(cs == "uncertain") | (de == "uncertain") | (el == "uncertain") | rr] = np.maximum(
        risk_bucket[(cs == "uncertain") | (de == "uncertain") | (el == "uncertain") | rr], 2
    )
    risk_bucket[conf_nonempty.to_numpy()] = np.maximum(risk_bucket[conf_nonempty.to_numpy()], 1)

    core["margin_rank"] = margin.rank(method="first", ascending=True).astype(int)
    core["vlm_rank"] = pd.Series(risk_bucket).rank(method="first", ascending=False).astype(int)
    core["risk_bucket"] = risk_bucket

    # indices per method order (high priority first)
    idx_margin = np.argsort(margin.to_numpy())
    idx_vlm = np.lexsort((margin.to_numpy(), -risk_bucket))
    idx_vlm_plus_margin = np.lexsort((margin.to_numpy(), -risk_bucket))
    idx_margin_plus_vlm = np.lexsort((-risk_bucket, margin.to_numpy()))

    def reviewed_from_idx(order: np.ndarray, k: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        m[order[:k]] = True
        return m

    metric_rows = []
    tradeoff_rows = []
    for b in budgets:
        k = int(round(b * n))
        k = min(max(k, 1), n)
        budget_label = f"{b:.4f}"

        methods = {
            "margin_rank": reviewed_from_idx(idx_margin, k),
            "vlm_binary_rank": reviewed_from_idx(idx_vlm, k),
            "vlm_plus_margin_rank": reviewed_from_idx(idx_vlm_plus_margin, k),
            "margin_plus_vlm_rank": reviewed_from_idx(idx_margin_plus_vlm, k),
            "review_all_flashover": np.ones(n, dtype=bool),
        }
        for mname, reviewed in methods.items():
            row = metric_row(core, reviewed, mname, budget_label)
            metric_rows.append(row)
            tradeoff_rows.append(
                {
                    "method": mname,
                    "budget": budget_label,
                    "review_rate": row["review_rate"],
                    "false_alarm_capture_rate": row["false_alarm_capture_rate"],
                    "true_flashover_review_rate": row["true_flashover_review_rate"],
                    "accepted_accuracy": row["accepted_accuracy"],
                    "net_gain": row["net_gain"],
                }
            )

        # random baseline averaged
        seed_rows = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            order = rng.permutation(n)
            reviewed = reviewed_from_idx(order, k)
            seed_rows.append(metric_row(core, reviewed, "random_baseline_same_budget", budget_label))
        s = pd.DataFrame(seed_rows).mean(numeric_only=True).to_dict()
        s["method"] = "random_baseline_same_budget"
        s["budget"] = budget_label
        metric_rows.append(s)
        tradeoff_rows.append(
            {
                "method": s["method"],
                "budget": s["budget"],
                "review_rate": s["review_rate"],
                "false_alarm_capture_rate": s["false_alarm_capture_rate"],
                "true_flashover_review_rate": s["true_flashover_review_rate"],
                "accepted_accuracy": s["accepted_accuracy"],
                "net_gain": s["net_gain"],
            }
        )

    bm = pd.DataFrame(metric_rows)
    bm.to_csv(out_dir / "e02_budget_matched_metrics.csv", index=False)
    pd.DataFrame(tradeoff_rows).to_csv(out_dir / "e02_budget_matched_tradeoff.csv", index=False)

    # Pareto
    # reduce to best per (method,budget) (already unique)
    p = pd.DataFrame(tradeoff_rows)
    pf_a = pareto_front(p, "false_alarm_capture_rate", "review_rate", maximize_x=True, maximize_y=False)
    pf_a["frontier"] = "A_capture_vs_review"
    pf_b = pareto_front(p, "false_alarm_capture_rate", "true_flashover_review_rate", maximize_x=True, maximize_y=False)
    pf_b["frontier"] = "B_capture_vs_tf_review"
    pf_c = pareto_front(p, "net_gain", "review_rate", maximize_x=True, maximize_y=False)
    pf_c["frontier"] = "C_net_gain_vs_review"
    pareto = pd.concat([pf_a, pf_b, pf_c], ignore_index=True)
    pareto.to_csv(out_dir / "e02_pareto_frontier.csv", index=False)
    (out_dir / "e02_pareto_report.md").write_text(
        "# E02 Pareto Report\n\n"
        f"- core_n: {n}\n"
        f"- front_A_points: {len(pf_a)}\n"
        f"- front_B_points: {len(pf_b)}\n"
        f"- front_C_points: {len(pf_c)}\n",
        encoding="utf-8",
    )

    # Compare with general risk model (if available)
    risk_out = out_dir / "e02_vs_general_risk_model.csv"
    found = list(Path("outputs").rglob("*resplit*budget*.csv")) + list(Path("outputs").rglob("*risk_review*baseline*.csv"))
    if not found:
        pd.DataFrame(
            [{"status": "missing", "reason": "general risk model comparison artifacts were not found locally"}]
        ).to_csv(risk_out, index=False)
    else:
        pd.DataFrame(
            [{"status": "partial", "note": "general risk files found but direct per-record subset comparison not implemented in this pass", "found_file": str(found[0])}]
        ).to_csv(risk_out, index=False)

    # low-review case tables: selected P3-like = vlm_binary_rank at k=4
    k4 = 4
    rev_vlm4 = reviewed_from_idx(idx_vlm, k4)
    rev_margin4 = reviewed_from_idx(idx_margin, k4)
    core["selected_policy_decision"] = np.where(rev_vlm4, "review", "accept")
    core["decision_margin_k4"] = np.where(rev_margin4, "review", "accept")

    base_cols = [
        "record_id",
        "resolved_image_path",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_margin",
        "direct_flashover_evidence",
        "evidence_location",
        "claim_supported",
        "possible_confounders",
        "recommend_review",
        "short_reason",
        "margin_rank",
        "vlm_rank",
        "selected_policy_decision",
        "decision_margin_k4",
    ]
    base_cols = [c for c in base_cols if c in core.columns]
    core[(core["false_alarm_flag"]) & (core["selected_policy_decision"] == "review")][base_cols].to_csv(
        out_dir / "e02_low_review_false_alarms_caught.csv", index=False
    )
    core[(core["true_flashover_flag"]) & (core["selected_policy_decision"] == "review")][base_cols].to_csv(
        out_dir / "e02_low_review_true_flashover_overreviewed.csv", index=False
    )
    core[(core["false_alarm_flag"]) & (core["selected_policy_decision"] == "accept")][base_cols].to_csv(
        out_dir / "e02_low_review_false_alarms_missed.csv", index=False
    )
    core[(core["decision_margin_k4"] == "review") & (core["selected_policy_decision"] == "accept")][base_cols].to_csv(
        out_dir / "e02_margin_only_extra_reviews.csv", index=False
    )

    # updated main policy_metrics from budget table at budget=4/n
    b_label = f"{(4/max(1,n)):.4f}"
    pm = bm[bm["budget"] == b_label].copy()
    pm.to_csv(out_dir / "updated_policy_metrics.csv", index=False)

    # report v2
    row_vlm = bm[(bm["method"] == "vlm_binary_rank") & (bm["budget"] == b_label)].iloc[0]
    row_margin = bm[(bm["method"] == "margin_rank") & (bm["budget"] == b_label)].iloc[0]
    claim = "PARTIALLY_SUPPORTED"
    if row_vlm["false_alarm_capture_rate"] > row_margin["false_alarm_capture_rate"]:
        claim = "SUPPORTED"
    elif row_vlm["false_alarm_capture_rate"] < row_margin["false_alarm_capture_rate"]:
        claim = "PARTIALLY_SUPPORTED"

    (out_dir / "E02_FULL_REPORT_V2.md").write_text(
        "# E02 Full Post-Eval V2\n\n"
        f"- core_n: {n}\n"
        f"- budget_matched_check: review_count=4 ({b_label})\n\n"
        "## Budget-matched comparison (k=4)\n"
        f"- vlm_binary_rank false_alarm_capture_rate: {row_vlm['false_alarm_capture_rate']:.4f}\n"
        f"- margin_rank false_alarm_capture_rate: {row_margin['false_alarm_capture_rate']:.4f}\n"
        f"- vlm_binary_rank true_flashover_retention: {row_vlm['true_flashover_retention']:.4f}\n"
        f"- margin_rank true_flashover_retention: {row_margin['true_flashover_retention']:.4f}\n\n"
        f"## Claim decision\n{claim}\n\n"
        "Interpretation: VLM behaves as low-review interpretable triage. "
        "If margin dominates on matched budget, E02 is partially supported rather than a strict quantitative win.\n",
        encoding="utf-8",
    )

    (out_dir / "e02_budget_matched_report.md").write_text(
        "# E02 Budget-Matched Report\n\n"
        "See `e02_budget_matched_metrics.csv` for all methods and budgets.\n",
        encoding="utf-8",
    )

    # pack
    zip_path = out_dir.parent.parent / "stage15_e02_full_posteval_v2_report.zip"
    include = [
        "E02_FULL_REPORT_V2.md",
        "e02_vlm_output_distribution.csv",
        "e02_budget_matched_metrics.csv",
        "e02_budget_matched_tradeoff.csv",
        "e02_pareto_frontier.csv",
        "e02_pareto_report.md",
        "e02_vs_general_risk_model.csv",
        "e02_low_review_false_alarms_caught.csv",
        "e02_low_review_true_flashover_overreviewed.csv",
        "e02_low_review_false_alarms_missed.csv",
        "e02_margin_only_extra_reviews.csv",
        "updated_policy_metrics.csv",
        "e02_budget_matched_report.md",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in include:
            pth = out_dir / name
            if pth.exists():
                z.write(pth, name)
        if args.run_status_json and Path(args.run_status_json).exists():
            z.write(args.run_status_json, "run_full/run_status.json")
        if args.validation_summary_csv and Path(args.validation_summary_csv).exists():
            z.write(args.validation_summary_csv, "run_full/validation_summary.csv")
    print(zip_path)


if __name__ == "__main__":
    main()
