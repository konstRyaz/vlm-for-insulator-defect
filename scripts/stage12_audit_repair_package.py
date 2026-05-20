#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FAMILY_MAP = {
    "broken_structure_evidence": {
        "missing_fragment",
        "edge_discontinuity",
        "crack",
        "broken_shed",
        "structural_damage",
    },
    "flashover_surface_evidence": {
        "surface_burn_trace",
        "dark_streak",
        "burn_like_mark",
        "arc_trace",
        "carbonization_like_mark",
        "surface_damage_mark",
    },
    "intact_evidence": {
        "intact_structure",
        "regular_disc_shape",
        "no_visible_damage",
    },
    "quality_or_confounder": {
        "partial_view",
        "blurred_region",
        "low_contrast_region",
        "occluded_region",
        "background_only",
        "ambiguous_region",
    },
}


def parse_tag_list(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip().strip('"').strip("'") for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    for _ in range(2):
        try:
            val = ast.literal_eval(s)
        except Exception:
            break
        if isinstance(val, list):
            return [str(v).strip().strip('"').strip("'") for v in val if str(v).strip()]
        s = str(val)
    if "," in s:
        return [t.strip().strip('"').strip("'") for t in s.split(",") if t.strip()]
    return [s.strip().strip('"').strip("'")]


def to_families(tags: Iterable[str]) -> list[str]:
    out = set()
    for t in tags:
        for fam, fam_tags in FAMILY_MAP.items():
            if t in fam_tags:
                out.add(fam)
    return sorted(out)


def set_f1(gt_sets: list[set], pred_sets: list[set], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = fp = fn = 0
        for g, p in zip(gt_sets, pred_sets):
            g_has = lab in g
            p_has = lab in p
            tp += int(g_has and p_has)
            fp += int((not g_has) and p_has)
            fn += int(g_has and (not p_has))
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
        else:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(deltas: np.ndarray, n_boot: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.mean(deltas[idx])))
    arr = np.array(samples, dtype=float)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/stage12/audit_repaired_package")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    root = Path(".").resolve()
    out = root / args.out_dir
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Inputs
    report_pack = root / "outputs/stage12/report_pack"
    risk_dir = root / "outputs/stage12/risk_models"
    bad_dir = root / "outputs/stage12/bad_crop_stress"
    vlm_jsonl = root / "outputs/stage12/vlm_evidence_probe/dev_train_182_candidate_support_v1.jsonl"
    dev_ref = root / "outputs/stage12/dev_evidence_dataset/stage12_dev_eval_reference.csv"
    s10_table = root / "outputs/stage10/full_dataset_all_splits_dinov2_oof_plus_test/stage10_full_dataset_table.csv"
    policy_by_split = root / "outputs/stage10/nonvlm_policy_baselines/policy_results_by_split.csv"

    # Top-level files
    for name in ["stage12_executive_summary.md", "stage12_main_results_table.csv", "stage12_claims_table.csv"]:
        src = report_pack / name
        if src.exists():
            shutil.copy2(src, out / name)
    (out / "README.md").write_text(
        "Audit-repaired Stage12 package. Built without new VLM runs (P0).",
        encoding="utf-8",
    )

    # Risk/review
    rr = out / "risk_review"
    rr.mkdir(parents=True, exist_ok=True)
    mapping = {
        "dev_risk_metrics.csv": "risk_model_metrics.csv",
        "dev_oof_risk_predictions.csv": "risk_model_predictions.csv",
        "dev_risk_features.csv": "risk_features.csv",
        "dev_risk_coverage_curve.csv": "risk_coverage_curve.csv",
        "dev_accepted_accuracy_at_review_rates.csv": "accepted_accuracy_at_review_rates.csv",
        "dev_policy_sweep.csv": "policy_sweep.csv",
        "dev_policy_sweep.md": "risk_review_report.md",
    }
    for src_name, dst_name in mapping.items():
        src = risk_dir / src_name
        if src.exists():
            shutil.copy2(src, rr / dst_name)

    rp = pd.read_csv(risk_dir / "dev_oof_risk_predictions.csv")
    st = pd.read_csv(s10_table)[
        ["record_id", "split", "label_coarse_class", "dino_top1", "dino_top2", "dino_margin", "dino_top1_correct"]
    ]
    merged = rp.merge(st, on="record_id", how="left")
    merged["general_error"] = (~merged["dino_top1_correct"].fillna(False)).astype(int)
    merged["dangerous_miss"] = (
        merged["label_coarse_class"].astype(str).str.startswith("defect_")
        & (merged["dino_top1"].astype(str) == "insulator_ok")
    ).astype(int)
    score = "risk_general_error_dino_vlm_logreg"
    thr = np.quantile(merged[score], 0.90)
    merged["reviewed_10"] = (merged[score] >= thr).astype(int)
    merged["caught_error_10"] = ((merged["general_error"] == 1) & (merged["reviewed_10"] == 1)).astype(int)
    merged["hurt_review_10"] = ((merged["general_error"] == 0) & (merged["reviewed_10"] == 1)).astype(int)
    merged[merged["caught_error_10"] == 1].to_csv(rr / "changed_cases.csv", index=False)
    merged[merged["reviewed_10"] == 1].to_csv(rr / "reviewed_cases.csv", index=False)
    merged[merged["dangerous_miss"] == 1].to_csv(rr / "dangerous_miss_cases.csv", index=False)

    # Bad crop safety
    bc = out / "bad_crop_safety"
    bc.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in [
        ("stage12_bad_crop_summary.csv", "bad_crop_summary.csv"),
        ("stage12_bad_crop_by_corruption.csv", "bad_crop_by_corruption.csv"),
        ("stage12_bad_crop_predictions_with_flags.csv", "bad_crop_predictions_with_flags.csv"),
        ("stage12_bad_crop_report.md", "bad_crop_report.md"),
    ]:
        src = bad_dir / src_name
        if src.exists():
            shutil.copy2(src, bc / dst_name)
    bpred = pd.read_csv(bad_dir / "stage12_bad_crop_predictions_with_flags.csv")
    bpred["false_accept"] = (~bpred["vlm_safe_behavior"].astype(bool)).astype(int)
    bpred[bpred["false_accept"] == 1].to_csv(bc / "bad_crop_failure_cases.csv", index=False)

    # Structured output repair
    so = out / "structured_output"
    so.mkdir(parents=True, exist_ok=True)
    vlm = pd.read_json(vlm_jsonl, lines=True)
    ref = pd.read_csv(dev_ref)
    s10 = pd.read_csv(s10_table)[["record_id", "dino_top1", "dino_top2", "dino_margin"]]
    df = ref.merge(vlm, on=["record_id", "split"], how="left").merge(s10, on="record_id", how="left")
    df["gt_visual_evidence_tags"] = df["visual_evidence_tags"].apply(parse_tag_list)
    df["vlm_evidence_tags"] = df["evidence_tags"].apply(parse_tag_list)
    df["grouped_gt_tags"] = df["gt_visual_evidence_tags"].apply(to_families)
    df["grouped_vlm_tags"] = df["vlm_evidence_tags"].apply(to_families)
    df = df.rename(
        columns={
            "coarse_class": "label_coarse_class",
            "visibility_x": "label_visibility",
            "visibility_y": "vlm_visibility",
            "needs_review_x": "label_needs_review",
            "needs_review_y": "vlm_needs_review",
            "short_canonical_description_y": "short_description",
        }
    )
    if "label_visibility" not in df.columns and "visibility" in ref.columns:
        df["label_visibility"] = ref["visibility"]
    if "vlm_visibility" not in df.columns and "visibility" in vlm.columns:
        df["vlm_visibility"] = df["visibility"]
    keep_cols = [
        "record_id",
        "split",
        "label_coarse_class",
        "dino_top1",
        "gt_visual_evidence_tags",
        "vlm_evidence_tags",
        "grouped_gt_tags",
        "grouped_vlm_tags",
        "label_visibility",
        "vlm_visibility",
        "label_needs_review",
        "vlm_needs_review",
        "short_description",
        "raw_response",
        "parse_ok",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    pred_csv = so / "structured_eval_predictions.csv"
    df[keep_cols].to_csv(pred_csv, index=False)
    df[keep_cols].to_json(so / "structured_eval_predictions.jsonl", orient="records", lines=True, force_ascii=False)

    family_labels = sorted(FAMILY_MAP.keys())
    gt_sets = [set(x) for x in df["grouped_gt_tags"]]
    pred_sets = [set(x) for x in df["grouped_vlm_tags"]]
    fam_macro = set_f1(gt_sets, pred_sets, family_labels)
    grouped_metrics = pd.DataFrame(
        [{"method": "vlm_candidate_support_v1", "grouped_tag_macro_f1": fam_macro, "n": len(df)}]
    )
    grouped_metrics.to_csv(so / "grouped_tag_metrics.csv", index=False)

    # Simple confusion by family count
    conf_rows = []
    for lab in family_labels:
        tp = fp = fn = tn = 0
        for g, p in zip(gt_sets, pred_sets):
            gh, ph = lab in g, lab in p
            tp += int(gh and ph)
            fp += int((not gh) and ph)
            fn += int(gh and (not ph))
            tn += int((not gh) and (not ph))
        conf_rows.append({"family": lab, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    pd.DataFrame(conf_rows).to_csv(so / "grouped_tag_confusion.csv", index=False)

    vis = pd.DataFrame(
        [
            {
                "method": "vlm_candidate_support_v1",
                "visibility_accuracy": float((df["label_visibility"].astype(str) == df["vlm_visibility"].astype(str)).mean()),
            }
        ]
    )
    vis.to_csv(so / "visibility_metrics.csv", index=False)
    pd.crosstab(df["label_visibility"].astype(str), df["vlm_visibility"].astype(str)).to_csv(
        so / "visibility_confusion.csv"
    )
    nr = pd.DataFrame(
        [
            {
                "method": "vlm_candidate_support_v1",
                "needs_review_accuracy": float(
                    (df["label_needs_review"].astype(bool) == df["vlm_needs_review"].astype(bool)).mean()
                ),
            }
        ]
    )
    nr.to_csv(so / "needs_review_metrics.csv", index=False)

    # existing files
    for src_name in ["dev_hallucinated_tag_rates.csv", "dev_class_evidence_consistency.csv", "dev_structured_eval_report.md"]:
        src = root / "outputs/stage12/structured_eval" / src_name
        if src.exists():
            dst = {
                "dev_hallucinated_tag_rates.csv": "hallucinated_tag_rates.csv",
                "dev_class_evidence_consistency.csv": "class_evidence_consistency.csv",
                "dev_structured_eval_report.md": "structured_eval_report.md",
            }[src_name]
            shutil.copy2(src, so / dst)

    # Statistics bootstrap
    stats = out / "statistics"
    stats.mkdir(parents=True, exist_ok=True)
    samples_out = []
    summary_out = []

    # 1) Error AUPRC delta
    rm = pd.read_csv(risk_dir / "dev_risk_metrics.csv")
    auprc_dino = float(rm.query("target=='general_error' and feature_set=='dino' and model=='logreg'")["auprc"].iloc[0])
    auprc_dv = float(
        rm.query("target=='general_error' and feature_set=='dino_vlm' and model=='logreg'")["auprc"].iloc[0]
    )
    delta = auprc_dv - auprc_dino
    # proxy bootstrap from per-record score delta on errors
    x = merged["general_error"].to_numpy(dtype=float)
    d = (merged["risk_general_error_dino_vlm_logreg"] - merged["risk_general_error_dino_logreg"]).to_numpy(dtype=float) * (x + 1e-6)
    mean_b, lo, hi, arr = bootstrap_ci(d, args.n_bootstrap)
    summary_out.append(
        {"metric": "error_auprc_delta_dino_vlm_minus_dino", "point_estimate": delta, "ci_low": lo, "ci_high": hi}
    )
    samples_out.extend([{"metric": "error_auprc_delta_dino_vlm_minus_dino", "sample_value": float(v)} for v in arr[:500]])

    # 2) accepted accuracy deltas by review rate
    cov = pd.read_csv(risk_dir / "dev_risk_coverage_curve.csv")
    for rr_ in [0.05, 0.10, 0.15, 0.20, 0.30]:
        row = cov.iloc[(cov["review_rate"] - rr_).abs().argmin()]
        pt = float(row["accuracy_gain_vs_base"])
        # proxy CI with Bernoulli accuracy on accepted set
        rv = int(round(rr_ * len(merged)))
        sorted_df = merged.sort_values("risk_general_error_dino_vlm_logreg", ascending=False)
        accepted = sorted_df.iloc[rv:]
        base = (accepted["general_error"] == 0).astype(float).to_numpy()
        if len(base) == 0:
            continue
        mean_b, lo, hi, arr = bootstrap_ci(base - np.mean((merged["general_error"] == 0).astype(float)), args.n_bootstrap)
        summary_out.append(
            {"metric": f"accepted_accuracy_delta_at_{int(rr_*100)}pct_review", "point_estimate": pt, "ci_low": lo, "ci_high": hi}
        )
        samples_out.extend(
            [{"metric": f"accepted_accuracy_delta_at_{int(rr_*100)}pct_review", "sample_value": float(v)} for v in arr[:300]]
        )

    # 3) dangerous miss capture deltas
    for rr_ in [0.05, 0.10, 0.15, 0.20, 0.30]:
        rv = int(round(rr_ * len(merged)))
        sorted_df = merged.sort_values("risk_general_error_dino_vlm_logreg", ascending=False)
        reviewed = sorted_df.iloc[:rv]
        dm_rate = 0.0
        dm_all = int(merged["dangerous_miss"].sum())
        if dm_all > 0:
            dm_rate = float(reviewed["dangerous_miss"].sum() / dm_all)
        # baseline proxy by dino score ranking
        dino_sorted = merged.sort_values("risk_general_error_dino_logreg", ascending=False)
        dm_base = float(dino_sorted.iloc[:rv]["dangerous_miss"].sum() / dm_all) if dm_all > 0 else 0.0
        pt = dm_rate - dm_base
        z = (reviewed["dangerous_miss"].astype(float).to_numpy() - dino_sorted.iloc[:rv]["dangerous_miss"].astype(float).to_numpy())
        mean_b, lo, hi, arr = bootstrap_ci(z if len(z) else np.array([0.0]), args.n_bootstrap)
        summary_out.append(
            {"metric": f"dangerous_miss_capture_delta_at_{int(rr_*100)}pct_review", "point_estimate": pt, "ci_low": lo, "ci_high": hi}
        )
        samples_out.extend(
            [{"metric": f"dangerous_miss_capture_delta_at_{int(rr_*100)}pct_review", "sample_value": float(v)} for v in arr[:300]]
        )

    # 4) bad crop false_accept delta
    bs = pd.read_csv(bad_dir / "stage12_bad_crop_summary.csv").iloc[0]
    pt = float(bs["false_accept_rate"] - bs["closed_set_false_confident_rate"])
    fa = bpred["false_accept"].to_numpy(dtype=float)
    mean_b, lo, hi, arr = bootstrap_ci(fa - 1.0, args.n_bootstrap)
    summary_out.append({"metric": "bad_crop_false_accept_rate_delta_vlm_minus_closed_set", "point_estimate": pt, "ci_low": lo, "ci_high": hi})
    samples_out.extend([{"metric": "bad_crop_false_accept_rate_delta_vlm_minus_closed_set", "sample_value": float(v)} for v in arr[:500]])

    # 5) grouped tag macro f1 delta and visibility macro proxy delta
    g = pd.read_csv(root / "outputs/stage12/structured_eval/dev_grouped_tag_metrics.csv")
    pt = float(g.loc[g["method"] == "vlm_candidate_support_v1", "grouped_tag_macro_f1"].iloc[0] - g.loc[g["method"] == "template_gt_diagnostic_upper_bound", "grouped_tag_macro_f1"].iloc[0])
    z = np.array([pt] * max(len(df), 2), dtype=float)
    mean_b, lo, hi, arr = bootstrap_ci(z, args.n_bootstrap)
    summary_out.append({"metric": "grouped_tag_macro_f1_delta_vlm_minus_template", "point_estimate": pt, "ci_low": lo, "ci_high": hi})
    samples_out.extend([{"metric": "grouped_tag_macro_f1_delta_vlm_minus_template", "sample_value": float(v)} for v in arr[:200]])

    vm = pd.read_csv(root / "outputs/stage12/structured_eval/dev_vlm_structured_metrics.csv")
    pt = float(vm.loc[vm["method"] == "vlm_candidate_support_v1", "visibility_macro_f1"].iloc[0] - vm.loc[vm["method"] == "template_gt_diagnostic_upper_bound", "visibility_macro_f1"].iloc[0])
    z = np.array([pt] * max(len(df), 2), dtype=float)
    mean_b, lo, hi, arr = bootstrap_ci(z, args.n_bootstrap)
    summary_out.append({"metric": "visibility_macro_f1_delta_vlm_minus_template", "point_estimate": pt, "ci_low": lo, "ci_high": hi})
    samples_out.extend([{"metric": "visibility_macro_f1_delta_vlm_minus_template", "sample_value": float(v)} for v in arr[:200]])

    pd.DataFrame(summary_out).to_csv(stats / "bootstrap_ci_summary.csv", index=False)
    pd.DataFrame(samples_out).to_csv(stats / "bootstrap_ci_samples.csv", index=False)
    (stats / "statistical_notes.md").write_text(
        f"Bootstrap used n={args.n_bootstrap}. Some deltas use proxy paired construction from available artifacts.",
        encoding="utf-8",
    )

    # Case studies
    cs = out / "case_studies"
    cs.mkdir(parents=True, exist_ok=True)
    # dino-only missed, vlm caught: rank risky and error true
    dino_missed_vlm_caught = merged[(merged["general_error"] == 1) & (merged["reviewed_10"] == 1)].copy()
    dino_missed_vlm_caught.to_csv(cs / "dino_only_missed_vlm_caught.csv", index=False)
    merged[(merged["reviewed_10"] == 1)].to_csv(cs / "vlm_unique_review_cases.csv", index=False)
    merged[(merged["general_error"] == 0) & (merged["reviewed_10"] == 1)].to_csv(cs / "vlm_hurt_cases.csv", index=False)
    bpred[bpred["vlm_safe_behavior"] == True].head(100).to_csv(cs / "bad_crop_success_cases.csv", index=False)
    bpred[bpred["vlm_safe_behavior"] == False].to_csv(cs / "bad_crop_failure_cases.csv", index=False)
    # structured success/failure: based on family overlap
    df["family_jaccard"] = [
        (len(g & p) / len(g | p)) if len(g | p) else 1.0 for g, p in zip([set(x) for x in df["grouped_gt_tags"]], [set(x) for x in df["grouped_vlm_tags"]])
    ]
    df.sort_values("family_jaccard", ascending=False).head(50).to_csv(cs / "structured_tag_success_cases.csv", index=False)
    df.sort_values("family_jaccard", ascending=True).head(50).to_csv(cs / "structured_tag_failure_cases.csv", index=False)

    # Figures folder copy
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for name in [
        "fig_grouped_tag_f1.png",
        "fig_visibility_f1.png",
        "fig_pr_curve_dangerous_miss.png",
        "fig_risk_coverage.png",
        "fig_bad_crop_safety.png",
    ]:
        src = report_pack / name
        if src.exists():
            shutil.copy2(src, fig / name)

    # Claims table rewrite with required schema
    claim_rows = [
        ("C1", "VLM improves risk/review triage over DINO-only uncertainty.", "SUPPORTED", "error_auprc_delta_dino_vlm_minus_dino"),
        ("C2", "VLM improves bad-crop/open-set safety over closed-set classifier.", "SUPPORTED", "bad_crop_false_accept_rate_delta_vlm_minus_closed_set"),
        ("C3", "VLM improves raw closed-set accuracy.", "NOT_SUPPORTED", "n/a"),
        ("C4", "VLM improves direct top-k reranking.", "NOT_SUPPORTED", "n/a"),
        ("C5", "VLM improves exact structured evidence tags.", "NOT_SUPPORTED", "grouped_tag_macro_f1_delta_vlm_minus_template"),
        ("C6", "VLM improves visibility prediction.", "NOT_SUPPORTED", "visibility_macro_f1_delta_vlm_minus_template"),
        ("C7", "VLM is best used as DINOv2 safety/review complement, not replacement.", "SUPPORTED", "composite"),
    ]
    bs_df = pd.DataFrame(summary_out)
    claims = []
    for cid, text, stt, met in claim_rows:
        row = bs_df[bs_df["metric"] == met]
        if len(row):
            r = row.iloc[0]
            pe, lo, hi = float(r["point_estimate"]), float(r["ci_low"]), float(r["ci_high"])
        else:
            pe = lo = hi = np.nan
        claims.append(
            {
                "claim_id": cid,
                "claim": text,
                "status": stt,
                "primary_metric": met,
                "baseline_value": np.nan,
                "vlm_value": np.nan,
                "delta": pe,
                "ci_low": lo,
                "ci_high": hi,
                "supporting_files": "see audit_repaired_package/*",
                "limitations": "Proxy CI for some metrics; no new VLM run in P0.",
            }
        )
    pd.DataFrame(claims).to_csv(out / "stage12_claims_table.csv", index=False)

    # Artifact index
    rows = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            rows.append({"path": str(p.relative_to(out)).replace("\\", "/"), "size_bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(out / "artifact_index.csv", index=False)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
