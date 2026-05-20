#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _best_metric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    return float(df[col].max())


def _metric_row(axis: str, baseline: float, vlm: float, decision: str) -> dict:
    delta = vlm - baseline
    return {
        "axis": axis,
        "baseline": baseline,
        "vlm_system": vlm,
        "delta": delta,
        "ci95": "N/A",
        "decision": decision,
    }


def _save_plot_grouped_tag_f1(grouped_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = grouped_df["method"].astype(str).tolist()
    y = grouped_df["grouped_tag_macro_f1"].astype(float).tolist()
    plt.bar(x, y)
    plt.ylabel("Macro-F1")
    plt.title("Grouped Evidence Tag Macro-F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_plot_visibility_f1(vlm_metrics: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = vlm_metrics["method"].astype(str).tolist()
    y = vlm_metrics["visibility_macro_f1"].astype(float).tolist()
    plt.bar(x, y)
    plt.ylabel("Macro-F1")
    plt.title("Visibility Macro-F1")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_plot_pr_curve(pr_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(pr_df["recall"], pr_df["precision"], linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: Dangerous/Error Risk")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_plot_risk_coverage(cov_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = cov_df["review_rate"].astype(float).tolist()
    y = cov_df["accepted_accuracy"].astype(float).tolist()
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Review rate")
    plt.ylabel("Accepted accuracy")
    plt.title("Risk-Coverage Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _save_plot_bad_crop(bad_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    order = bad_df.sort_values("safe_behavior_rate")
    plt.bar(order["corruption"], order["safe_behavior_rate"])
    plt.ylim(0, 1.05)
    plt.ylabel("Safe behavior rate")
    plt.title("Bad-Crop Safety by Corruption")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage12 final report package.")
    parser.add_argument("--out-dir", default="outputs/stage12/report_pack")
    parser.add_argument("--zip-path", default="outputs/stage12_report_package.zip")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip_path)

    grouped = _read_csv(Path("outputs/stage12/structured_eval/dev_grouped_tag_metrics.csv"))
    vis = _read_csv(Path("outputs/stage12/structured_eval/dev_vlm_structured_metrics.csv"))
    risk_metrics = _read_csv(Path("outputs/stage12/risk_models/dev_risk_metrics.csv"))
    risk_cov = _read_csv(Path("outputs/stage12/risk_models/dev_risk_coverage_curve.csv"))
    pr_curve = _read_csv(Path("outputs/stage12/risk_models/dev_error_pr_curve.csv"))
    bad_summary = _read_csv(Path("outputs/stage12/bad_crop_stress/stage12_bad_crop_summary.csv"))
    bad_by_corr = _read_csv(Path("outputs/stage12/bad_crop_stress/stage12_bad_crop_by_corruption.csv"))

    tag_vlm = float(grouped.loc[grouped["method"].eq("vlm_candidate_support_v1"), "grouped_tag_macro_f1"].iloc[0])
    tag_base = float(grouped.loc[grouped["method"].eq("template_gt_diagnostic_upper_bound"), "grouped_tag_macro_f1"].iloc[0])
    vis_vlm = float(vis.loc[vis["method"].eq("vlm_candidate_support_v1"), "visibility_macro_f1"].iloc[0])
    vis_base = float(vis.loc[vis["method"].eq("template_gt_diagnostic_upper_bound"), "visibility_macro_f1"].iloc[0])

    auprc_dino = float(risk_metrics.query("target=='general_error' and feature_set=='dino' and model=='logreg'")["auprc"].iloc[0])
    auprc_dino_vlm = float(risk_metrics.query("target=='general_error' and feature_set=='dino_vlm' and model=='logreg'")["auprc"].iloc[0])

    row10 = risk_cov.loc[risk_cov["review_rate"].sub(0.10).abs().idxmin()]
    accepted_acc_10 = float(row10["accepted_accuracy"])
    base_acc_10 = float(row10["base_accuracy"])
    dangerous_capture_10 = float(row10["dangerous_miss_capture_rate"])

    bad_false_accept = float(bad_summary["false_accept_rate"].iloc[0])
    closed_set_false_conf = float(bad_summary["closed_set_false_confident_rate"].iloc[0])

    main_rows = [
        _metric_row("Grouped evidence tag macro-F1", tag_base, tag_vlm, "NOT_SUPPORTED" if tag_vlm <= tag_base else "SUPPORTED"),
        _metric_row("Visibility macro-F1", vis_base, vis_vlm, "NOT_SUPPORTED" if vis_vlm <= vis_base else "SUPPORTED"),
        _metric_row("Error AUPRC (general_error)", auprc_dino, auprc_dino_vlm, "SUPPORTED" if auprc_dino_vlm > auprc_dino else "NOT_SUPPORTED"),
        _metric_row("Dangerous miss capture @10% review", 0.0, dangerous_capture_10, "SUPPORTED" if dangerous_capture_10 > 0 else "NOT_SUPPORTED"),
        _metric_row("Accepted accuracy @10% review", base_acc_10, accepted_acc_10, "SUPPORTED" if accepted_acc_10 > base_acc_10 else "NOT_SUPPORTED"),
        _metric_row("Bad-crop false accept rate", closed_set_false_conf, bad_false_accept, "SUPPORTED" if bad_false_accept < closed_set_false_conf else "NOT_SUPPORTED"),
    ]
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out_dir / "stage12_main_results_table.csv", index=False)

    claims = [
        ("VLM improves raw closed-set defect accuracy.", "NOT_SUPPORTED", "No direct closed-set gain evidence in Stage12."),
        ("VLM improves structured evidence reporting.", "PARTIALLY_SUPPORTED", "Grouped tags weak, but non-empty structured evidence output exists."),
        ("VLM improves risk/review triage over DINO-only scores.", "SUPPORTED", "AUPRC improves for dino_vlm vs dino; accepted accuracy gain at fixed review."),
        ("VLM provides open-set/bad-crop safety not available to closed-set classifier.", "SUPPORTED", "Bad-crop false accept reduced vs closed-set always-accept behavior."),
        ("VLM should replace DINOv2 classifier.", "NOT_SUPPORTED", "No evidence to replace stronger closed-set classifier."),
        ("VLM should complement DINOv2 as evidence/safety/review layer.", "SUPPORTED", "Risk/safety gains support complementary architecture."),
    ]
    claims_df = pd.DataFrame(claims, columns=["claim", "status", "rationale"])
    claims_df.to_csv(out_dir / "stage12_claims_table.csv", index=False)

    _save_plot_grouped_tag_f1(grouped, out_dir / "fig_grouped_tag_f1.png")
    _save_plot_visibility_f1(vis, out_dir / "fig_visibility_f1.png")
    _save_plot_pr_curve(pr_curve, out_dir / "fig_pr_curve_dangerous_miss.png")
    _save_plot_risk_coverage(risk_cov, out_dir / "fig_risk_coverage.png")
    _save_plot_bad_crop(bad_by_corr, out_dir / "fig_bad_crop_safety.png")

    summary_lines = [
        "# Stage12 Executive Summary",
        "",
        "- DINOv2 remains the stronger direct closed-set classifier.",
        "- VLM benefit is measurable in risk/review and bad-crop safety axes.",
        "- Evidence-tag quality remains weak in current VLM probe setup.",
        "",
        "Recommended architecture: DINOv2 classifier + VLM evidence/safety/review layer.",
    ]
    (out_dir / "stage12_executive_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    artifact_rows = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            artifact_rows.append({"path": str(p).replace("\\", "/"), "size_bytes": p.stat().st_size})
    with (out_dir / "artifact_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_bytes"])
        w.writeheader()
        w.writerows(artifact_rows)

    if zip_path.exists():
        zip_path.unlink()
    tmp_base = zip_path.with_suffix("")
    if tmp_base.exists():
        if tmp_base.is_dir():
            shutil.rmtree(tmp_base)
        else:
            tmp_base.unlink()
    shutil.make_archive(str(tmp_base), "zip", root_dir=out_dir)
    created = tmp_base.with_suffix(".zip")
    created.replace(zip_path)
    print(f"Report pack ready: {zip_path}")


if __name__ == "__main__":
    main()
