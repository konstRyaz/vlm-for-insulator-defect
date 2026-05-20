#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def as_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def pick_margin_threshold(df: pd.DataFrame) -> float:
    dev = df[df["split"] == "train"].copy()
    if dev.empty:
        dev = df.copy()
    cands = sorted(set(np.round(dev["dino_margin"].fillna(0.0).astype(float), 4).tolist()))
    if not cands:
        return 0.2
    best_thr = cands[0]
    best_score = -1.0
    fa = ((dev["label_coarse_class"] == "insulator_ok") & (dev["dino_top1"] == "defect_flashover")).values
    tf = ((dev["label_coarse_class"] == "defect_flashover") & (dev["dino_top1"] == "defect_flashover")).values
    for thr in cands:
        review = dev["dino_margin"].fillna(0.0).astype(float).values <= thr
        fa_cap = float((review & fa).sum() / max(1, fa.sum()))
        tf_keep = float(((~review) & tf).sum() / max(1, tf.sum()))
        score = fa_cap + tf_keep
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def policy_decision(df: pd.DataFrame, name: str, margin_thr: float) -> np.ndarray:
    claim_supported = df["claim_supported"].fillna("uncertain").astype(str)
    evidence = df["direct_flashover_evidence"].fillna("uncertain").astype(str)
    loc = df["evidence_location"].fillna("uncertain").astype(str)
    confounders_nonempty = (
        df["possible_confounders"]
        .apply(lambda x: isinstance(x, list) and len(x) > 0)
        .astype(bool)
    )
    rec_review = df["recommend_review"].map(as_bool)
    low_margin = df["dino_margin"].fillna(0.0).astype(float) <= float(margin_thr)

    if name == "P0_dino_accept_all":
        return np.zeros(len(df), dtype=bool)
    if name == "P1_review_all_flashover":
        return np.ones(len(df), dtype=bool)
    if name == "P2_vlm_strict":
        accept = (claim_supported == "yes") & (evidence == "yes") & (loc == "on_insulator")
        return ~accept
    if name == "P3_vlm_balanced":
        review = (claim_supported == "no") | (loc == "background_or_shadow") | (rec_review & (claim_supported != "yes"))
        return review.to_numpy()
    if name == "P4_vlm_confounder_aware":
        review = (claim_supported != "yes") | (confounders_nonempty & (evidence != "yes"))
        return review.to_numpy()
    if name == "P5_dino_margin_only":
        return low_margin.to_numpy()
    if name == "P6_dino_vlm_hybrid":
        review = low_margin | (claim_supported != "yes") | ((loc == "background_or_shadow") & (evidence != "yes"))
        return review.to_numpy()
    if name == "P7_random_same_rate_as_p3":
        target_rate = ((claim_supported == "no") | (loc == "background_or_shadow") | (rec_review & (claim_supported != "yes"))).mean()
        rng = np.random.default_rng(42)
        return rng.random(len(df)) < float(target_rate)
    raise ValueError(name)


def compute_metrics(df: pd.DataFrame, policy: str, review_mask: np.ndarray, split_name: str) -> dict:
    y = df["label_coarse_class"].astype(str).values
    pred = df["dino_top1"].astype(str).values
    is_correct = pred == y
    false_alarm = (y == "insulator_ok") & (pred == "defect_flashover")
    true_flash = (y == "defect_flashover") & (pred == "defect_flashover")
    accepted = ~review_mask
    accepted_correct = accepted & is_correct

    baseline_correct = is_correct
    policy_correct = accepted_correct  # reviewed are unresolved, not auto-corrected
    helped = int((~baseline_correct & policy_correct).sum())
    hurt = int((baseline_correct & ~policy_correct).sum())

    return {
        "policy": policy,
        "split": split_name,
        "n_candidates": int(len(df)),
        "review_rate": float(review_mask.mean()) if len(df) else 0.0,
        "accept_rate": float(accepted.mean()) if len(df) else 0.0,
        "accepted_accuracy": float(accepted_correct.sum() / max(1, accepted.sum())),
        "helped": helped,
        "hurt": hurt,
        "net_gain": int(helped - hurt),
        "false_alarm_capture_rate": float((review_mask & false_alarm).sum() / max(1, false_alarm.sum())),
        "false_alarm_remaining_rate": float((accepted & false_alarm).sum() / max(1, false_alarm.sum())),
        "false_alarm_review_yield": float((review_mask & false_alarm).sum() / max(1, review_mask.sum())),
        "true_flashover_accept_rate": float((accepted & true_flash).sum() / max(1, true_flash.sum())),
        "true_flashover_review_rate": float((review_mask & true_flash).sum() / max(1, true_flash.sum())),
        "true_flashover_retention": float((accepted & true_flash).sum() / max(1, true_flash.sum())),
    }


def save_case_tables(df: pd.DataFrame, decisions: pd.Series, out_dir: Path) -> None:
    base_cols = [
        "record_id",
        "resolved_image_path",
        "split",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_margin",
        "claim_supported",
        "direct_flashover_evidence",
        "evidence_location",
        "possible_confounders",
        "recommend_review",
        "short_reason",
    ]
    cols = [c for c in base_cols if c in df.columns]
    w = df.copy()
    w["policy_decision"] = decisions.values

    fa_caught = w[
        (w["label_coarse_class"] == "insulator_ok")
        & (w["dino_top1"] == "defect_flashover")
        & (w["policy_decision"] == "review")
    ]
    tf_accept = w[
        (w["label_coarse_class"] == "defect_flashover")
        & (w["dino_top1"] == "defect_flashover")
        & (w["policy_decision"] == "accept")
    ]
    tf_overreview = w[
        (w["label_coarse_class"] == "defect_flashover")
        & (w["dino_top1"] == "defect_flashover")
        & (w["policy_decision"] == "review")
    ]
    ambiguous = w[
        (w["claim_supported"].astype(str).isin(["uncertain", "no"]))
        | (w["evidence_location"].astype(str).isin(["uncertain", "background_or_shadow"]))
    ]

    fa_caught[cols + ["policy_decision"]].to_csv(out_dir / "false_alarm_caught_by_e02.csv", index=False)
    tf_accept[cols + ["policy_decision"]].to_csv(out_dir / "true_flashover_accepted_by_e02.csv", index=False)
    tf_overreview[cols + ["policy_decision"]].to_csv(out_dir / "true_flashover_overreviewed_by_e02.csv", index=False)
    ambiguous[cols + ["policy_decision"]].to_csv(out_dir / "e02_failure_or_ambiguous_cases.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--outputs-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest_csv)
    o = load_jsonl(Path(args.outputs_jsonl))
    df = m.merge(o, on="record_id", how="left", suffixes=("", "_vlm"))

    for col in ["parse_ok", "schema_ok"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
        else:
            df[col] = False
    if "runtime_error" in df.columns:
        df["runtime_error"] = df["runtime_error"].fillna("").astype(str)
    else:
        df["runtime_error"] = ""

    margin_thr = pick_margin_threshold(df)
    policies = [
        "P0_dino_accept_all",
        "P1_review_all_flashover",
        "P2_vlm_strict",
        "P3_vlm_balanced",
        "P4_vlm_confounder_aware",
        "P5_dino_margin_only",
        "P6_dino_vlm_hybrid",
        "P7_random_same_rate_as_p3",
    ]

    rows_all: list[dict] = []
    splits = {"all": df}
    if "split" in df.columns:
        for s in sorted(df["split"].dropna().astype(str).unique()):
            splits[s] = df[df["split"].astype(str) == s].copy()

    for sname, sdf in splits.items():
        for p in policies:
            rmask = policy_decision(sdf, p, margin_thr)
            rows_all.append(compute_metrics(sdf, p, rmask, sname))

    metrics = pd.DataFrame(rows_all)
    metrics.to_csv(out_dir / "policy_metrics_by_split.csv", index=False)
    metrics[metrics["split"] == "all"].to_csv(out_dir / "policy_metrics.csv", index=False)

    # Case studies for main policy P3.
    p3_mask = policy_decision(df, "P3_vlm_balanced", margin_thr)
    decisions = pd.Series(np.where(p3_mask, "review", "accept"), index=df.index)
    save_case_tables(df, decisions, out_dir)

    report = [
        "# E02 Full Policy Evaluation",
        "",
        f"- n_candidates: {len(df)}",
        f"- margin_threshold_selected_on_dev: {margin_thr:.4f}",
        f"- parse_ok_rate: {df['parse_ok'].mean():.4f}",
        f"- schema_ok_rate: {df['schema_ok'].mean():.4f}",
        f"- runtime_error_rate: {(df['runtime_error'].str.len() > 0).mean():.4f}",
        "",
        "## Main Table",
        "",
        "See `policy_metrics.csv` and `policy_metrics_by_split.csv`.",
    ]
    (out_dir / "E02_FULL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    summary = {
        "n_candidates": int(len(df)),
        "margin_threshold_selected_on_dev": float(margin_thr),
        "parse_ok_rate": float(df["parse_ok"].mean()),
        "schema_ok_rate": float(df["schema_ok"].mean()),
        "runtime_error_rate": float((df["runtime_error"].str.len() > 0).mean()),
    }
    (out_dir / "e02_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
