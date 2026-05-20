#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _to_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _list_nonempty(v: object) -> bool:
    return isinstance(v, list) and len(v) > 0


def _extract_parsed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = r.get("parsed")
        if not isinstance(p, dict):
            p = {}
        rows.append(
            {
                "record_id": r["record_id"],
                "direct_flashover_evidence": p.get("direct_flashover_evidence", "uncertain"),
                "evidence_location": p.get("evidence_location", "uncertain"),
                "visible_evidence_types": p.get("visible_evidence_types", []),
                "possible_confounders": p.get("possible_confounders", []),
                "claim_supported": p.get("claim_supported", "uncertain"),
                "recommend_review": bool(p.get("recommend_review", False)),
                "short_reason": p.get("short_reason", ""),
            }
        )
    return pd.DataFrame(rows)


def _pick_margin_threshold_on_dev(core: pd.DataFrame) -> tuple[float, str]:
    if "split" not in core.columns:
        return float(core["dino_margin"].quantile(0.25)), "diagnostic_no_split"
    dev = core[core["split"].astype(str) == "train"].copy()
    if dev.empty:
        return float(core["dino_margin"].quantile(0.25)), "diagnostic_no_dev_rows"
    y_fa = (dev["label_coarse_class"] == "insulator_ok").to_numpy()
    y_tf = (dev["label_coarse_class"] == "defect_flashover").to_numpy()
    margins = dev["dino_margin"].fillna(0.0).astype(float).to_numpy()
    best_thr, best_score = float(np.quantile(margins, 0.25)), -1e9
    for thr in np.unique(np.quantile(margins, np.linspace(0.05, 0.95, 19))):
        review = margins <= thr
        fa_cap = (review & y_fa).sum() / max(1, y_fa.sum())
        tf_keep = ((~review) & y_tf).sum() / max(1, y_tf.sum())
        score = fa_cap + tf_keep
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, "dev_selected"


def _policy_masks(core: pd.DataFrame, margin_thr: float) -> dict[str, np.ndarray]:
    claim_supported = core["claim_supported"].fillna("uncertain").astype(str)
    direct = core["direct_flashover_evidence"].fillna("uncertain").astype(str)
    loc = core["evidence_location"].fillna("uncertain").astype(str)
    conf = core["possible_confounders"].apply(_list_nonempty)
    rr = core["recommend_review"].map(_to_bool)
    low_margin = core["dino_margin"].fillna(0.0).astype(float) <= margin_thr

    out: dict[str, np.ndarray] = {}
    out["P0_dino_accept_all"] = np.zeros(len(core), dtype=bool)
    out["P1_review_all_flashover"] = np.ones(len(core), dtype=bool)
    out["P2_vlm_strict"] = ~((claim_supported == "yes") & (direct == "yes") & (loc == "on_insulator")).to_numpy()
    out["P3_vlm_balanced"] = ((claim_supported == "no") | (loc == "background_or_shadow") | (rr & (claim_supported != "yes"))).to_numpy()
    out["P4_vlm_confounder_aware"] = ((claim_supported != "yes") | (conf & (direct != "yes"))).to_numpy()
    out["P5_vlm_review_if_uncertain_or_no"] = claim_supported.isin(["no", "uncertain"]).to_numpy()
    out["P6_margin_only"] = low_margin.to_numpy()
    out["P7_hybrid_margin_vlm"] = ((claim_supported != "yes") | low_margin).to_numpy()
    return out


def _metrics_for_policy(core: pd.DataFrame, policy: str, review: np.ndarray) -> dict:
    y = core["label_coarse_class"].astype(str).to_numpy()
    false_alarm = y == "insulator_ok"
    true_flash = y == "defect_flashover"
    accepted = ~review
    review_count = int(review.sum())
    accept_count = int(accepted.sum())
    fa_reviewed = int((review & false_alarm).sum())
    tf_reviewed = int((review & true_flash).sum())
    tf_accepted = int((accepted & true_flash).sum())
    accepted_accuracy = float((accepted & true_flash).sum() / max(1, accept_count))
    return {
        "policy": policy,
        "n": int(len(core)),
        "review_count": review_count,
        "review_rate": float(review_count / max(1, len(core))),
        "accept_count": accept_count,
        "accept_rate": float(accept_count / max(1, len(core))),
        "accepted_accuracy": accepted_accuracy,
        "accepted_false_alarm_rate": float((accepted & false_alarm).sum() / max(1, accept_count)),
        "accepted_true_flashover_rate": float((accepted & true_flash).sum() / max(1, accept_count)),
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


def _build_report(out_dir: Path, run_validity: dict, manifest_summary: pd.DataFrame, core_summary: pd.DataFrame, pm: pd.DataFrame, threshold_note: str, threshold: float) -> str:
    main = pm.sort_values(["net_gain", "false_alarm_capture_rate", "true_flashover_retention"], ascending=[False, False, False]).iloc[0]
    claim = "PARTIALLY_SUPPORTED"
    if main["false_alarm_capture_rate"] > 0.5 and main["review_rate"] < 0.95 and main["true_flashover_review_rate"] < 1.0:
        claim = "SUPPORTED"
    if main["false_alarm_capture_rate"] < 0.2:
        claim = "NOT_SUPPORTED"
    lines = [
        "# E02_FULL_REPORT",
        "",
        "## 1. Run validity",
        f"- gpu: {run_validity.get('gpu_name','unknown')}",
        f"- model: Qwen/Qwen2.5-VL-3B-Instruct",
        f"- n: {run_validity.get('n','')}",
        f"- image_load_success_rate: {run_validity.get('image_load_success_rate','')}",
        f"- parse_ok_rate: {run_validity.get('parse_ok_rate','')}",
        f"- schema_ok_rate: {run_validity.get('schema_ok_rate','')}",
        f"- runtime_error_rate: {run_validity.get('runtime_error_rate','')}",
        "",
        "## 2. Candidate set",
        manifest_summary.to_string(index=False),
        "",
        "Core subset summary:",
        core_summary.to_string(index=False),
        "",
        "## 3. Policy results",
        pm.to_string(index=False),
        "",
        "## 4. Main selected policy",
        f"- selected_policy: {main['policy']}",
        f"- false_alarm_capture_rate: {main['false_alarm_capture_rate']:.4f}",
        f"- true_flashover_retention: {main['true_flashover_retention']:.4f}",
        f"- review_rate: {main['review_rate']:.4f}",
        f"- accepted_accuracy: {main['accepted_accuracy']:.4f}",
        "",
        "## 5. Comparison to baselines",
        "- Includes P0_dino_accept_all, P1_review_all_flashover, P6_margin_only.",
        f"- margin_threshold: {threshold:.6f} ({threshold_note})",
        "",
        f"## 6. Claim decision: {claim}",
        "",
        "## 7. Limitations",
        "- dataset is small",
        "- review action prevents automatic error but does not auto-correct class",
        "- no field validation yet",
        "- controls may be present beyond dino_top1 flashover core subset",
    ]
    text = "\n".join(lines)
    (out_dir / "E02_FULL_REPORT.md").write_text(text, encoding="utf-8")
    return claim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--raw-outputs-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--validation-csv", default="")
    ap.add_argument("--run-status-json", default="")
    ap.add_argument("--stdout-log", default="")
    ap.add_argument("--stderr-log", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest_csv)
    r = _read_jsonl(Path(args.raw_outputs_jsonl))
    p = _extract_parsed(r)
    base = r.drop(columns=["parsed"], errors="ignore").merge(p, on="record_id", how="left")
    j = m.merge(base, on="record_id", how="left")

    # normalize booleans
    for c in ["parse_ok", "schema_ok", "image_load_success", "recommend_review"]:
        if c in j.columns:
            j[c] = j[c].map(_to_bool)

    j.to_csv(out_dir / "e02_full_joined_predictions.csv", index=False)

    # manifest summaries
    summary_rows = [
        {"metric": "n_total", "value": int(len(m))},
        {"metric": "n_dino_top1_flashover", "value": int((m["dino_top1"] == "defect_flashover").sum())},
        {"metric": "n_label_insulator_ok", "value": int((m["label_coarse_class"] == "insulator_ok").sum())},
        {"metric": "n_label_defect_flashover", "value": int((m["label_coarse_class"] == "defect_flashover").sum())},
        {"metric": "n_label_defect_broken", "value": int((m["label_coarse_class"] == "defect_broken").sum())},
        {
            "metric": "n_ok_to_flashover_false_alarm",
            "value": int(((m["label_coarse_class"] == "insulator_ok") & (m["dino_top1"] == "defect_flashover")).sum()),
        },
        {
            "metric": "n_true_flashover",
            "value": int(((m["label_coarse_class"] == "defect_flashover") & (m["dino_top1"] == "defect_flashover")).sum()),
        },
    ]
    manifest_summary = pd.DataFrame(summary_rows)
    manifest_summary.to_csv(out_dir / "e02_manifest_summary.csv", index=False)

    core = j[j["dino_top1"].astype(str) == "defect_flashover"].copy()
    core_summary = pd.DataFrame(
        [
            {"metric": "n_core", "value": int(len(core))},
            {"metric": "n_core_false_alarm", "value": int((core["label_coarse_class"] == "insulator_ok").sum())},
            {"metric": "n_core_true_flashover", "value": int((core["label_coarse_class"] == "defect_flashover").sum())},
            {"metric": "n_core_broken", "value": int((core["label_coarse_class"] == "defect_broken").sum())},
        ]
    )
    core_summary.to_csv(out_dir / "e02_core_subset_summary.csv", index=False)

    margin_thr, thr_note = _pick_margin_threshold_on_dev(core)
    masks = _policy_masks(core, margin_thr)

    # policy metrics + split version
    all_rows = []
    split_rows = []
    for pol, mask in masks.items():
        all_rows.append(_metrics_for_policy(core, pol, mask))
        if "split" in core.columns:
            for s in sorted(core["split"].dropna().astype(str).unique()):
                sub = core[core["split"].astype(str) == s].copy()
                if len(sub):
                    sub_mask = _policy_masks(sub, margin_thr)[pol]
                    row = _metrics_for_policy(sub, pol, sub_mask)
                    row["split"] = s
                    split_rows.append(row)

    pm = pd.DataFrame(all_rows)
    pm.to_csv(out_dir / "policy_metrics.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "policy_metrics_by_split.csv", index=False)
    pm[
        ["policy", "review_rate", "false_alarm_capture_rate", "true_flashover_review_rate", "accepted_accuracy", "net_gain"]
    ].to_csv(out_dir / "policy_tradeoff_points.csv", index=False)

    # decisions table
    full_with_decisions = core.copy()
    for pol, mask in masks.items():
        full_with_decisions[f"decision_{pol}"] = np.where(mask, "review", "accept")
    full_with_decisions["false_alarm_flag"] = full_with_decisions["label_coarse_class"] == "insulator_ok"
    full_with_decisions["true_flashover_flag"] = full_with_decisions["label_coarse_class"] == "defect_flashover"
    full_with_decisions.to_csv(out_dir / "e02_all_predictions_with_policy_decisions.csv", index=False)

    selected_policy = "P3_vlm_balanced" if "P3_vlm_balanced" in masks else list(masks.keys())[0]
    sel = np.where(masks[selected_policy], "review", "accept")
    t = full_with_decisions.copy()
    t["selected_policy"] = selected_policy
    t["selected_policy_decision"] = sel

    keep = [
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
        "selected_policy",
        "selected_policy_decision",
    ]
    keep = [c for c in keep if c in t.columns]

    t[
        (t["label_coarse_class"] == "insulator_ok")
        & (t["selected_policy_decision"] == "review")
    ][keep].to_csv(out_dir / "false_alarm_caught_by_e02.csv", index=False)
    t[
        (t["label_coarse_class"] == "defect_flashover")
        & (t["selected_policy_decision"] == "accept")
    ][keep].to_csv(out_dir / "true_flashover_accepted_by_e02.csv", index=False)
    t[
        (t["label_coarse_class"] == "defect_flashover")
        & (t["selected_policy_decision"] == "review")
    ][keep].to_csv(out_dir / "true_flashover_overreviewed_by_e02.csv", index=False)
    t[
        (t["claim_supported"].astype(str) != "yes")
        | (t["evidence_location"].astype(str) != "on_insulator")
        | (t["possible_confounders"].apply(_list_nonempty))
    ][keep].to_csv(out_dir / "e02_ambiguous_or_confounder_cases.csv", index=False)

    run_valid = {
        "gpu_name": "unknown",
        "n": int(len(j)),
        "image_load_success_rate": float(j["image_load_success"].mean()) if "image_load_success" in j else float("nan"),
        "parse_ok_rate": float(j["parse_ok"].mean()) if "parse_ok" in j else float("nan"),
        "schema_ok_rate": float(j["schema_ok"].mean()) if "schema_ok" in j else float("nan"),
        "runtime_error_rate": float((j["runtime_error"].astype(str).str.len() > 0).mean()) if "runtime_error" in j else float("nan"),
    }
    if args.run_status_json and Path(args.run_status_json).exists():
        try:
            rs = json.loads(Path(args.run_status_json).read_text(encoding="utf-8"))
            run_valid["gpu_name"] = rs.get("device_diagnostics", {}).get("gpu_name", "unknown")
        except Exception:
            pass
    if args.validation_csv and Path(args.validation_csv).exists():
        vv = pd.read_csv(args.validation_csv).iloc[0].to_dict()
        for k in ["n", "image_load_success_rate", "parse_ok_rate", "schema_ok_rate", "runtime_error_rate"]:
            if k in vv:
                run_valid[k] = vv[k]

    _build_report(out_dir, run_valid, manifest_summary, core_summary, pm, thr_note, margin_thr)

    # copy runtime artifacts
    run_dir = out_dir / "run_full"
    run_dir.mkdir(parents=True, exist_ok=True)
    for src in [args.run_status_json, args.validation_csv, args.raw_outputs_jsonl]:
        if src and Path(src).exists():
            shutil.copy2(src, run_dir / Path(src).name)
    for src in [args.stdout_log, args.stderr_log]:
        if src and Path(src).exists():
            shutil.copy2(src, out_dir / Path(src).name)


if __name__ == "__main__":
    main()
