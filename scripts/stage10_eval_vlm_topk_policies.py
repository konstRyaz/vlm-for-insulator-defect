#!/usr/bin/env python3
"""Evaluate Stage 10 VLM top-k outputs with offline policies.

Policy selection must be done on the historical train/development split. The
historical val split should be used only after a policy is fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASSES = ["insulator_ok", "defect_flashover", "defect_broken"]
DEFECTS = {"defect_flashover", "defect_broken"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def load_join(vlm_jsonl: Path, eval_reference_csv: Path) -> pd.DataFrame:
    vlm = pd.DataFrame(read_jsonl(vlm_jsonl))
    ref = pd.read_csv(eval_reference_csv)
    if "label_coarse_class" not in ref.columns:
        raise ValueError("eval reference must contain label_coarse_class")
    keep = [
        "record_id",
        "split",
        "label_coarse_class",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
    ]
    ref = ref[[c for c in keep if c in ref.columns]].copy()
    df = ref.merge(vlm, on=["record_id", "split"], how="inner", suffixes=("", "_vlm"))
    if len(df) != len(vlm):
        missing = sorted(set(vlm["record_id"].astype(str)) - set(df["record_id"].astype(str)))
        raise ValueError(f"Could not join all VLM rows to eval reference. Missing sample: {missing[:10]}")
    return df


def selected_known(row: pd.Series) -> str | None:
    cls = str(row.get("selected_class", "unknown"))
    if cls in CLASSES and cls in {row.get("dino_top1"), row.get("dino_top2"), row.get("dino_top3")}:
        return cls
    rank = str(row.get("selected_rank", "uncertain"))
    if rank in {"top1", "top2", "top3"}:
        cls = str(row.get(f"dino_{rank}", ""))
        if cls in CLASSES:
            return cls
    return None


def apply_policy(df: pd.DataFrame, policy: str, margin_threshold: float = 0.30) -> pd.DataFrame:
    out = df.copy()
    pred: list[str] = []
    review: list[bool] = []
    for _, row in out.iterrows():
        top1 = str(row["dino_top1"])
        chosen = selected_known(row)
        rank = str(row.get("selected_rank", "uncertain"))
        strength = str(row.get("evidence_strength", "none"))
        quality = str(row.get("crop_quality", "ambiguous"))
        needs_review = safe_bool(row.get("needs_review", False))
        margin = float(row.get("dino_margin", 1.0))

        p = top1
        r = False
        if policy == "dino_top1":
            pass
        elif policy == "vlm_selected_always":
            if chosen is not None:
                p = chosen
            else:
                r = True
        elif policy == "switch_strong":
            if chosen is not None and chosen != top1 and strength == "strong":
                p = chosen
        elif policy == "switch_strong_or_medium_clear":
            if chosen is not None and chosen != top1 and strength in {"strong", "medium"} and quality == "clear":
                p = chosen
        elif policy == "switch_strong_low_margin":
            if chosen is not None and chosen != top1 and strength == "strong" and margin <= margin_threshold:
                p = chosen
        elif policy == "review_uncertain":
            r = needs_review or rank == "uncertain" or chosen is None
        elif policy == "hybrid_strong_switch_review_uncertain":
            if chosen is not None and chosen != top1 and strength == "strong":
                p = chosen
            elif needs_review or rank == "uncertain" or chosen is None or quality in {"bad", "ambiguous"}:
                r = True
        else:
            raise ValueError(f"Unknown policy: {policy}")
        pred.append(p)
        review.append(r)

    out["policy"] = policy if policy != "switch_strong_low_margin" else f"{policy}_{margin_threshold:.2f}"
    out["policy_pred"] = pred
    out["policy_review"] = review
    return out


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["label_coarse_class"].astype(str)
    p = df["policy_pred"].astype(str)
    review = df["policy_review"].astype(bool)
    dino = df["dino_top1"].astype(str)
    dino_correct = dino == y
    policy_correct = p == y
    accepted = ~review
    dino_errors = ~dino_correct
    correct_top1 = dino_correct

    out: dict[str, Any] = {
        "policy": str(df["policy"].iloc[0]) if len(df) else "",
        "split": str(df["split"].iloc[0]) if len(df["split"].unique()) == 1 else "mixed",
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, p)) if len(df) else 0.0,
        "macro_f1": float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)) if len(df) else 0.0,
        "coverage": float(accepted.mean()) if len(df) else 0.0,
        "review_rate": float(review.mean()) if len(df) else 0.0,
        "accepted_accuracy": float(policy_correct[accepted].mean()) if int(accepted.sum()) else 0.0,
        "switch_rate": float((p != dino).mean()) if len(df) else 0.0,
        "helped": int((dino_errors & policy_correct).sum()),
        "hurt": int((correct_top1 & ~policy_correct).sum()),
        "net_gain": int((dino_errors & policy_correct).sum() - (correct_top1 & ~policy_correct).sum()),
        "error_capture_rate": float((review & dino_errors).sum() / dino_errors.sum()) if int(dino_errors.sum()) else 0.0,
        "correct_rejection_rate": float((review & correct_top1).sum() / correct_top1.sum()) if int(correct_top1.sum()) else 0.0,
    }
    for cls in CLASSES:
        mask = y == cls
        out[f"recall_{cls}"] = float(policy_correct[mask].mean()) if int(mask.sum()) else 0.0
    defects = y.isin(DEFECTS)
    out["dangerous_miss_rate"] = float((p[defects] == "insulator_ok").mean()) if int(defects.sum()) else 0.0
    return out


def evaluate(args: argparse.Namespace) -> None:
    df = load_join(Path(args.vlm_jsonl), Path(args.eval_reference_csv))
    if args.split != "all":
        df = df[df["split"].astype(str) == args.split].copy()
    if df.empty:
        raise ValueError(f"No rows selected for split={args.split}")

    policy_frames: list[pd.DataFrame] = []
    base_policies = [
        "dino_top1",
        "vlm_selected_always",
        "switch_strong",
        "switch_strong_or_medium_clear",
        "review_uncertain",
        "hybrid_strong_switch_review_uncertain",
    ]
    for policy in base_policies:
        policy_frames.append(apply_policy(df, policy))
    for threshold in args.margin_thresholds:
        policy_frames.append(apply_policy(df, "switch_strong_low_margin", threshold))

    pred_df = pd.concat(policy_frames, ignore_index=True)
    metrics_df = pd.DataFrame([metrics(g) for _, g in pred_df.groupby("policy", sort=False)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "stage10_vlm_policy_predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "stage10_vlm_policy_metrics.csv", index=False)

    lines = [
        "# Stage 10 VLM Top-k Offline Policy Metrics",
        "",
        f"- VLM outputs: `{args.vlm_jsonl}`",
        f"- eval split: `{args.split}`",
        "- Policy metrics use ground truth for evaluation only, not for inference.",
        "",
        "| policy | n | accuracy | macro_f1 | review_rate | switch_rate | helped | hurt | net_gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_df.to_dict(orient="records"):
        lines.append(
            f"| {row['policy']} | {int(row['n'])} | {float(row['accuracy']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['review_rate']):.4f} | "
            f"{float(row['switch_rate']):.4f} | {int(row['helped'])} | {int(row['hurt'])} | {int(row['net_gain'])} |"
        )
    (out_dir / "stage10_vlm_policy_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_dir / 'stage10_vlm_policy_metrics.csv'}")
    print(f"Wrote: {out_dir / 'stage10_vlm_policy_metrics.md'}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-jsonl", required=True)
    parser.add_argument("--eval-reference-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    parser.add_argument("--margin-thresholds", nargs="*", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    evaluate(args)


if __name__ == "__main__":
    main()
