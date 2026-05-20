#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_MAP = {
    "intact_evidence": {"intact_structure", "regular_disc_shape", "no_visible_damage"},
    "broken_structure_evidence": {"missing_fragment", "edge_discontinuity", "crack", "broken_shed", "structural_damage"},
    "flashover_surface_evidence": {
        "surface_burn_trace",
        "dark_streak",
        "burn_like_mark",
        "arc_trace",
        "carbonization_like_mark",
        "surface_damage_mark",
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


def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip().strip('"').strip("'") for v in x if str(v).strip()]
    s = str(x)
    try:
        y = ast.literal_eval(s)
        if isinstance(y, list):
            return [str(v).strip().strip('"').strip("'") for v in y if str(v).strip()]
    except Exception:
        pass
    return [s.strip()]


def to_family(tags):
    out = set()
    for t in tags:
        for fam, vals in FAMILY_MAP.items():
            if t in vals:
                out.add(fam)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ref", default="outputs/stage12/dev_evidence_dataset/stage12_dev_eval_reference.csv")
    ap.add_argument("--vlm-jsonl", default="outputs/stage12/vlm_evidence_probe/dev_train_182_candidate_support_v1.jsonl")
    ap.add_argument("--out-dir", default="outputs/stage12/structured_output_v2_pilot")
    ap.add_argument("--pilot-n", type=int, default=72)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.dev_ref)
    vlm = pd.read_json(args.vlm_jsonl, lines=True)
    df = ref.merge(vlm, on=["record_id", "split"], how="inner")

    # pilot sampling: include many difficult classes
    hard = df[df["coarse_class"].astype(str).str.startswith("defect_")]
    ok = df[df["coarse_class"].astype(str) == "insulator_ok"]
    n_h = min(len(hard), int(args.pilot_n * 0.65))
    n_o = min(len(ok), args.pilot_n - n_h)
    pilot = pd.concat([hard.head(n_h), ok.head(n_o)], ignore_index=True)

    pilot["gt_tags"] = pilot["visual_evidence_tags"].apply(parse_list)
    pilot["pred_tags"] = pilot["evidence_tags"].apply(parse_list)
    pilot["gt_family"] = pilot["gt_tags"].apply(to_family)
    pilot["pred_family"] = pilot["pred_tags"].apply(to_family)

    labels = sorted(FAMILY_MAP.keys())
    rows = []
    for lab in labels:
        tp = fp = fn = 0
        for g, p in zip(pilot["gt_family"], pilot["pred_family"]):
            gh, ph = lab in g, lab in p
            tp += int(gh and ph)
            fp += int((not gh) and ph)
            fn += int(gh and (not ph))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"family": lab, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    fam = pd.DataFrame(rows)
    fam.to_csv(out / "structured_v2_family_metrics.csv", index=False)
    fam[["family", "tp", "fp", "fn"]].to_csv(out / "structured_v2_family_confusion.csv", index=False)

    ok_controls = pilot[pilot["coarse_class"] == "insulator_ok"]
    hall = 0.0
    if len(ok_controls):
        hall = float(
            ok_controls["pred_family"].apply(
                lambda x: int(("broken_structure_evidence" in x) or ("flashover_surface_evidence" in x))
            ).mean()
        )
    pd.DataFrame([{"hallucinated_defect_family_rate_on_ok": hall, "n_ok_controls": len(ok_controls)}]).to_csv(
        out / "structured_v2_hallucination_metrics.csv", index=False
    )

    nr_acc = float((pilot["needs_review_x"].astype(bool) == pilot["needs_review_y"].astype(bool)).mean())
    pd.DataFrame([{"needs_review_accuracy": nr_acc, "n": len(pilot)}]).to_csv(
        out / "structured_v2_needs_review_metrics.csv", index=False
    )

    pilot.to_json(out / "structured_v2_pilot_outputs.jsonl", orient="records", lines=True, force_ascii=False)
    pilot.to_csv(out / "structured_v2_pilot_predictions.csv", index=False)
    pilot["jaccard"] = [
        (len(set(g) & set(p)) / len(set(g) | set(p))) if len(set(g) | set(p)) else 1.0
        for g, p in zip(pilot["gt_family"], pilot["pred_family"])
    ]
    pilot.sort_values("jaccard", ascending=False).head(30).to_csv(out / "structured_v2_success_cases.csv", index=False)
    pilot.sort_values("jaccard", ascending=True).head(30).to_csv(out / "structured_v2_failure_cases.csv", index=False)
    (out / "structured_v2_prompt.txt").write_text(
        "Operational family-level structured output prompt (pilot reconstruction from existing VLM outputs).",
        encoding="utf-8",
    )
    (out / "structured_v2_report.md").write_text(
        f"Pilot size={len(pilot)}; family-level macro-F1={fam['f1'].mean():.4f}; hallucination_on_ok={hall:.4f}; needs_review_acc={nr_acc:.4f}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
