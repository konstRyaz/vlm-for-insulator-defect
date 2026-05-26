#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


def parse_tags(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip().strip('"').strip("'") for v in x if str(v).strip()]
    s = str(x).strip()
    if not s:
        return []
    try:
        y = ast.literal_eval(s)
        if isinstance(y, list):
            return [str(v).strip().strip('"').strip("'") for v in y if str(v).strip()]
    except Exception:
        pass
    vocab = [
        "missing_fragment",
        "edge_discontinuity",
        "burn_like_mark",
        "surface_damage_mark",
        "dark_surface_trace",
        "intact_structure",
        "regular_disc_shape",
        "blurred_region",
        "partial_view",
        "ambiguous_evidence",
    ]
    return [v for v in vocab if v in s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", default="outputs/stage12/structured_output_v2_binary_real_pilot/pilot_manifest.csv")
    ap.add_argument("--outputs-jsonl", default="outputs/stage12/structured_output_v2_binary_real_pilot/vlm_binary_checklist_outputs.jsonl")
    ap.add_argument("--out-dir", default="outputs/stage12/structured_output_v2_binary_real_pilot")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(args.manifest_csv)
    pred = pd.read_json(args.outputs_jsonl, lines=True)
    df = man.merge(pred, on="record_id", how="inner")

    def gt_has(tags_col, toks):
        return tags_col.apply(lambda x: int(any(t in parse_tags(x) for t in toks))).to_numpy()

    items = [
        ("missing_fragment_visible", {"missing_fragment"}),
        ("edge_discontinuity_visible", {"edge_discontinuity"}),
        ("burn_or_arc_trace_visible", {"burn_like_mark", "surface_damage_mark", "dark_surface_trace"}),
        ("intact_regular_shape_visible", {"intact_structure", "regular_disc_shape"}),
        ("crop_quality_issue", {"blurred_region", "partial_view", "ambiguous_evidence"}),
    ]
    rows = []
    for col, toks in items:
        g = gt_has(df["label_visual_evidence_tags"], toks)
        p = (df[col].astype(str) == "yes").astype(int).to_numpy()
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"item": col, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    m = pd.DataFrame(rows)
    m.to_csv(out / "binary_checklist_metrics.csv", index=False)

    ok = df[df["label_coarse_class"] == "insulator_ok"].copy()
    hall = 0.0
    if len(ok):
        hall = float(
            ok.apply(
                lambda r: int(
                    (str(r.get("missing_fragment_visible")) == "yes")
                    or (str(r.get("edge_discontinuity_visible")) == "yes")
                    or (str(r.get("burn_or_arc_trace_visible")) == "yes")
                    or (str(r.get("dark_surface_trace_visible")) == "yes")
                ),
                axis=1,
            ).mean()
        )
    pd.DataFrame([{"hallucinated_defect_evidence_on_ok": hall, "n_ok_controls": len(ok)}]).to_csv(
        out / "hallucination_on_ok_controls.csv", index=False
    )

    prev_base = np.nan
    prev_path = Path("outputs/stage12/structured_output_v2_pilot/structured_v2_family_metrics.csv")
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        if "f1" in prev.columns:
            prev_base = float(prev["f1"].mean())
    pd.DataFrame(
        [
            {"method": "binary_checklist_real_vlm", "mean_binary_f1": float(m["f1"].mean())},
            {"method": "structured_v2_previous_baseline", "mean_binary_f1": prev_base},
        ]
    ).to_csv(out / "structured_v2_vs_baselines.csv", index=False)

    report = [
        "# Binary Checklist Pilot (Real VLM Run)",
        "",
        f"- n={len(df)}",
        f"- parse_ok_rate={float(df['parse_ok'].mean()):.4f}",
        f"- schema_ok_rate={float(df['schema_ok'].mean()):.4f}",
        f"- mean_binary_f1={float(m['f1'].mean()):.4f}",
        f"- hallucinated_defect_evidence_on_ok={hall:.4f}",
    ]
    (out / "binary_checklist_pilot_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
