#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from stage12_eval_grouped_tags import to_group_set


def parse_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip().strip('"') for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out: list[str] = []
            for x in v:
                t = str(x).strip().strip('"').strip("'").strip("[]")
                if t:
                    out.append(t)
            return [x for x in out if x]
    except Exception:
        pass
    s = s.replace("[", " ").replace("]", " ").replace('"', " ").replace("'", " ")
    return [x.strip() for x in s.split(",") if x.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def set_metrics(y_true: list[set[str]], y_pred: list[set[str]]) -> dict[str, float]:
    jacc = []
    for a, b in zip(y_true, y_pred):
        u = len(a | b)
        jacc.append(1.0 if u == 0 else len(a & b) / u)
    all_labels = sorted(set().union(*y_true, *y_pred))
    if not all_labels:
        return {"tag_exact_micro_f1": 1.0, "tag_exact_macro_f1": 1.0, "tag_mean_jaccard": 1.0}
    yt, yp = [], []
    for a, b in zip(y_true, y_pred):
        yt.append([1 if l in a else 0 for l in all_labels])
        yp.append([1 if l in b else 0 for l in all_labels])
    return {
        "tag_exact_micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "tag_exact_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "tag_mean_jaccard": float(np.mean(jacc)),
    }


def class_template(cls: str) -> tuple[list[str], str, bool]:
    if cls == "insulator_ok":
        return ["intact_structure", "regular_disc_shape", "no_visible_break"], "clear", False
    if cls == "defect_broken":
        return ["missing_fragment", "edge_discontinuity", "surface_damage_mark"], "clear", False
    if cls == "defect_flashover":
        return ["burn_like_mark", "dark_surface_trace", "surface_stain"], "clear", False
    return ["ambiguous_evidence"], "ambiguous", True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--vlm-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.reference_csv)
    ref = ref.rename(columns={"coarse_class": "label_coarse_class", "visibility": "label_visibility", "needs_review": "label_needs_review"})
    ref["label_tags"] = ref["visual_evidence_tags"].apply(parse_tags)
    ref["label_groups"] = ref["label_tags"].apply(lambda x: sorted(to_group_set(x)))

    vlm = pd.DataFrame(read_jsonl(Path(args.vlm_jsonl)))
    vlm = vlm[["record_id", "visible_insulator", "visibility", "needs_review", "evidence_tags", "short_canonical_description"]].copy()
    vlm = vlm.rename(columns={"short_canonical_description": "pred_short_canonical_description"})
    vlm["pred_tags"] = vlm["evidence_tags"].apply(parse_tags)
    vlm["pred_groups"] = vlm["pred_tags"].apply(lambda x: sorted(to_group_set(x)))
    df = ref.merge(vlm, on="record_id", how="inner")

    yv = df["label_visibility"].astype(str).tolist()
    pv = df["visibility"].astype(str).tolist()
    vt = [set(x) for x in df["label_tags"]]
    vp = [set(x) for x in df["pred_tags"]]
    gtg = [set(x) for x in df["label_groups"]]
    vpg = [set(x) for x in df["pred_groups"]]

    base_rows = []

    m = {
        "method": "vlm_candidate_support_v1",
        "n": len(df),
        "visibility_accuracy": float(accuracy_score(yv, pv)),
        "visibility_macro_f1": float(f1_score(yv, pv, average="macro", zero_division=0)),
        "description_non_empty_rate": float((df["pred_short_canonical_description"].astype(str).str.len() > 0).mean()),
    }
    m.update(set_metrics(vt, vp))
    gm = set_metrics(gtg, vpg)
    m["grouped_tag_micro_f1"] = gm["tag_exact_micro_f1"]
    m["grouped_tag_macro_f1"] = gm["tag_exact_macro_f1"]
    base_rows.append(m)

    # B1 template baseline from DINO top1.
    temp_tags, temp_vis, temp_rev = [], [], []
    for cls in df["label_coarse_class"].astype(str).tolist():
        t, v, r = class_template(cls)
        temp_tags.append(set(t))
        temp_vis.append(v)
        temp_rev.append(r)
    b1 = {
        "method": "template_gt_diagnostic_upper_bound",
        "n": len(df),
        "visibility_accuracy": float(accuracy_score(yv, temp_vis)),
        "visibility_macro_f1": float(f1_score(yv, temp_vis, average="macro", zero_division=0)),
        "description_non_empty_rate": 0.0,
    }
    b1.update(set_metrics(vt, temp_tags))
    gb1 = set_metrics(gtg, [to_group_set(x) for x in temp_tags])
    b1["grouped_tag_micro_f1"] = gb1["tag_exact_micro_f1"]
    b1["grouped_tag_macro_f1"] = gb1["tag_exact_macro_f1"]
    base_rows.append(b1)

    metrics_df = pd.DataFrame(base_rows)
    metrics_df.to_csv(out_dir / "dev_vlm_structured_metrics.csv", index=False)

    halluc = pd.DataFrame(
        [
            {
                "method": "vlm_candidate_support_v1",
                "hallucinated_defect_tag_rate_on_ok": float(
                    (
                        df[df["label_coarse_class"] == "insulator_ok"]["pred_tags"]
                        .apply(lambda x: any(t in {"missing_fragment", "edge_discontinuity", "burn_like_mark", "dark_surface_trace"} for t in x))
                    ).mean()
                ),
                "unsupported_tag_rate": float(
                    np.mean([len(set(p) - set(t)) / max(1, len(set(p))) for p, t in zip(df["pred_tags"], df["label_tags"])])
                ),
            }
        ]
    )
    halluc.to_csv(out_dir / "dev_hallucinated_tag_rates.csv", index=False)

    consistency = pd.DataFrame(
        [
            {
                "method": "vlm_candidate_support_v1",
                "class_evidence_consistency": float(
                    (
                        (df["label_coarse_class"] == "insulator_ok") & df["pred_tags"].apply(lambda x: "intact_structure" in x or "regular_disc_shape" in x)
                        | (df["label_coarse_class"] == "defect_broken") & df["pred_tags"].apply(lambda x: "missing_fragment" in x or "edge_discontinuity" in x)
                        | (df["label_coarse_class"] == "defect_flashover") & df["pred_tags"].apply(lambda x: "burn_like_mark" in x or "dark_surface_trace" in x or "surface_damage_mark" in x)
                    ).mean()
                )
            }
        ]
    )
    consistency.to_csv(out_dir / "dev_class_evidence_consistency.csv", index=False)

    grouped = metrics_df[["method", "grouped_tag_micro_f1", "grouped_tag_macro_f1"]].copy()
    grouped.to_csv(out_dir / "dev_grouped_tag_metrics.csv", index=False)
    metrics_df.to_csv(out_dir / "dev_template_baselines.csv", index=False)
    pd.DataFrame([{"method": "random_baseline", "note": "not_run"}]).to_csv(out_dir / "dev_random_baselines.csv", index=False)

    report = out_dir / "dev_structured_eval_report.md"
    report.write_text(
        "\n".join(
            [
                "# Stage12 Structured Eval (Dev)",
                "",
                f"- n: {len(df)}",
                f"- grouped_tag_macro_f1 (VLM): {metrics_df.loc[metrics_df.method=='vlm_candidate_support_v1','grouped_tag_macro_f1'].iloc[0]:.4f}",
                f"- grouped_tag_macro_f1 (template upper): {metrics_df.loc[metrics_df.method=='template_gt_diagnostic_upper_bound','grouped_tag_macro_f1'].iloc[0]:.4f}",
                f"- visibility_macro_f1 (VLM): {metrics_df.loc[metrics_df.method=='vlm_candidate_support_v1','visibility_macro_f1'].iloc[0]:.4f}",
                f"- class_evidence_consistency: {consistency['class_evidence_consistency'].iloc[0]:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
