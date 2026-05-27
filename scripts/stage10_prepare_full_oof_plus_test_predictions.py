import argparse
import json
from pathlib import Path

import pandas as pd


ID_CANDIDATES = ["record_id", "sample_id", "id", "crop_id"]
TOP1_CANDIDATES = ["dino_top1", "top1_class", "pred_coarse_class", "pred_class", "dino_pred", "predicted_class", "pred", "y_pred"]


def read_label_ids(paths: list[str]) -> set[str]:
    ids = set()
    for path_s in paths:
        path = Path(path_s)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(str(json.loads(line).get("record_id")))
    return ids


def infer_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for col in df.columns:
        cl = str(col).lower()
        if any(cand.lower() in cl for cand in candidates):
            # Very short names such as "id" and "pred" create false positives
            # in derived columns like dino_prediction_matched or pred_insulator_ok.
            if any(cand.lower() in {"id", "pred"} and cand.lower() != cl for cand in candidates):
                continue
            return col
    return None


def add_topk_from_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
    if not prob_cols:
        return df
    out = df.copy()
    probs = out[prob_cols].apply(pd.to_numeric, errors="coerce")
    for i, row in probs.iterrows():
        ranked = sorted(
            [(str(c)[5:], float(v)) for c, v in row.items() if pd.notna(v)],
            key=lambda x: x[1],
            reverse=True,
        )
        for rank in range(3):
            if len(ranked) > rank:
                out.loc[i, f"top{rank + 1}_class"] = ranked[rank][0]
                out.loc[i, f"top{rank + 1}_score"] = ranked[rank][1]
    return out


def normalize_predictions(path: Path, protocol: str) -> pd.DataFrame:
    raw = add_topk_from_probability_columns(pd.read_csv(path))
    id_col = infer_col(raw, ID_CANDIDATES)
    top1_col = infer_col(raw, TOP1_CANDIDATES)
    if id_col is None or top1_col is None:
        raise ValueError(f"Cannot infer record_id/top1 columns from {path}")
    top2_col = "dino_top2" if "dino_top2" in raw.columns else ("top2_class" if "top2_class" in raw.columns else None)
    top3_col = "dino_top3" if "dino_top3" in raw.columns else ("top3_class" if "top3_class" in raw.columns else None)
    top1_score_col = "dino_top1_score" if "dino_top1_score" in raw.columns else ("top1_score" if "top1_score" in raw.columns else ("confidence" if "confidence" in raw.columns else None))
    top2_score_col = "dino_top2_score" if "dino_top2_score" in raw.columns else ("top2_score" if "top2_score" in raw.columns else None)
    top3_score_col = "dino_top3_score" if "dino_top3_score" in raw.columns else ("top3_score" if "top3_score" in raw.columns else None)

    out = pd.DataFrame()
    out["record_id"] = raw[id_col].fillna("").astype(str)
    out["prediction_protocol"] = protocol
    out["dino_top1"] = raw[top1_col].fillna("").astype(str)
    out["dino_top2"] = raw[top2_col].fillna("").astype(str) if top2_col else ""
    out["dino_top3"] = raw[top3_col].fillna("").astype(str) if top3_col else ""
    out["dino_top1_score"] = pd.to_numeric(raw[top1_score_col], errors="coerce") if top1_score_col else pd.NA
    out["dino_top2_score"] = pd.to_numeric(raw[top2_score_col], errors="coerce") if top2_score_col else pd.NA
    out["dino_top3_score"] = pd.to_numeric(raw[top3_score_col], errors="coerce") if top3_score_col else pd.NA
    out["dino_margin"] = out["dino_top1_score"] - out["dino_top2_score"]
    out["prediction_source_file"] = str(path)
    return out


def inventory_csvs(search_roots: list[str], dev_ids: set[str], test_ids: set[str]) -> pd.DataFrame:
    rows = []
    seen = set()
    for root_s in search_roots:
        root = Path(root_s)
        if root.is_file():
            paths = [root]
        else:
            paths = list(root.rglob("*.csv")) if root.exists() else []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            path_l = str(path).lower()
            if "outputs\\stage10\\" in path_l or "outputs/stage10/" in path_l:
                continue
            if not any(k in path_l for k in ["dino", "dinov2", "oof", "cv", "feature_backbone", "traincv", "fulltrain"]):
                continue
            try:
                df = pd.read_csv(path, nrows=1000)
            except Exception as exc:
                rows.append({"path": str(path), "read_error": str(exc)})
                continue
            id_col = infer_col(df, ID_CANDIDATES)
            top1_col = infer_col(df, TOP1_CANDIDATES)
            prob_cols = [c for c in df.columns if str(c).startswith("prob_")]
            topk_cols = [c for c in df.columns if str(c) in {"top1_class", "top2_class", "top3_class"}]
            values = set(df[id_col].dropna().astype(str)) if id_col else set()
            rows.append(
                {
                    "path": str(path),
                    "n_preview_rows": len(df),
                    "id_col": id_col or "",
                    "top1_col": top1_col or "",
                    "prob_cols": ";".join(prob_cols),
                    "topk_cols": ";".join(topk_cols),
                    "has_topk_or_probs": bool(id_col and top1_col and (prob_cols or topk_cols)),
                    "dev_overlap": len(values & dev_ids),
                    "test_overlap": len(values & test_ids),
                    "read_error": "",
                }
            )
    inv = pd.DataFrame(rows)
    if len(inv):
        inv = inv.sort_values(["dev_overlap", "test_overlap", "path"], ascending=[False, False, True])
    return inv


def write_readme(out: Path, dev_missing: int, test_missing: int, wrote_unified: bool):
    lines = [
        "# Stage 10 Full OOF + Test DINOv2 Prediction Preparation",
        "",
        "Historical `train` is the development split for VLM/policy tuning.",
        "Historical `val`/`val_v2` is the test split.",
        "",
        "Development predictions must be out-of-fold. Test predictions can be final model predictions.",
        "If development predictions are in-sample, they must not be used for final policy-selection claims.",
        "",
        f"- missing development OOF predictions: {dev_missing}",
        f"- missing test predictions: {test_missing}",
        f"- unified prediction file written: {wrote_unified}",
        "",
        "If the unified file was not written, the required per-record development OOF top-k predictions were not found.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-labels-jsonl", action="append", required=True)
    ap.add_argument("--test-labels-jsonl", action="append", required=True)
    ap.add_argument("--test-predictions-csv", required=True)
    ap.add_argument("--dev-oof-predictions-csv")
    ap.add_argument("--search-root", action="append", default=["outputs"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev_ids = read_label_ids(args.dev_labels_jsonl)
    test_ids = read_label_ids(args.test_labels_jsonl)
    inv = inventory_csvs(args.search_root + [args.test_predictions_csv], dev_ids, test_ids)
    inv.to_csv(out / "stage10_prediction_source_inventory.csv", index=False)

    test_pred = normalize_predictions(Path(args.test_predictions_csv), "test_final")
    test_pred = test_pred[test_pred["record_id"].isin(test_ids)].copy()

    if args.dev_oof_predictions_csv:
        dev_pred = normalize_predictions(Path(args.dev_oof_predictions_csv), "dev_oof")
        dev_pred = dev_pred[dev_pred["record_id"].isin(dev_ids)].copy()
    else:
        # Only accept an automatically found file if it has per-record dev overlap and top-k probabilities.
        viable = inv[(inv["dev_overlap"] > 0) & (inv["has_topk_or_probs"] == True)].copy() if len(inv) else pd.DataFrame()
        if len(viable):
            dev_pred = normalize_predictions(Path(viable.iloc[0]["path"]), "dev_oof")
            dev_pred = dev_pred[dev_pred["record_id"].isin(dev_ids)].copy()
        else:
            dev_pred = pd.DataFrame(columns=test_pred.columns)

    pieces = [df for df in [dev_pred, test_pred] if len(df)]
    available = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=test_pred.columns)
    duplicate = available[available["record_id"].duplicated(keep=False)].sort_values("record_id")
    duplicate.to_csv(out / "stage10_duplicate_prediction_record_ids.csv", index=False)

    dev_missing_ids = sorted(dev_ids - set(dev_pred["record_id"].astype(str)))
    test_missing_ids = sorted(test_ids - set(test_pred["record_id"].astype(str)))
    missing = pd.DataFrame(
        [{"record_id": rid, "split_role": "development", "required_protocol": "dev_oof"} for rid in dev_missing_ids]
        + [{"record_id": rid, "split_role": "test", "required_protocol": "test_final"} for rid in test_missing_ids]
    )
    missing.to_csv(out / "stage10_missing_prediction_record_ids.csv", index=False)

    wrote_unified = False
    if not dev_missing_ids and not test_missing_ids and duplicate.empty:
        available.to_csv(out / "stage10_dinov2_full_oof_plus_test_predictions.csv", index=False)
        wrote_unified = True
    else:
        available.to_csv(out / "stage10_dinov2_available_predictions_not_full.csv", index=False)

    write_readme(out, len(dev_missing_ids), len(test_missing_ids), wrote_unified)
    print(
        {
            "dev_ids": len(dev_ids),
            "test_ids": len(test_ids),
            "dev_predictions": len(dev_pred),
            "test_predictions": len(test_pred),
            "missing_dev_predictions": len(dev_missing_ids),
            "missing_test_predictions": len(test_missing_ids),
            "wrote_unified": wrote_unified,
        }
    )


if __name__ == "__main__":
    main()
