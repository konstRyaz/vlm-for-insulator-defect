import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["defect_broken", "defect_flashover", "insulator_ok"]
DEFAULT_FEATURE_MODEL = "facebook/dinov2-base"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["_jsonl_path"] = str(path)
            rec["_jsonl_line"] = line_no
            rows.append(rec)
    return rows


def resolve_image_path(rec: dict, roots: list[Path]) -> Path:
    raw = rec.get("crop_path") or rec.get("image_path") or rec.get("path")
    if not raw:
        raise FileNotFoundError(f"Record {rec.get('record_id')} has no crop/image path.")
    path = Path(str(raw))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    for root in roots:
        candidates.append(root / path)
        candidates.append(root / path.name)
        candidates.append(root / "crops" / path)
    jsonl_parent = Path(rec["_jsonl_path"]).parent
    candidates.extend(
        [
            jsonl_parent / path,
            jsonl_parent.parent / path,
            jsonl_parent.parent / "crops" / path,
        ]
    )
    record_id = str(rec.get("record_id", ""))
    if record_id:
        for root in roots + [jsonl_parent, jsonl_parent.parent]:
            candidates.extend(root.rglob(f"{record_id}.*"))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    unique = []
    seen = set()
    for candidate in candidates[:20]:
        key = str(candidate)
        if key not in seen:
            unique.append(key)
            seen.add(key)
    raise FileNotFoundError(
        f"Could not resolve crop for record_id={rec.get('record_id')} crop_path={raw}. "
        f"Tried examples: {unique}"
    )


def build_frame(label_paths: list[Path], image_roots: list[Path]) -> pd.DataFrame:
    rows = []
    for label_path in label_paths:
        for rec in read_jsonl(label_path):
            record_id = str(rec.get("record_id", ""))
            label = str(rec.get("coarse_class", ""))
            if not record_id or not label:
                continue
            image_path = resolve_image_path(rec, image_roots)
            rows.append(
                {
                    "record_id": record_id,
                    "split": rec.get("split", ""),
                    "true_class": label,
                    "crop_path": rec.get("crop_path", ""),
                    "resolved_image_path": str(image_path),
                    "label_jsonl_path": str(label_path),
                }
            )
    if not rows:
        raise ValueError("No usable records read from labels JSONL.")
    return pd.DataFrame(rows).drop_duplicates("record_id", keep="first")


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def extract_features(
    df: pd.DataFrame,
    model_id: str,
    batch_size: int,
    device: str,
    cache_path: Path | None,
    reuse_features: bool,
) -> np.ndarray:
    if reuse_features and cache_path and cache_path.exists():
        return np.load(cache_path)

    import torch
    from transformers import AutoImageProcessor, AutoModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    feats = []
    paths = [Path(p) for p in df["resolved_image_path"].tolist()]
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_feats = outputs.pooler_output
            else:
                batch_feats = outputs.last_hidden_state[:, 0]
            feats.append(batch_feats.detach().cpu().float().numpy())

    arr = np.concatenate(feats, axis=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
    return arr


def topk_from_scores(scores: np.ndarray, classes: list[str]) -> list[dict]:
    rows = []
    for row in scores:
        ranked_idx = list(np.argsort(row)[::-1])
        rec = {}
        for rank in range(3):
            if len(ranked_idx) > rank:
                idx = ranked_idx[rank]
                rec[f"dino_top{rank + 1}"] = classes[idx]
                rec[f"dino_top{rank + 1}_score"] = float(row[idx])
            else:
                rec[f"dino_top{rank + 1}"] = ""
                rec[f"dino_top{rank + 1}_score"] = math.nan
        rec["dino_margin"] = rec["dino_top1_score"] - rec["dino_top2_score"]
        rows.append(rec)
    return rows


def aligned_predict_proba(clf, x: np.ndarray, classes: list[str]) -> np.ndarray:
    proba = clf.predict_proba(x)
    aligned = np.zeros((len(x), len(classes)), dtype=float)
    class_to_idx = {str(c): i for i, c in enumerate(clf.classes_)}
    for out_i, cls in enumerate(classes):
        if cls in class_to_idx:
            aligned[:, out_i] = proba[:, class_to_idx[cls]]
    return aligned


def make_classifier(c_value: float, class_weight: str | None, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        ),
    )


def generate_oof(
    df: pd.DataFrame,
    features: np.ndarray,
    n_splits: int,
    seed: int,
    c_value: float,
    class_weight: str | None,
    feature_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df["true_class"].astype(str).to_numpy()
    counts = Counter(y)
    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError(f"Cannot run stratified OOF: minimum class count is {min_count}. Counts: {dict(counts)}")
    folds = min(n_splits, min_count)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    pred_rows = []
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(features, y)):
        clf = make_classifier(c_value, class_weight, seed)
        clf.fit(features[train_idx], y[train_idx])
        scores = aligned_predict_proba(clf, features[val_idx], CLASSES)
        topk_rows = topk_from_scores(scores, CLASSES)
        y_val = y[val_idx]
        y_pred = [r["dino_top1"] for r in topk_rows]
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "macro_f1": float(f1_score(y_val, y_pred, labels=CLASSES, average="macro", zero_division=0)),
            }
        )
        for idx, topk in zip(val_idx, topk_rows):
            rec = df.iloc[idx].to_dict()
            pred_rows.append(
                {
                    "record_id": rec["record_id"],
                    "split": rec["split"],
                    "fold": fold,
                    "prediction_protocol": "dev_oof",
                    "true_class": rec["true_class"],
                    **topk,
                    "feature_model": feature_model,
                    "classifier_model": f"LogisticRegression(C={c_value}, class_weight={class_weight})",
                    "prediction_source": "reconstructed_stage10_dev_oof",
                    "resolved_image_path": rec["resolved_image_path"],
                }
            )

    pred_df = pd.DataFrame(pred_rows).sort_values("record_id")
    fold_df = pd.DataFrame(fold_rows)
    cm = pd.DataFrame(
        confusion_matrix(pred_df["true_class"], pred_df["dino_top1"], labels=CLASSES),
        index=CLASSES,
        columns=CLASSES,
    )
    return pred_df, fold_df, cm


def normalize_existing_test_predictions(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    prob_cols = [c for c in raw.columns if str(c).startswith("prob_")]
    if "record_id" not in raw.columns:
        raise ValueError(f"Existing test predictions must contain record_id: {path}")
    if prob_cols:
        rows = []
        for _, row in raw.iterrows():
            ranked = sorted(
                [(str(c)[5:], float(row[c])) for c in prob_cols if pd.notna(row[c])],
                key=lambda x: x[1],
                reverse=True,
            )
            rec = {"record_id": str(row["record_id"]), "prediction_protocol": "test_final"}
            for rank in range(3):
                if len(ranked) > rank:
                    rec[f"dino_top{rank + 1}"] = ranked[rank][0]
                    rec[f"dino_top{rank + 1}_score"] = ranked[rank][1]
                else:
                    rec[f"dino_top{rank + 1}"] = ""
                    rec[f"dino_top{rank + 1}_score"] = math.nan
            rec["dino_margin"] = rec["dino_top1_score"] - rec["dino_top2_score"]
            rec["prediction_source_file"] = str(path)
            rows.append(rec)
        return pd.DataFrame(rows)

    top1_col = "dino_top1" if "dino_top1" in raw.columns else "pred_coarse_class"
    if top1_col not in raw.columns:
        raise ValueError(f"Cannot infer top1/probability columns from {path}")
    out = pd.DataFrame({"record_id": raw["record_id"].astype(str), "prediction_protocol": "test_final"})
    out["dino_top1"] = raw[top1_col].astype(str)
    out["dino_top2"] = raw["dino_top2"].astype(str) if "dino_top2" in raw.columns else ""
    out["dino_top3"] = raw["dino_top3"].astype(str) if "dino_top3" in raw.columns else ""
    out["dino_top1_score"] = pd.to_numeric(raw.get("confidence", raw.get("dino_top1_score", np.nan)), errors="coerce")
    out["dino_top2_score"] = pd.to_numeric(raw.get("dino_top2_score", np.nan), errors="coerce")
    out["dino_top3_score"] = pd.to_numeric(raw.get("dino_top3_score", np.nan), errors="coerce")
    out["dino_margin"] = out["dino_top1_score"] - out["dino_top2_score"]
    out["prediction_source_file"] = str(path)
    return out


def predict_test_reconstructed(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dev_features: np.ndarray,
    test_features: np.ndarray,
    c_value: float,
    class_weight: str | None,
    seed: int,
    feature_model: str,
) -> pd.DataFrame:
    clf = make_classifier(c_value, class_weight, seed)
    clf.fit(dev_features, dev_df["true_class"].astype(str).to_numpy())
    scores = aligned_predict_proba(clf, test_features, CLASSES)
    topk_rows = topk_from_scores(scores, CLASSES)
    rows = []
    for (_, rec), topk in zip(test_df.iterrows(), topk_rows):
        rows.append(
            {
                "record_id": rec["record_id"],
                "split": rec["split"],
                "prediction_protocol": "test_reconstructed_train_dev",
                "true_class": rec["true_class"],
                **topk,
                "feature_model": feature_model,
                "classifier_model": f"LogisticRegression(C={c_value}, class_weight={class_weight})",
                "prediction_source": "reconstructed_train_on_all_dev_predict_test",
                "resolved_image_path": rec["resolved_image_path"],
            }
        )
    return pd.DataFrame(rows).sort_values("record_id")


def compare_test(existing: pd.DataFrame, reconstructed: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    gt = test_df[["record_id", "true_class"]].copy()
    merged = (
        gt.merge(existing[["record_id", "dino_top1"]], on="record_id", how="left")
        .rename(columns={"dino_top1": "existing_top1"})
        .merge(reconstructed[["record_id", "dino_top1"]], on="record_id", how="left")
        .rename(columns={"dino_top1": "reconstructed_top1"})
    )
    has_both = merged["existing_top1"].fillna("") != ""
    has_both &= merged["reconstructed_top1"].fillna("") != ""
    sub = merged[has_both].copy()
    if len(sub) == 0:
        return pd.DataFrame(
            [
                {
                    "n": 0,
                    "agreement_rate": 0.0,
                    "existing_accuracy": 0.0,
                    "reconstructed_accuracy": 0.0,
                    "accuracy_diff_reconstructed_minus_existing": 0.0,
                }
            ]
        )
    existing_acc = (sub["existing_top1"] == sub["true_class"]).mean()
    reconstructed_acc = (sub["reconstructed_top1"] == sub["true_class"]).mean()
    return pd.DataFrame(
        [
            {
                "n": int(len(sub)),
                "agreement_rate": float((sub["existing_top1"] == sub["reconstructed_top1"]).mean()),
                "existing_accuracy": float(existing_acc),
                "reconstructed_accuracy": float(reconstructed_acc),
                "accuracy_diff_reconstructed_minus_existing": float(reconstructed_acc - existing_acc),
            }
        ]
    )


def write_readme(out: Path, args, dev_df: pd.DataFrame, test_df: pd.DataFrame | None, fold_df: pd.DataFrame):
    class_counts = dev_df["true_class"].value_counts().to_dict()
    lines = [
        "# Stage 10 DINOv2 Development OOF Predictions",
        "",
        "Historical `train` is treated as the development split for VLM/policy tuning.",
        "Historical `val`/`val_v2` is treated as the test split.",
        "",
        "Development predictions in this directory are out-of-fold: each crop is predicted by a classifier trained on the other folds only.",
        "Test predictions in the unified file come from the existing final DINOv2 prediction CSV when `--test-predictions-csv` is provided.",
        "",
        "No VLM inference is run here, and no test labels are used for threshold or policy selection.",
        "",
        "## Model",
        "",
        f"- feature model: `{args.feature_model_id}`",
        f"- classifier: `LogisticRegression(C={args.classifier_c}, class_weight={args.class_weight})`",
        "- status: reconstructed DINOv2 OOF baseline, aligned with the previous no-VLM protocol.",
        "",
        "## Development Split",
        "",
        f"- records: {len(dev_df)}",
        f"- class counts: `{class_counts}`",
        f"- folds: {len(fold_df)}",
        "",
    ]
    if test_df is not None:
        lines += ["## Test Split", "", f"- records with labels loaded for comparison: {len(test_df)}", ""]
    lines += [
        "## Outputs",
        "",
        "- `stage10_dinov2_dev_oof_predictions.csv`",
        "- `stage10_dev_oof_predictions.csv`",
        "- `stage10_dev_oof_fold_metrics.csv`",
        "- `stage10_dev_oof_confusion_matrix.csv`",
        "- `stage10_reconstructed_test_predictions.csv` when test labels are provided",
        "- `stage10_reconstructed_vs_existing_test_comparison.csv` when existing test predictions are provided",
        "- `stage10_dinov2_full_oof_plus_test_predictions.csv` when existing test predictions are provided",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-labels-jsonl", action="append", required=True)
    ap.add_argument("--test-labels-jsonl", action="append")
    ap.add_argument("--image-root", action="append", required=True)
    ap.add_argument("--test-predictions-csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature-model-id", default=DEFAULT_FEATURE_MODEL)
    ap.add_argument("--classifier-c", type=float, default=0.03)
    ap.add_argument("--class-weight", default="balanced")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--max-records", type=int, default=None, help="Smoke-test limit for development rows.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_roots = [Path(p) for p in args.image_root]
    class_weight = None if str(args.class_weight).lower() in {"none", "null", ""} else args.class_weight

    dev_df = build_frame([Path(p) for p in args.dev_labels_jsonl], image_roots)
    if args.max_records:
        dev_df = dev_df.groupby("true_class", group_keys=False).head(max(1, args.max_records // max(1, dev_df["true_class"].nunique())))
        dev_df = dev_df.head(args.max_records).reset_index(drop=True)
    dev_features = extract_features(
        dev_df,
        args.feature_model_id,
        args.batch_size,
        args.device,
        out / "stage10_dinov2_features_dev.npy",
        args.reuse_features,
    )
    oof_df, fold_df, cm = generate_oof(
        dev_df,
        dev_features,
        args.n_splits,
        args.seed,
        args.classifier_c,
        class_weight,
        args.feature_model_id,
    )
    oof_path = out / "stage10_dinov2_dev_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    oof_df.to_csv(out / "stage10_dev_oof_predictions.csv", index=False)
    fold_df.to_csv(out / "stage10_dev_oof_fold_metrics.csv", index=False)
    cm.to_csv(out / "stage10_dev_oof_confusion_matrix.csv")

    test_df = None
    reconstructed_test = None
    if args.test_labels_jsonl:
        test_df = build_frame([Path(p) for p in args.test_labels_jsonl], image_roots)
        test_features = extract_features(
            test_df,
            args.feature_model_id,
            args.batch_size,
            args.device,
            out / "stage10_dinov2_features_test.npy",
            args.reuse_features,
        )
        reconstructed_test = predict_test_reconstructed(
            dev_df,
            test_df,
            dev_features,
            test_features,
            args.classifier_c,
            class_weight,
            args.seed,
            args.feature_model_id,
        )
        reconstructed_test.to_csv(out / "stage10_reconstructed_test_predictions.csv", index=False)

    if args.test_predictions_csv:
        existing_test = normalize_existing_test_predictions(Path(args.test_predictions_csv))
        if test_df is not None:
            existing_test = existing_test.merge(test_df[["record_id", "split"]], on="record_id", how="left")
        else:
            existing_test["split"] = "test"
        existing_test.to_csv(out / "stage10_test_prediction_source_used.csv", index=False)
        if reconstructed_test is not None and test_df is not None:
            compare_test(existing_test, reconstructed_test, test_df).to_csv(
                out / "stage10_reconstructed_vs_existing_test_comparison.csv",
                index=False,
            )
        unified = pd.concat(
            [
                oof_df[
                    [
                        "record_id",
                        "split",
                        "prediction_protocol",
                        "dino_top1",
                        "dino_top2",
                        "dino_top3",
                        "dino_top1_score",
                        "dino_top2_score",
                        "dino_top3_score",
                        "dino_margin",
                        "prediction_source",
                    ]
                ].rename(columns={"prediction_source": "prediction_source_file"}),
                existing_test[
                    [
                        "record_id",
                        "split",
                        "prediction_protocol",
                        "dino_top1",
                        "dino_top2",
                        "dino_top3",
                        "dino_top1_score",
                        "dino_top2_score",
                        "dino_top3_score",
                        "dino_margin",
                        "prediction_source_file",
                    ]
                ],
            ],
            ignore_index=True,
        )
        unified.to_csv(out / "stage10_dinov2_full_oof_plus_test_predictions.csv", index=False)

    write_readme(out, args, dev_df, test_df, fold_df)
    print(
        {
            "dev_records": len(dev_df),
            "folds": len(fold_df),
            "dev_oof_predictions": len(oof_df),
            "test_records": 0 if test_df is None else len(test_df),
            "out_dir": str(out),
        }
    )


if __name__ == "__main__":
    main()
