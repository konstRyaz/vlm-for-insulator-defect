import argparse
import shutil
from pathlib import Path

import pandas as pd


LEAKAGE_FORBIDDEN = {
    "label_coarse_class",
    "label_visual_evidence_tags",
    "label_visibility",
    "label_needs_review",
    "label_short_canonical_description",
    "label_report_snippet",
    "category_name",
    "annotator_notes",
    "label_version",
    "crop_path",
    "bbox_xywh",
}

MANIFEST_COLUMNS = [
    "record_id",
    "split",
    "source",
    "resolved_image_path",
    "dino_top1",
    "dino_top2",
    "dino_top3",
    "dino_top1_score",
    "dino_top2_score",
    "dino_top3_score",
    "dino_margin",
]

REFERENCE_COLUMNS = [
    "record_id",
    "split",
    "source",
    "resolved_image_path",
    "label_coarse_class",
    "label_visual_evidence_tags",
    "label_visibility",
    "label_needs_review",
    "label_short_canonical_description",
    "label_report_snippet",
    "category_name",
    "annotator_notes",
    "label_version",
    "dino_top1",
    "dino_top2",
    "dino_top3",
    "dino_top1_score",
    "dino_top2_score",
    "dino_top3_score",
    "dino_margin",
    "dino_top1_correct",
    "gt_in_top2",
    "gt_in_top3",
]


def resolve_image_path(row: pd.Series, roots: list[Path]) -> str:
    raw = str(row.get("crop_path", "") or row.get("image_path", "") or "")
    if not raw or raw.lower() == "nan":
        return ""
    path = Path(raw)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    for root in roots:
        candidates.extend(
            [
                root / path,
                root / path.name,
                root / "crops" / path,
            ]
        )
    record_id = str(row.get("record_id", "") or "")
    if record_id:
        for root in roots:
            try:
                candidates.extend(root.rglob(f"{record_id}.*"))
            except FileNotFoundError:
                pass
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return ""


def write_jsonl(df: pd.DataFrame, path: Path):
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def export_neutral_images(df: pd.DataFrame, out: Path) -> pd.Series:
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    neutral_paths = []
    for _, row in df.iterrows():
        src = str(row.get("_source_resolved_image_path", "") or "")
        if not src:
            neutral_paths.append("")
            continue
        src_path = Path(src)
        suffix = src_path.suffix or ".jpg"
        dst = image_dir / f"{row['record_id']}{suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src_path, dst)
        neutral_paths.append(str(dst))
    return pd.Series(neutral_paths, index=df.index)


def write_readme(out: Path, args, manifest: pd.DataFrame, missing: pd.DataFrame, split_counts: pd.DataFrame, pair_counts: pd.DataFrame):
    def simple_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "(none)"
        view = df.copy()
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = ["| " + " | ".join(map(str, row)) + " |" for row in view.values.tolist()]
        return "\n".join([header, sep] + body)

    lines = [
        "# Stage 10 VLM Top-k Inference Manifest",
        "",
        "This directory prepares a leakage-safe manifest for a future VLM top-k checker/reranker run.",
        "",
        "The manifest contains only crop image paths and DINOv2 candidate predictions/scores.",
        "It deliberately excludes ground-truth labels, evidence tags, report snippets, and class-bearing crop paths.",
        "Images are exported under class-neutral filenames in `images/<record_id>.<ext>` before writing the manifest.",
        "",
        "Historical `train` is the development split for selecting VLM policy. Historical `val` is the test split for final reporting.",
        "",
        "## Inputs",
        "",
        f"- table: `{args.table_csv}`",
        *[f"- image root: `{p}`" for p in args.images_root],
        "",
        "## Resolution Summary",
        "",
        f"- manifest rows: {len(manifest)}",
        f"- resolved image paths: {int((manifest['resolved_image_path'].fillna('') != '').sum())}",
        f"- missing image paths: {len(missing)}",
        "",
        "## Split Counts",
        "",
        simple_markdown(split_counts),
        "",
        "## Top1/Top2 Pair Counts",
        "",
        simple_markdown(pair_counts.head(30)),
        "",
        "## Outputs",
        "",
        "- `stage10_vlm_manifest.csv`",
        "- `stage10_vlm_manifest.jsonl`",
        "- `stage10_vlm_eval_reference.csv`",
        "- `stage10_vlm_manifest_missing_images.csv`",
        "- `stage10_vlm_manifest_split_counts.csv`",
        "- `stage10_vlm_manifest_top_pair_counts.csv`",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", required=True)
    ap.add_argument("--images-root", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in args.images_root]

    df = pd.read_csv(args.table_csv)
    df = df[(df["record_id"].fillna("") != "") & (df["dino_top1"].fillna("") != "")].copy()
    df["_source_resolved_image_path"] = df.apply(lambda row: resolve_image_path(row, roots), axis=1)
    df["resolved_image_path"] = export_neutral_images(df, out)

    manifest = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = ""
    manifest = manifest[MANIFEST_COLUMNS].copy()

    leaked = sorted(set(manifest.columns) & LEAKAGE_FORBIDDEN)
    if leaked:
        raise ValueError(f"Leakage columns entered manifest: {leaked}")
    if manifest["resolved_image_path"].astype(str).str.contains("defect_|insulator_ok", case=False, regex=True).any():
        raise ValueError("resolved_image_path contains class names after neutral export.")

    reference = df.copy()
    for col in REFERENCE_COLUMNS:
        if col not in reference.columns:
            reference[col] = ""
    reference = reference[REFERENCE_COLUMNS].copy()

    missing = manifest[manifest["resolved_image_path"].fillna("").astype(str) == ""].copy()
    split_counts = manifest["split"].fillna("").astype(str).value_counts().rename_axis("split").reset_index(name="count")
    pair_counts = (
        manifest.groupby(["dino_top1", "dino_top2"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    manifest.to_csv(out / "stage10_vlm_manifest.csv", index=False)
    write_jsonl(manifest, out / "stage10_vlm_manifest.jsonl")
    reference.to_csv(out / "stage10_vlm_eval_reference.csv", index=False)
    missing.to_csv(out / "stage10_vlm_manifest_missing_images.csv", index=False)
    split_counts.to_csv(out / "stage10_vlm_manifest_split_counts.csv", index=False)
    pair_counts.to_csv(out / "stage10_vlm_manifest_top_pair_counts.csv", index=False)
    write_readme(out, args, manifest, missing, split_counts, pair_counts)
    print(
        {
            "rows": len(manifest),
            "resolved_image_paths": int((manifest["resolved_image_path"].fillna("") != "").sum()),
            "missing_image_paths": len(missing),
            "out_dir": str(out),
        }
    )


if __name__ == "__main__":
    main()
