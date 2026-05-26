#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage10-table", default="outputs/stage10/full_dataset_all_splits_dinov2_oof_plus_test/stage10_full_dataset_table.csv")
    ap.add_argument("--images-root", default="outputs/stage10/vlm_topk_inference_manifest/images")
    ap.add_argument("--out-dir", default="outputs/stage12/structured_output_v2_binary_real_pilot")
    ap.add_argument("--pilot-n", type=int, default=72)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.stage10_table)
    dev = df[df["split"] == "train"].copy()
    dev["is_error"] = (~dev["dino_top1_correct"].astype(bool)).astype(int)
    dev["is_ok"] = (dev["label_coarse_class"] == "insulator_ok").astype(int)
    dev["is_flashover_risk"] = (
        (dev["dino_top1"] == "defect_flashover") | (dev["dino_top2"] == "defect_flashover")
    ).astype(int)
    dev["is_broken_risk"] = (
        (dev["dino_top1"] == "defect_broken") | (dev["dino_top2"] == "defect_broken")
    ).astype(int)

    # Build pilot: all/most errors + balanced controls
    err = dev[dev["is_error"] == 1].copy()
    ok = dev[(dev["is_ok"] == 1) & (dev["is_error"] == 0)].copy()
    rest = dev[(dev["is_ok"] == 0) & (dev["is_error"] == 0)].copy()

    n_err = min(len(err), int(args.pilot_n * 0.55))
    n_ok = min(len(ok), int(args.pilot_n * 0.30))
    n_rest = max(0, args.pilot_n - n_err - n_ok)

    pilot = pd.concat([err.head(n_err), ok.head(n_ok), rest.head(n_rest)], ignore_index=True)
    pilot = pilot.drop_duplicates("record_id").head(args.pilot_n).copy()

    images_root = Path(args.images_root)
    pilot["resolved_image_path"] = pilot["record_id"].astype(str).map(lambda rid: str((images_root / f"{rid}.jpg").as_posix()))
    pilot["image_exists"] = pilot["resolved_image_path"].map(lambda p: Path(p).exists())

    cols = [
        "record_id",
        "split",
        "resolved_image_path",
        "label_coarse_class",
        "label_visual_evidence_tags",
        "label_visibility",
        "label_needs_review",
        "dino_top1",
        "dino_top2",
        "dino_top3",
        "dino_top1_score",
        "dino_top2_score",
        "dino_top3_score",
        "dino_margin",
        "is_error",
        "is_flashover_risk",
        "is_broken_risk",
        "image_exists",
    ]
    pilot[cols].to_csv(out / "pilot_manifest.csv", index=False)
    print(f"pilot rows: {len(pilot)}")
    print(f"missing images: {(~pilot['image_exists']).sum()}")


if __name__ == "__main__":
    main()
