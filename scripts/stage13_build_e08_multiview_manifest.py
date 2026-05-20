#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def build_zoom(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
        crop = im.crop((x0, y0, x1, y1)).resize((w, h), Image.Resampling.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", default="outputs/stage10/vlm_topk_inference_manifest/stage10_vlm_eval_reference.csv")
    ap.add_argument("--out-dir", default="outputs/stage13_tradeoff_benefit_expansion/E08_multiview_evidence")
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(args.reference_csv)
    test = ref[ref["split"] == "val"].copy().head(args.limit)
    zdir = out / "zoom_views"

    rows = []
    for _, r in test.iterrows():
        p = Path(str(r["resolved_image_path"]))
        zp = zdir / f"{r['record_id']}_zoom.jpg"
        if p.exists():
            build_zoom(p, zp)
        rows.append(
            {
                "record_id": r["record_id"],
                "split": r["split"],
                "image_main": str(p),
                "image_zoom": str(zp),
                "label_coarse_class": r["label_coarse_class"],
                "dino_top1": r["dino_top1"],
                "dino_top2": r["dino_top2"],
                "dino_top3": r["dino_top3"],
            }
        )
    pd.DataFrame(rows).to_csv(out / "multiview_manifest.csv", index=False)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
