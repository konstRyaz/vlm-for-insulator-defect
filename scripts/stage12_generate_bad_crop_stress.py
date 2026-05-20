#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


CORRUPTIONS = [
    "background_gray",
    "crop_shift",
    "gaussian_blur",
    "low_contrast",
    "overzoom",
    "partial_crop",
]


def apply_corruption(img: Image.Image, kind: str) -> Image.Image:
    w, h = img.size
    if kind == "background_gray":
        return Image.new("RGB", img.size, (127, 127, 127))
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=4.0))
    if kind == "low_contrast":
        return ImageEnhance.Contrast(img).enhance(0.45)
    if kind == "overzoom":
        crop = img.crop((int(0.2 * w), int(0.2 * h), int(0.8 * w), int(0.8 * h)))
        return crop.resize((w, h), Image.Resampling.BICUBIC)
    if kind == "crop_shift":
        x0 = int(0.15 * w)
        y0 = int(0.15 * h)
        crop = img.crop((x0, y0, w, h))
        out = Image.new("RGB", (w, h), (0, 0, 0))
        out.paste(crop, (0, 0))
        return out
    if kind == "partial_crop":
        out = img.copy()
        block = Image.new("RGB", (int(0.45 * w), int(0.45 * h)), (0, 0, 0))
        out.paste(block, (int(0.5 * w), int(0.5 * h)))
        return out
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    ap.add_argument("--max-base", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import json
    import random

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for line in Path(args.manifest_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if args.split != "all":
        rows = [r for r in rows if str(r.get("split", "")) == args.split]
    rng.shuffle(rows)
    base = rows[: args.max_base]

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for r in base:
        src = Path(str(r["resolved_image_path"]).replace("\\", "/"))
        if not src.exists():
            continue
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            for kind in CORRUPTIONS:
                cimg = apply_corruption(rgb, kind)
                out_name = f"{r['record_id']}__{kind}.jpg"
                dst = img_dir / out_name
                cimg.save(dst, quality=90)
                out_rows.append(
                    {
                        "record_id": str(r["record_id"]),
                        "split": str(r.get("split", "")),
                        "corruption": kind,
                        "bad_crop_id": f"{r['record_id']}__{kind}",
                        "resolved_image_path": str(dst).replace("\\", "/"),
                        "dino_top1": r.get("dino_top1", ""),
                        "dino_top2": r.get("dino_top2", ""),
                        "dino_top3": r.get("dino_top3", ""),
                        "dino_top1_score": r.get("dino_top1_score", None),
                        "dino_top2_score": r.get("dino_top2_score", None),
                        "dino_top3_score": r.get("dino_top3_score", None),
                        "dino_margin": r.get("dino_margin", None),
                    }
                )

    df = pd.DataFrame(out_rows)
    df.to_csv(out_dir / "stage12_bad_crop_manifest.csv", index=False)
    print(f"Wrote: {out_dir / 'stage12_bad_crop_manifest.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
