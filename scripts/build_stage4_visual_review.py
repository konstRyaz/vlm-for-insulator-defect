#!/usr/bin/env python3
"""Build an HTML visual review for paired Stage 4 helped/hurt cases."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 4 visual review HTML.")
    parser.add_argument("--paired-cases", required=True, type=Path)
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out-html", required=True, type=Path)
    parser.add_argument("--max-per-category", type=int, default=24)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rid = row.get("record_id")
                if isinstance(rid, str):
                    rows[rid] = row
    return rows


def find_image(images_dir: Path, image_path: object) -> Path | None:
    if not isinstance(image_path, str) or not image_path:
        return None
    p = Path(image_path)
    candidates = [images_dir / p.name, images_dir / image_path, p]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def crop_to_data_uri(image_path: Path, bbox_xywh: object, padding_ratio: float) -> str:
    if not isinstance(bbox_xywh, list) or len(bbox_xywh) != 4:
        return ""
    img = Image.open(image_path).convert("RGB")
    x, y, w, h = [float(v) for v in bbox_xywh]
    pad_x = w * padding_ratio
    pad_y = h * padding_ratio
    left = max(0, int(round(x - pad_x)))
    top = max(0, int(round(y - pad_y)))
    right = min(img.width, int(round(x + w + pad_x)))
    bottom = min(img.height, int(round(y + h + pad_y)))
    crop = img.crop((left, top, right, bottom))
    crop.thumbnail((360, 260))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def card(row: dict[str, str], pred: dict[str, object] | None, images_dir: Path, padding_ratio: float) -> str:
    rid = row["record_id"]
    matched = row.get("candidate_matched_pred") or row.get("matched_pred_record_id") or ""
    pred = pred or {}
    image_path = find_image(images_dir, pred.get("image_path"))
    uri = crop_to_data_uri(image_path, pred.get("bbox_xywh"), padding_ratio) if image_path else ""
    img_html = f'<img src="{uri}" alt="{html.escape(rid)}">' if uri else "<div class='missing'>image unavailable</div>"
    return f"""
    <article class="card {html.escape(row.get('category', ''))}">
      <div class="thumb">{img_html}</div>
      <div class="meta">
        <h3>{html.escape(rid)}</h3>
        <p><b>GT</b>: {html.escape(row.get('gt_coarse_class', ''))}</p>
        <p><b>Baseline</b>: {html.escape(row.get('baseline_pred', ''))} ({row.get('baseline_correct')})</p>
        <p><b>Champion</b>: {html.escape(row.get('candidate_pred', ''))} ({row.get('candidate_correct')})</p>
        <p><b>Matched pred</b>: {html.escape(matched)}</p>
        <p><b>IoU</b>: {html.escape(row.get('candidate_match_iou', ''))}</p>
        <p><b>Bucket</b>: {html.escape(row.get('candidate_error_bucket', ''))}</p>
      </div>
    </article>
    """


def main() -> None:
    args = parse_args()
    rows = read_csv(args.paired_cases)
    preds = read_jsonl_by_id(args.candidate_predictions)
    helped = [r for r in rows if r.get("category") == "helped"][: args.max_per_category]
    hurt = [r for r in rows if r.get("category") == "hurt"][: args.max_per_category]

    def pred_for(row: dict[str, str]) -> dict[str, object] | None:
        matched = row.get("candidate_matched_pred") or row.get("matched_pred_record_id") or ""
        if matched in preds:
            return preds[matched]
        # The paired script stores only the case row by default, so use candidate_pred
        # metadata only if the prediction id is unavailable.
        return None

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Stage 4 Visual Review</title>",
        "<style>",
        "body{font-family:Georgia,serif;background:#f6f1e8;color:#241d16;margin:32px}",
        "h1{font-size:34px;margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}",
        ".card{display:flex;gap:14px;background:#fffaf0;border:1px solid #d6c6aa;border-radius:16px;padding:14px;box-shadow:0 8px 20px #0001}",
        ".card.helped{border-left:8px solid #2c7a4b}.card.hurt{border-left:8px solid #b33a32}",
        ".thumb{width:220px;min-height:140px;display:flex;align-items:center;justify-content:center;background:#eadfce;border-radius:10px;overflow:hidden}",
        "img{max-width:220px;max-height:180px}.meta h3{margin:0 0 8px;font-size:17px}.meta p{margin:4px 0}.missing{color:#7c6d5a}",
        "</style></head><body>",
        "<h1>Stage 4 Visual Review</h1>",
        "<p>Helped and hurt GT objects for the DINOv2+Qwen champion against the Qwen Stage 4 baseline.</p>",
        f"<h2>Helped cases ({len(helped)})</h2><section class='grid'>",
    ]
    html_parts.extend(card(r, pred_for(r), args.images_dir, args.padding_ratio) for r in helped)
    html_parts.append("</section>")
    html_parts.append(f"<h2>Hurt cases ({len(hurt)})</h2><section class='grid'>")
    html_parts.extend(card(r, pred_for(r), args.images_dir, args.padding_ratio) for r in hurt)
    html_parts.append("</section></body></html>")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(args.out_html)


if __name__ == "__main__":
    main()
