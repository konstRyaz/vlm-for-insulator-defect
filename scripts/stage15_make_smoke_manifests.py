#!/usr/bin/env python3
"""Build small E02/E05/E06 smoke manifests with resolved image paths.

Input is the Stage10 eval reference / manifest-like CSV.
This script is intentionally conservative: it creates tiny manifests and runs path preflight.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stage15_safe_path_resolver import preflight_manifest, read_table, write_csv, resolve_one, default_roots, build_basename_index


def find_col(rows: list[dict], candidates: list[str]) -> str | None:
    cols = set(rows[0].keys()) if rows else set()
    for c in candidates:
        if c in cols:
            return c
    return None


def label(row: dict) -> str:
    return row.get("label_coarse_class") or row.get("coarse_class") or row.get("category_name") or row.get("label") or ""


def dino_top1(row: dict) -> str:
    return row.get("dino_top1") or row.get("top1_class") or row.get("pred_coarse_class") or ""


def dino_top2(row: dict) -> str:
    return row.get("dino_top2") or row.get("top2_class") or row.get("second_best_class") or ""


def choose_rows(rows: list[dict], predicate, limit: int) -> list[dict]:
    out = [r for r in rows if predicate(r)]
    if len(out) < limit:
        # fill with deterministic extras
        seen = {id(r) for r in out}
        for r in rows:
            if id(r) not in seen:
                out.append(r)
            if len(out) >= limit:
                break
    return out[:limit]


def add_resolved(rows: list[dict], image_root: str) -> list[dict]:
    roots = default_roots([image_root])
    idx = build_basename_index(roots)
    # infer path column
    cols = rows[0].keys()
    path_col = None
    for c in ["resolved_image_path", "image_path", "crop_path", "path", "filename"]:
        if c in cols:
            path_col = c
            break
    if path_col is None:
        for c in cols:
            if "path" in c.lower() or "image" in c.lower() or "crop" in c.lower():
                path_col = c
                break
    if path_col is None:
        raise ValueError("No image path column found")

    fixed = []
    for r in rows:
        rr = resolve_one(r.get(path_col), roots, idx)
        x = dict(r)
        x["resolved_image_path"] = rr.resolved
        x["image_exists"] = str(rr.exists)
        fixed.append(x)
    return fixed


def make_e02(rows: list[dict], limit: int) -> list[dict]:
    # Prefer flashover predictions.
    chosen = choose_rows(rows, lambda r: dino_top1(r) == "defect_flashover", limit)
    out = []
    for r in chosen:
        x = {
            "record_id": r.get("record_id", ""),
            "resolved_image_path": r.get("resolved_image_path", ""),
            "dino_top1": dino_top1(r),
            "dino_top2": dino_top2(r),
            "dino_top3": r.get("dino_top3") or r.get("top3_class") or "",
            "dino_margin": r.get("dino_margin") or r.get("margin") or "",
            "label_coarse_class": label(r),  # kept for later eval; runner does not use in prompt
        }
        out.append(x)
    return out


def make_e05(rows: list[dict], limit: int) -> list[dict]:
    chosen = choose_rows(rows, lambda r: True, limit)
    out = []
    claim_templates = [
        ("flashover_surface_evidence", "There is visible burn, arc, or dark surface trace on the insulator surface."),
        ("broken_structure_evidence", "There is visible missing material, a broken shed or disc, crack, or edge discontinuity."),
        ("ok_intact_evidence", "The visible insulator appears structurally intact and no direct defect evidence is visible."),
    ]
    for i, r in enumerate(chosen):
        cid, ctext = claim_templates[i % len(claim_templates)]
        out.append({
            "record_id": r.get("record_id", ""),
            "resolved_image_path": r.get("resolved_image_path", ""),
            "claim_id": cid,
            "claim_text": ctext,
            "dino_top1": dino_top1(r),
            "dino_top2": dino_top2(r),
            "label_coarse_class": label(r),
        })
    return out


def make_e06(rows: list[dict], limit: int) -> list[dict]:
    chosen = choose_rows(rows, lambda r: True, limit)
    out = []
    for r in chosen:
        # Use same image as main/zoom/context placeholder, runner handles duplicates.
        p = r.get("resolved_image_path", "")
        out.append({
            "record_id": r.get("record_id", ""),
            "resolved_image_path": p,
            "main_image_path": p,
            "zoom_image_path": p,
            "context_image_path": p,
            "dino_top1": dino_top1(r),
            "dino_top2": dino_top2(r),
            "dino_top3": r.get("dino_top3") or r.get("top3_class") or "",
            "label_coarse_class": label(r),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = read_table(Path(args.reference_csv))
    rows = add_resolved(rows, args.image_root)

    specs = {
        "E02_flashover": make_e02(rows, args.limit),
        "E05_claim": make_e05(rows, args.limit),
        "E06_multiview": make_e06(rows, args.limit),
    }

    summary = {}
    for name, manifest_rows in specs.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        manifest = d / "manifest.csv"
        write_csv(manifest, manifest_rows)
        try:
            pf = preflight_manifest(manifest, d, image_col="resolved_image_path", roots=[args.image_root], fail_on_missing=True)
            summary[name] = pf
        except SystemExit as e:
            summary[name] = {"status": "FAILED", "error": str(e)}
            raise

    (out_root / "manifest_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
