#!/usr/bin/env python3
"""Robust image path resolver for Kaggle/Linux runs.

This module fixes common failure modes:
- Windows backslashes
- URL-encoded separators such as %5C and %2F
- absolute paths from another machine
- paths relative to repo, Kaggle input root, or an image bundle root

It can be used as CLI or imported by Stage15 safe runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def decode_path_string(s: object) -> str:
    if s is None:
        return ""
    s = str(s).strip().strip('"').strip("'")
    # Decode repeatedly, because sometimes %255C appears.
    for _ in range(3):
        dec = urllib.parse.unquote(s)
        if dec == s:
            break
        s = dec
    s = s.replace("\\", "/")
    s = s.replace("%5C", "/").replace("%5c", "/")
    s = s.replace("%2F", "/").replace("%2f", "/")
    s = re.sub(r"/+", "/", s)
    # Remove Windows drive prefix, keeping rest.
    s = re.sub(r"^[A-Za-z]:/", "", s)
    return s


def candidate_suffixes(raw: str) -> list[str]:
    s = decode_path_string(raw)
    parts = [p for p in s.split("/") if p and p not in {".", ".."}]
    out = []
    if s:
        out.append(s)
    # Useful suffixes: last N components.
    for n in range(1, min(len(parts), 8) + 1):
        out.append("/".join(parts[-n:]))
    # If path includes known marker, keep from marker.
    markers = ["stage10/vlm_topk_inference_manifest", "vlm_topk_inference_manifest", "images", "crops"]
    for m in markers:
        if m in s:
            out.append(s[s.index(m):])
    # unique in order
    seen = set()
    res = []
    for x in out:
        x = x.strip("/")
        if x and x not in seen:
            seen.add(x)
            res.append(x)
    return res


def build_basename_index(roots: Iterable[Path], max_files: int = 50000) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    count = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                idx.setdefault(p.name, []).append(p)
                count += 1
                if count >= max_files:
                    return idx
    return idx


@dataclass
class ResolveResult:
    original: str
    normalized: str
    resolved: str
    exists: bool
    strategy: str
    error: str = ""


def resolve_one(raw: object, roots: Iterable[Path], basename_index: Optional[dict[str, list[Path]]] = None) -> ResolveResult:
    original = "" if raw is None else str(raw)
    norm = decode_path_string(original)

    if not norm:
        return ResolveResult(original, norm, "", False, "empty", "empty_path")

    p = Path(norm)
    if p.is_absolute() and p.exists():
        return ResolveResult(original, norm, str(p), True, "absolute_exists")

    roots = [Path(r) for r in roots]
    suffixes = candidate_suffixes(norm)

    # Try root / suffix.
    for root in roots:
        for suf in suffixes:
            cand = root / suf
            if cand.exists():
                return ResolveResult(original, norm, str(cand), True, f"root_suffix:{root}")

    # Try direct relative to cwd.
    for suf in suffixes:
        cand = Path(suf)
        if cand.exists():
            return ResolveResult(original, norm, str(cand.resolve()), True, "cwd_suffix")

    # Try basename search.
    name = Path(norm).name
    if basename_index is not None and name in basename_index:
        matches = basename_index[name]
        if len(matches) == 1:
            return ResolveResult(original, norm, str(matches[0]), True, "basename_unique")
        if len(matches) > 1:
            # Prefer matches whose suffix overlaps normalized path.
            norm_parts = set(Path(norm).parts)
            best = sorted(matches, key=lambda x: -len(norm_parts.intersection(set(x.parts))))[0]
            return ResolveResult(original, norm, str(best), True, f"basename_multi:{len(matches)}")

    return ResolveResult(original, norm, "", False, "not_found", f"not_found basename={name}")


def read_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def infer_path_col(rows: list[dict], requested: Optional[str] = None) -> str:
    if requested:
        return requested
    candidates = [
        "resolved_image_path", "image_path", "crop_path", "path",
        "main_image_path", "image_main", "image", "filename"
    ]
    cols = set(rows[0].keys()) if rows else set()
    for c in candidates:
        if c in cols:
            return c
    # heuristic
    for c in cols:
        lc = c.lower()
        if "image" in lc or "crop" in lc or "path" in lc:
            return c
    raise ValueError("Could not infer image path column")


def default_roots(extra: list[str]) -> list[Path]:
    roots = [Path.cwd(), Path("/kaggle/working"), Path("/kaggle/input")]
    roots += [Path(x) for x in extra if x]
    # common bundle roots
    roots += [
        Path("/kaggle/input/datasets"),
        Path("/kaggle/working/stage15"),
    ]
    # unique
    out = []
    seen = set()
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def preflight_manifest(
    manifest_path: Path,
    out_dir: Path,
    image_col: Optional[str] = None,
    roots: Optional[list[str]] = None,
    fail_on_missing: bool = True,
) -> dict:
    rows = read_table(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    roots_p = default_roots(roots or [])
    idx = build_basename_index(roots_p)
    col = infer_path_col(rows, image_col)

    out_rows = []
    fixed_rows = []
    for i, r in enumerate(rows):
        rr = resolve_one(r.get(col), roots_p, idx)
        out_rows.append({
            "row_idx": i,
            "record_id": r.get("record_id", ""),
            "path_col": col,
            "original_image_path": rr.original,
            "normalized_image_path": rr.normalized,
            "resolved_image_path": rr.resolved,
            "exists": rr.exists,
            "resolution_strategy": rr.strategy,
            "error": rr.error,
        })
        fr = dict(r)
        fr["resolved_image_path"] = rr.resolved
        fr["image_exists"] = str(bool(rr.exists))
        fixed_rows.append(fr)

    n = len(out_rows)
    missing = sum(1 for r in out_rows if not r["exists"])
    summary = {
        "manifest": str(manifest_path),
        "path_col": col,
        "n": n,
        "missing": missing,
        "exists": n - missing,
        "exists_rate": (n - missing) / n if n else 0.0,
        "roots": [str(r) for r in roots_p],
        "status": "OK" if missing == 0 else "MISSING_PATHS",
    }

    write_csv(out_dir / "path_preflight.csv", out_rows)
    write_csv(out_dir / "manifest_resolved.csv", fixed_rows)
    (out_dir / "path_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_dir / "first_missing_paths.csv", [r for r in out_rows if not r["exists"]][:50])

    if fail_on_missing and missing > 0:
        raise SystemExit(f"Path preflight failed: {missing}/{n} missing. See {out_dir/'path_preflight.csv'}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-col", default=None)
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args()

    summary = preflight_manifest(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        image_col=args.image_col,
        roots=args.root,
        fail_on_missing=not args.no_fail,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
