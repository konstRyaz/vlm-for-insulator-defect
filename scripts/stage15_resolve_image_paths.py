#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _choose_id_col(df: pd.DataFrame) -> str:
    for c in ("record_id", "id", "sample_id"):
        if c in df.columns:
            return c
    return ""


def _normalize_raw_path(value: str) -> str:
    s = str(value or "").strip()
    s = urllib.parse.unquote(s)
    s = s.replace("%5C", "/").replace("%2F", "/")
    s = s.replace("\\", "/")
    # Drop Windows drive prefix (C:/...) to avoid leaking host-specific roots into Linux runtime.
    if len(s) >= 2 and s[1] == ":":
        s = s[2:]
    s = s.lstrip("/")
    while "//" in s:
        s = s.replace("//", "/")
    return s


def _candidate_roots(extra_roots: Iterable[str]) -> list[Path]:
    roots = [
        Path.cwd(),
        Path("/kaggle/working"),
        Path("/kaggle/working/repo"),
        Path("/kaggle/input"),
        Path("outputs/stage10/vlm_topk_inference_manifest/images"),
    ]
    for r in extra_roots:
        if r:
            roots.append(Path(r))
    # unique preserve order
    uniq: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        k = str(r)
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def _build_name_index(roots: Iterable[Path]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                idx.setdefault(p.name.lower(), str(p))
    return idx


def _resolve_one(
    original: str,
    normalized: str,
    roots: list[Path],
    name_index: dict[str, str],
) -> tuple[str, bool, str, str]:
    # 1) as-is absolute/relative normalized
    p = Path(normalized)
    if p.exists():
        return str(p), True, "normalized_exists", ""

    # 2) attach to known roots
    for root in roots:
        cand = root / normalized
        if cand.exists():
            return str(cand), True, f"join_root:{root}", ""

    # 3) basename lookup
    name = Path(normalized).name.lower()
    if name in name_index:
        return name_index[name], True, "basename_index", ""

    return "", False, "missing", f"not_found:{original}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV or JSONL manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--path-cols", default="resolved_image_path,image_main,image_zoom,view_main,view_context,view_zoom")
    ap.add_argument("--extra-root", action="append", default=[])
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_manifest(manifest_path)
    path_cols = [c.strip() for c in args.path_cols.split(",") if c.strip() and c.strip() in df.columns]
    if not path_cols:
        raise SystemExit("No path columns found in manifest for preflight.")

    rid_col = _choose_id_col(df)
    roots = _candidate_roots(args.extra_root)
    name_idx = _build_name_index(roots)

    rows: list[dict[str, object]] = []
    resolved_df = df.copy()
    for i, r in df.iterrows():
        rid = str(r.get(rid_col, i)) if rid_col else str(i)
        for col in path_cols:
            original = str(r.get(col, ""))
            normalized = _normalize_raw_path(original)
            resolved, exists, strategy, err = _resolve_one(original, normalized, roots, name_idx)
            rows.append(
                {
                    "record_id": rid,
                    "path_col": col,
                    "original_image_path": original,
                    "normalized_image_path": normalized,
                    "resolved_image_path": resolved,
                    "exists": bool(exists),
                    "resolution_strategy": strategy,
                    "error": err,
                }
            )
            if exists and resolved:
                resolved_df.at[i, col] = resolved

    pre = pd.DataFrame(rows)
    pre.to_csv(out_dir / "path_preflight.csv", index=False)

    missing = pre[~pre["exists"]].copy()
    missing.head(200).to_csv(out_dir / "first_missing_paths.csv", index=False)

    summary = {
        "manifest": str(manifest_path),
        "n_records": int(df.shape[0]),
        "n_path_checks": int(pre.shape[0]),
        "n_missing": int((~pre["exists"]).sum()),
        "exists_rate": float(pre["exists"].mean()) if len(pre) else 0.0,
        "path_cols_checked": path_cols,
        "roots": [str(r) for r in roots],
    }
    (out_dir / "path_preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    resolved_out = out_dir / f"{manifest_path.stem}.resolved{manifest_path.suffix}"
    if manifest_path.suffix.lower() == ".jsonl":
        resolved_df.to_json(resolved_out, orient="records", lines=True, force_ascii=False)
    else:
        resolved_df.to_csv(resolved_out, index=False)
    summary["resolved_manifest"] = str(resolved_out)
    (out_dir / "path_preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary["n_missing"] > 0:
        print(json.dumps(summary, ensure_ascii=False))
        sys.exit(2)


if __name__ == "__main__":
    main()
