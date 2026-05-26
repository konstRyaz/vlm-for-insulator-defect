#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_path_str(s: str) -> str:
    return str(s).replace("%5C", "\\").replace("%2F", "/").replace("\\", "/")


def build_image_index(search_roots: Iterable[Path]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                idx.setdefault(p.name.lower(), str(p))
    return idx


def resolve_path(path_value: str, image_index: dict[str, str]) -> str:
    p = Path(normalize_path_str(path_value))
    if p.exists():
        return str(p)
    key = p.name.lower()
    if key in image_index:
        return image_index[key]
    return str(p)


def apply_resolve(df: pd.DataFrame, col: str, image_index: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).map(lambda s: resolve_path(s, image_index))
    return out

