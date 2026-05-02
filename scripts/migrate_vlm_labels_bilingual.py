#!/usr/bin/env python3
from __future__ import annotations

"""
Migrate legacy vlm_labels_v1 JSONL records to bilingual-ready fields.

Adds optional fields:
- short_canonical_description_ru
- short_canonical_description_en
- report_snippet_ru
- report_snippet_en

Legacy EN compatibility fields are preserved.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate vlm_labels_v1 JSONL to bilingual-ready fields.")
    parser.add_argument("--input", required=True, type=str, help="Input JSONL path.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path. Default: <input_stem>.bilingual.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output path if it exists.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def ensure_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}.bilingual{input_path.suffix}")
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path} (use --overwrite)")

    rows = load_jsonl(input_path)
    migrated: list[dict[str, Any]] = []

    for rec in rows:
        row = dict(rec)
        legacy_desc = ensure_str(row.get("short_canonical_description"))
        legacy_snippet = ensure_str(row.get("report_snippet"))

        row["short_canonical_description_en"] = ensure_str(
            row.get("short_canonical_description_en") if "short_canonical_description_en" in row else legacy_desc
        )
        row["short_canonical_description_ru"] = ensure_str(row.get("short_canonical_description_ru"))
        row["report_snippet_en"] = ensure_str(row.get("report_snippet_en") if "report_snippet_en" in row else legacy_snippet)
        row["report_snippet_ru"] = ensure_str(row.get("report_snippet_ru"))

        # Keep legacy fields present and synced with EN for backward compatibility.
        row["short_canonical_description"] = row["short_canonical_description_en"]
        row["report_snippet"] = row["report_snippet_en"]
        migrated.append(row)

    write_jsonl_atomic(output_path, migrated)
    print(f"Migrated records: {len(migrated)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

