#!/usr/bin/env python3
"""Validate safe runner JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.jsonl))
    n = len(rows)
    summary = {
        "n": n,
        "image_load_success": sum(bool(r.get("image_load_success")) for r in rows),
        "parse_ok": sum(bool(r.get("parse_ok")) for r in rows),
        "schema_ok": sum(bool(r.get("schema_ok")) for r in rows),
        "runtime_errors": sum(1 for r in rows if r.get("runtime_error")),
    }
    for k in ["image_load_success", "parse_ok", "schema_ok", "runtime_errors"]:
        summary[k + "_rate"] = summary[k] / n if n else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "validation_summary.csv", [summary])

    failures = []
    for r in rows:
        if not (r.get("image_load_success") and r.get("parse_ok") and r.get("schema_ok")):
            failures.append({
                "record_id": r.get("record_id"),
                "runtime_error": r.get("runtime_error", ""),
                "parse_error": r.get("parse_error", ""),
                "schema_error": r.get("schema_error", ""),
                "raw_response_head": str(r.get("raw_response", ""))[:500],
            })
    write_csv(out_dir / "failures.csv", failures)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
