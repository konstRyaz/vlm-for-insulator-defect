#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--schema-json", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    schema = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    req = schema.get("required", [])
    props = schema.get("properties", {})

    rows = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parse_ok = True
            schema_ok = True
            err = ""
            obj = {}
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("not_object")
            except Exception as exc:
                parse_ok = False
                schema_ok = False
                err = f"parse_error:{exc}"
            if parse_ok:
                for k in req:
                    if k not in obj:
                        schema_ok = False
                        err = f"missing_required:{k}"
                        break
                if schema_ok:
                    for k, spec in props.items():
                        if k in obj and "enum" in spec:
                            if obj[k] not in spec["enum"]:
                                schema_ok = False
                                err = f"invalid_enum:{k}"
                                break
            rows.append(
                {
                    "line_no": i,
                    "record_id": obj.get("record_id", ""),
                    "parse_ok": parse_ok,
                    "schema_ok": schema_ok,
                    "error": err,
                }
            )

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
