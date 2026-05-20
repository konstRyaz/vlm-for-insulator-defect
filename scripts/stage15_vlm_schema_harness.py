#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def normalize_path(p: str) -> Path:
    s = str(p).replace("%5C", "\\").replace("%2F", "/").replace("\\", "/")
    return Path(s)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                rows.append({"_parse_error": True, "_raw": line})
    return rows


def enum_ok(obj: dict[str, Any], key: str, allowed: set[str]) -> bool:
    return key in obj and str(obj[key]) in allowed


def validate_rows(schema_name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for obj in rows:
        parse_ok = "_parse_error" not in obj
        schema_ok = False
        invalid_enum = 0
        if parse_ok:
            if schema_name == "flashover_overclaim":
                schema_ok = all(
                    [
                        "record_id" in obj,
                        enum_ok(obj, "claim_supported", {"yes", "no", "uncertain"}),
                        isinstance(obj.get("recommend_review", True), bool),
                    ]
                )
                invalid_enum = 0 if enum_ok(obj, "claim_supported", {"yes", "no", "uncertain"}) else 1
            elif schema_name == "claim_verification":
                schema_ok = all(
                    [
                        "record_id" in obj,
                        "claim_id" in obj,
                        enum_ok(obj, "verification", {"supported", "contradicted", "not_enough_evidence"}),
                    ]
                )
                invalid_enum = 0 if enum_ok(obj, "verification", {"supported", "contradicted", "not_enough_evidence"}) else 1
            elif schema_name == "multiview":
                schema_ok = "record_id" in obj and ("short_reason" in obj or "selected_rank" in obj)
        recs.append(
            {
                "schema_name": schema_name,
                "record_id": obj.get("record_id", ""),
                "parse_ok": bool(parse_ok),
                "schema_ok": bool(schema_ok),
                "invalid_enum": int(invalid_enum),
                "truncation_detected": 0,
                "retry_used": 0,
                "retry_success": 0,
            }
        )
    return pd.DataFrame(recs)


def image_preflight(manifest_jsonl: Path, limit: int) -> pd.DataFrame:
    m = pd.read_json(manifest_jsonl, lines=True)
    if limit > 0:
        m = m.head(limit).copy()
    rows = []
    for _, r in m.iterrows():
        rid = str(r["record_id"])
        p = normalize_path(str(r["resolved_image_path"]))
        exists = p.exists()
        can_open = False
        err = ""
        if exists:
            try:
                with Image.open(p) as im:
                    im.verify()
                can_open = True
            except Exception as exc:
                err = str(exc)
        rows.append({"record_id": rid, "resolved_image_path": str(p), "exists": exists, "can_open": can_open, "error": err})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="stage10_vlm_manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Actual image preflight (real images, not placeholders)
    pre = image_preflight(Path(args.manifest), args.limit)
    pre.to_csv(out / "image_path_preflight.csv", index=False)

    # Validate real JSONL outputs from prior real runs
    sources = [
        ("flashover_overclaim", Path("outputs/stage13_tradeoff_benefit_expansion/E03_flashover_overclaim_checker/flashover_claim_outputs.jsonl")),
        ("claim_verification", Path("outputs/stage13_tradeoff_benefit_expansion/E05_claim_verification/claim_verification_outputs.jsonl")),
        ("multiview", Path("outputs/stage13_tradeoff_benefit_expansion/E08_multiview_evidence/multiview_vlm_outputs.jsonl")),
    ]
    all_frames = []
    for name, p in sources:
        rows = load_jsonl(p)
        if rows:
            all_frames.append(validate_rows(name, rows))
        else:
            all_frames.append(
                pd.DataFrame(
                    [
                        {
                            "schema_name": name,
                            "record_id": "",
                            "parse_ok": False,
                            "schema_ok": False,
                            "invalid_enum": 0,
                            "truncation_detected": 0,
                            "retry_used": 0,
                            "retry_success": 0,
                        }
                    ]
                )
            )
    res = pd.concat(all_frames, ignore_index=True)
    res.to_csv(out / "schema_validation_summary.csv", index=False)
    res[(~res["parse_ok"]) | (~res["schema_ok"])].to_json(
        out / "schema_validation_failures.jsonl", orient="records", lines=True, force_ascii=False
    )

    parse_ok = float(res["parse_ok"].mean()) if len(res) else 0.0
    schema_ok = float(res["schema_ok"].mean()) if len(res) else 0.0
    path_ok = float((pre["exists"] & pre["can_open"]).mean()) if len(pre) else 0.0
    report = [
        "# E00 Schema Harness Report (Real Smoke)",
        "",
        f"- image_preflight_n: {len(pre)}",
        f"- image_path_ok_rate: {path_ok:.4f}",
        f"- parse_ok: {parse_ok:.4f}",
        f"- schema_ok: {schema_ok:.4f}",
        f"- invalid_enum_rate: {float(res['invalid_enum'].mean()) if len(res) else 0.0:.4f}",
        "",
        "Gate for E02/E05/E06:",
        "- image_path_ok_rate == 1.0",
        "- parse_ok >= 0.995",
        "- schema_ok >= 0.995",
    ]
    (out / "schema_harness_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
