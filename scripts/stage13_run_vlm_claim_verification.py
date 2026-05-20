#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from scripts.stage13_real_vlm_utils import QwenHFBackend, parse_json_object
except Exception:
    from stage13_real_vlm_utils import QwenHFBackend, parse_json_object

SYSTEM = "You are a strict claim verifier for visual evidence. Return JSON only."


def prompt(record_id: str, claim_id: str, claim_text: str, row: dict) -> str:
    return f"""Verify this claim from the image.
Record id: {record_id}
Claim id: {claim_id}
Claim text: {claim_text}
DINO candidates for context only: top1={row.get('dino_top1')}, top2={row.get('dino_top2')}, top3={row.get('dino_top3')}

Output strict JSON:
{{
  "record_id": "{record_id}",
  "claim_id": "{claim_id}",
  "verification": "<supported|contradicted|not_enough_evidence>",
  "visual_basis": "...",
  "confounders": ["shadow","blur","partial_crop","background","glare","low_contrast"],
  "needs_review": true
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preflight-out-dir", default="")
    ap.add_argument("--min-parse-ok-rate", type=float, default=0.90)
    args = ap.parse_args()

    preflight_dir = args.preflight_out_dir or str(Path(args.out_jsonl).parent / "preflight")
    manifest_for_run = Path(args.manifest_csv)
    if preflight_dir:
        cmd = [
            sys.executable,
            "scripts/stage15_resolve_image_paths.py",
            "--manifest",
            args.manifest_csv,
            "--out-dir",
            preflight_dir,
            "--path-cols",
            "resolved_image_path",
        ]
        subprocess.check_call(cmd)
        summary_path = Path(preflight_dir) / "path_preflight_summary.json"
        if summary_path.exists():
            info = json.loads(summary_path.read_text(encoding="utf-8"))
            if info.get("resolved_manifest"):
                manifest_for_run = Path(str(info["resolved_manifest"]))

    df = pd.read_csv(manifest_for_run)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    backend = QwenHFBackend(args.model_name, max_new_tokens=170, max_pixels=401408)

    rows = []
    image_ok = 0
    for _, r in df.iterrows():
        rid = str(r["record_id"])
        cid = str(r["claim_id"])
        raw = ""
        parse_ok = False
        perr = ""
        obj = {}
        try:
            image_path = Path(str(r["resolved_image_path"]))
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            image_ok += 1
            raw = backend.generate(
                SYSTEM,
                prompt(rid, cid, str(r["claim_text"]), r.to_dict()),
                [image_path],
            )
            parsed, perr = parse_json_object(raw)
            if parsed is not None:
                obj = parsed
                obj["record_id"] = rid
                obj["claim_id"] = cid
                parse_ok = True
        except Exception as exc:
            perr = f"runtime_error: {exc}"
        rows.append({"record_id": rid, "claim_id": cid, **obj, "raw_response": raw, "parse_ok": parse_ok, "parse_error": perr})
        print(f"{rid} {cid}: parse_ok={parse_ok}")

    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parse_ok_rate = float(pd.Series([r["parse_ok"] for r in rows]).mean()) if rows else 0.0
    runtime_error_rate = float(pd.Series([str(r.get("parse_error", "")).startswith("runtime_error:") for r in rows]).mean()) if rows else 0.0
    image_load_success_rate = (float(image_ok) / float(len(rows))) if rows else 0.0
    schema_ok_rate = float(
        pd.Series([str(r.get("verification", "")) in {"supported", "contradicted", "not_enough_evidence"} for r in rows]).mean()
    ) if rows else 0.0
    status = "VALID" if (image_load_success_rate >= 1.0 and parse_ok_rate >= args.min_parse_ok_rate) else "INVALID_RUN"
    pd.DataFrame(
        [
            {
                "n": len(rows),
                "image_load_success_rate": image_load_success_rate,
                "parse_ok_rate": parse_ok_rate,
                "schema_ok_rate": schema_ok_rate,
                "runtime_error_rate": runtime_error_rate,
                "run_status": status,
            }
        ]
    ).to_csv(outp.with_suffix(".summary.csv"), index=False)
    if status != "VALID":
        raise SystemExit(f"INVALID_RUN: image_load_success_rate={image_load_success_rate:.4f}, parse_ok_rate={parse_ok_rate:.4f}")


if __name__ == "__main__":
    main()
