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

SYSTEM = "You are a strict visual evidence checker for insulator defects. Return JSON only."


def prompt(record_id: str, row: dict) -> str:
    return f"""You receive TWO views of same crop: main view and zoomed center.
Record id: {record_id}
DINO candidates: top1={row.get('dino_top1')}, top2={row.get('dino_top2')}, top3={row.get('dino_top3')}

Return JSON:
{{
  "record_id": "{record_id}",
  "flashover_evidence": "<yes|no|uncertain>",
  "broken_evidence": "<yes|no|uncertain>",
  "confounders_present": "<yes|no|uncertain>",
  "needs_review": true,
  "selected_class_hint": "<top1|top2|top3|uncertain>",
  "short_reason": "..."
}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
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
            "image_main,image_zoom,view_main,view_context,view_zoom",
        ]
        subprocess.check_call(cmd)
        summary_path = Path(preflight_dir) / "path_preflight_summary.json"
        if summary_path.exists():
            info = json.loads(summary_path.read_text(encoding="utf-8"))
            if info.get("resolved_manifest"):
                manifest_for_run = Path(str(info["resolved_manifest"]))

    df = pd.read_csv(manifest_for_run)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    b = QwenHFBackend(args.model_name, max_new_tokens=180, max_pixels=401408)
    rows = []
    image_ok = 0
    for _, r in df.iterrows():
        rid = str(r["record_id"])
        raw = ""
        ok = False
        perr = ""
        obj = {}
        try:
            main_p = Path(str(r.get("image_main", r.get("view_main", ""))))
            zoom_p = Path(str(r.get("image_zoom", r.get("view_zoom", r.get("view_context", "")))))
            if (not main_p.exists()) or (not zoom_p.exists()):
                raise FileNotFoundError(f"main={main_p} zoom={zoom_p}")
            image_ok += 1
            raw = b.generate(SYSTEM, prompt(rid, r.to_dict()), [main_p, zoom_p])
            parsed, perr = parse_json_object(raw)
            if parsed is not None:
                obj = parsed
                obj["record_id"] = rid
                ok = True
        except Exception as exc:
            perr = f"runtime_error: {exc}"
        rows.append({"record_id": rid, **obj, "raw_response": raw, "parse_ok": ok, "parse_error": perr})
        print(f"{rid}: {ok}")
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    parse_ok_rate = float(pd.Series([r["parse_ok"] for r in rows]).mean()) if rows else 0.0
    runtime_error_rate = float(pd.Series([str(r.get("parse_error", "")).startswith("runtime_error:") for r in rows]).mean()) if rows else 0.0
    image_load_success_rate = (float(image_ok) / float(len(rows))) if rows else 0.0
    schema_ok_rate = float(
        pd.Series([str(r.get("selected_class_hint", "")) in {"top1", "top2", "top3", "uncertain"} for r in rows]).mean()
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
