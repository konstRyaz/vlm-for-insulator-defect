#!/usr/bin/env python3
"""Debug a single Stage15 VLM call with full prompt/render tracing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image

from stage15_safe_vlm_runner import QwenRunner, make_prompt, select_image_paths


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["e02", "e05", "e06"])
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    rows = read_csv(Path(args.manifest_csv))
    if not rows:
        raise RuntimeError("empty_manifest")
    if args.row_index < 0 or args.row_index >= len(rows):
        raise RuntimeError(f"row_index_out_of_range:{args.row_index}")

    row = rows[args.row_index]
    rid = row.get("record_id", f"row_{args.row_index}")
    image_paths = select_image_paths(args.task, row)
    if not image_paths:
        raise RuntimeError("no_existing_image_paths")

    image_meta = []
    for p in image_paths:
        with Image.open(p) as img:
            image_meta.append({"path": p, "size": [int(img.width), int(img.height)]})

    prompt = make_prompt(args.task, row)
    if not isinstance(prompt, str) or len(prompt.strip()) <= 50:
        raise RuntimeError(f"invalid_prompt_len:{len(prompt.strip()) if isinstance(prompt, str) else -1}")

    runner = QwenRunner(args.model_name, args.device, allow_cpu_full=False, limit=1)
    raw_text, trace = runner.generate(image_paths, prompt, max_new_tokens=args.max_new_tokens)

    payload = {
        "task": args.task,
        "record_id": rid,
        "model_name": args.model_name,
        "device_info": runner.device_info,
        "model_class": getattr(runner, "model_class", ""),
        "processor_class": getattr(runner, "processor_class", ""),
        "image_sizes": image_meta,
        "prompt_len": len(prompt),
        "prompt": prompt,
        "rendered_chat_template_text_len": trace.get("rendered_chat_template_text_len", 0),
        "rendered_chat_template_text_head": trace.get("rendered_chat_template_text_head", ""),
        "raw_decoded_output": raw_text,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), "record_id": rid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
