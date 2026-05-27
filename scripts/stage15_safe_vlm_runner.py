#!/usr/bin/env python3
"""Safe VLM runner for Stage15 E02/E05/E06.

This script is intentionally independent from older stage13_run_vlm_*.py scripts.
It adds:
- strict path preflight
- CUDA capability guard
- CPU fallback for smoke
- structured JSON extraction
- schema validation
- run validity status

Tasks:
- e02: flashover overclaim checker
- e05: claim verification
- e06: multiview / asset packet evidence check
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Optional

def early_env_for_device(device: str) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> tuple[Optional[dict], str]:
    if text is None:
        return None, "empty_response"
    s = str(text).strip()

    # Strip code fences.
    s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.I).strip()
    s = re.sub(r"```$", "", s.strip()).strip()

    # Try direct.
    try:
        return json.loads(s), ""
    except Exception:
        pass

    # Find balanced first JSON object.
    start = s.find("{")
    if start < 0:
        return None, "no_open_brace"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i+1]
                    cand = re.sub(r",\s*([}\]])", r"\1", cand)
                    try:
                        return json.loads(cand), ""
                    except Exception as e:
                        return None, f"json_load_error:{e}"
    return None, "no_balanced_json"


SCHEMAS = {
    "e02": {
        "direct_flashover_evidence": {"yes", "no", "uncertain"},
        "evidence_location": {"on_insulator", "background_or_shadow", "uncertain", "none"},
        "claim_supported": {"yes", "no", "uncertain"},
    },
    "e05": {
        "verification": {"supported", "contradicted", "not_enough_evidence"},
    },
    "e06": {
        "evidence_consistency": {"consistent", "inconsistent", "not_enough_evidence"},
        "best_view": {"main", "zoom", "context", "none", "uncertain"},
    },
}


def validate_schema(task: str, obj: Optional[dict]) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_dict"
    if not obj.get("record_id"):
        return False, "missing_record_id"

    if task == "e02":
        required = [
            "direct_flashover_evidence", "evidence_location", "visible_evidence_types",
            "possible_confounders", "claim_supported", "recommend_review", "short_reason"
        ]
    elif task == "e05":
        required = [
            "claim_id", "verification", "visible_basis", "confounders", "needs_review"
        ]
    elif task == "e06":
        required = [
            "evidence_consistency", "best_view", "need_reacquisition", "needs_review", "short_reason"
        ]
    else:
        return False, f"unknown_task:{task}"

    for k in required:
        if k not in obj:
            return False, f"missing:{k}"

    for k, allowed in SCHEMAS.get(task, {}).items():
        if obj.get(k) not in allowed:
            return False, f"invalid_enum:{k}={obj.get(k)}"

    # Basic list fields.
    for k in ["visible_evidence_types", "possible_confounders", "confounders"]:
        if k in obj and not isinstance(obj[k], list):
            return False, f"not_list:{k}"
    return True, ""


def make_prompt(task: str, row: dict) -> str:
    rid = row.get("record_id", "")
    d1 = row.get("dino_top1", row.get("top1_class", ""))
    d2 = row.get("dino_top2", row.get("top2_class", ""))
    d3 = row.get("dino_top3", row.get("top3_class", ""))
    margin = row.get("dino_margin", row.get("margin", ""))

    if task == "e02":
        return f"""
You are checking whether a predicted flashover defect is visually supported.
Do not classify the image from scratch.
Only judge the flashover claim.

Record ID: {rid}
DINO top1: {d1}
DINO top2: {d2}
DINO top3: {d3}
DINO margin: {margin}

Return ONLY valid JSON with this schema:
{{
  "record_id": "{rid}",
  "direct_flashover_evidence": "yes|no|uncertain",
  "evidence_location": "on_insulator|background_or_shadow|uncertain|none",
  "visible_evidence_types": ["burn_trace|arc_trace|dark_surface_trace|surface_damage"],
  "possible_confounders": ["shadow|glare|background|blur|low_contrast|partial_crop"],
  "claim_supported": "yes|no|uncertain",
  "recommend_review": true,
  "short_reason": "brief visual reason"
}}

Rules:
- Support flashover only if burn/arc/dark trace is directly visible on the insulator surface.
- Shadows, background, glare, blur, and low contrast are confounders, not flashover evidence.
- If unsure, use "uncertain" and recommend_review=true.
""".strip()

    if task == "e05":
        claim_id = row.get("claim_id") or "candidate_claim"
        claim_text = row.get("claim_text") or row.get("claim") or "There is visible defect evidence on the insulator."
        return f"""
Verify the visual claim using the image.
Do not infer beyond visible evidence.

Record ID: {rid}
Claim ID: {claim_id}
Claim: {claim_text}

Return ONLY valid JSON:
{{
  "record_id": "{rid}",
  "claim_id": "{claim_id}",
  "verification": "supported|contradicted|not_enough_evidence",
  "visible_basis": "brief image-grounded basis",
  "confounders": ["shadow|glare|blur|partial_crop|background|low_contrast"],
  "needs_review": true
}}

Rules:
- "supported" only if the claim is directly visible.
- "contradicted" if the visible image clearly goes against the claim.
- "not_enough_evidence" if crop quality or visibility is insufficient.
""".strip()

    if task == "e06":
        return f"""
Inspect the available view or views of the same insulator/asset.
Judge whether visual evidence is consistent enough for review.

Record ID: {rid}
DINO top1: {d1}
DINO top2: {d2}
DINO top3: {d3}

Return ONLY valid JSON:
{{
  "record_id": "{rid}",
  "evidence_consistency": "consistent|inconsistent|not_enough_evidence",
  "best_view": "main|zoom|context|none|uncertain",
  "need_reacquisition": true,
  "needs_review": true,
  "short_reason": "brief visual reason"
}}

Rules:
- If no view is adequate, set evidence_consistency="not_enough_evidence", best_view="none", need_reacquisition=true.
- If views disagree or are confounded, set inconsistent/uncertain and needs_review=true.
""".strip()

    raise ValueError(f"Unknown task: {task}")


def select_image_paths(task: str, row: dict) -> list[str]:
    # Use resolved paths first.
    candidates = []
    for k in [
        "resolved_image_path", "image_path", "crop_path",
        "main_image_path", "image_main", "context_image_path", "image_context",
        "zoom_image_path", "image_zoom",
    ]:
        v = row.get(k)
        if v and str(v).lower() not in {"nan", "none", "null"}:
            candidates.append(str(v))

    # For e02/e05 use one image. For e06 allow up to 3 distinct paths.
    seen = []
    for x in candidates:
        if x not in seen and Path(x).exists():
            seen.append(x)
    if task == "e06":
        return seen[:3]
    return seen[:1]


def detect_device(requested: str, allow_cpu_full: bool, limit: Optional[int]) -> tuple[str, dict]:
    early_env_for_device(requested)
    import torch

    info = {"requested": requested, "cuda_available": torch.cuda.is_available()}
    if requested == "cpu":
        return "cpu", info

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        info.update({"cuda_name": name, "cuda_capability": f"{cap[0]}.{cap[1]}"})
        if cap[0] < 7:
            # Current Kaggle torch often does not support P100 sm_60.
            if requested == "cuda":
                raise RuntimeError(
                    f"GPU {name} capability {cap} is too old for current PyTorch wheel. "
                    "Use Kaggle T4/P100-compatible torch, or run smoke with --device cpu."
                )
            # Auto mode: CPU only for tiny smoke, otherwise fail.
            if (limit is not None and limit <= 5) or allow_cpu_full:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return "cpu", info | {"forced_cpu_reason": "cuda_capability_lt_7"}
            raise RuntimeError(
                f"Auto detected unsupported GPU {name} capability {cap}. "
                "Use Kaggle T4 for full run, or pass --device cpu --limit <=5 for smoke."
            )
        return "cuda", info

    if requested == "cuda":
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return "cpu", info


class QwenRunner:
    def __init__(self, model_name: str, device_request: str, allow_cpu_full: bool, limit: Optional[int]):
        device, info = detect_device(device_request, allow_cpu_full, limit)
        self.device = device
        self.device_info = info | {"selected_device": device}

        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.processor_class = self.processor.__class__.__name__

        errors = []
        self.model = None

        # Official class selection by model family.
        candidates = []
        try:
            from transformers import Qwen2VLForConditionalGeneration
            from transformers import Qwen2_5_VLForConditionalGeneration
            if "Qwen2.5-VL" in model_name:
                candidates = [("Qwen2_5_VLForConditionalGeneration", Qwen2_5_VLForConditionalGeneration)]
            elif "Qwen2-VL" in model_name:
                candidates = [("Qwen2VLForConditionalGeneration", Qwen2VLForConditionalGeneration)]
            else:
                candidates = [
                    ("Qwen2_5_VLForConditionalGeneration", Qwen2_5_VLForConditionalGeneration),
                    ("Qwen2VLForConditionalGeneration", Qwen2VLForConditionalGeneration),
                ]
        except Exception as e:
            errors.append(f"import_qwen_classes: {e}")

        dtype = torch.float32 if device == "cpu" else torch.float16

        for name, cls in candidates:
            try:
                kwargs = {
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                # Avoid flash attention surprises.
                try:
                    model = cls.from_pretrained(model_name, attn_implementation="eager", **kwargs)
                except TypeError:
                    model = cls.from_pretrained(model_name, **kwargs)

                model.eval()
                if device == "cuda":
                    model.to("cuda")
                else:
                    model.to("cpu")
                self.model = model
                self.model_class = name
                return
            except Exception as e:
                errors.append(f"{name}: {e}")

        raise RuntimeError("Could not load model: " + " | ".join(errors[-5:]))

    def generate(self, image_paths: list[str], prompt: str, max_new_tokens: int = 220) -> tuple[str, dict]:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        normalized_paths = []
        image_sizes = []
        for p in image_paths:
            # Normalize and url-decode here for better runtime diagnostics.
            decoded = urllib.parse.unquote(str(p)).replace("%5C", "/").replace("\\", "/")
            path = Path(decoded)
            if not path.exists():
                raise FileNotFoundError(f"image_not_found:{decoded}")
            normalized_paths.append(str(path))
            with Image.open(path) as img:
                image_sizes.append({"path": str(path), "size": [int(img.width), int(img.height)]})

        content = [{"type": "image", "image": p} for p in normalized_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors.
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self.device)

        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        if "input_ids" in inputs:
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        else:
            generated_ids_trimmed = generated_ids

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        trace = {
            "rendered_chat_template_text_head": text[:1000],
            "rendered_chat_template_text_len": len(text),
            "image_paths": normalized_paths,
            "image_sizes": image_sizes,
        }
        return output_text, trace


def run(args) -> dict:
    from stage15_safe_path_resolver import preflight_manifest

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "raw_outputs.jsonl"
    if raw_out.exists():
        raw_out.unlink()

    run_status = {
        "task": args.task,
        "model_name": args.model_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "STARTED",
        "valid_for_metrics": False,
        "errors": [],
    }

    try:
        preflight_manifest(
            manifest_path=Path(args.manifest_csv),
            out_dir=out_dir,
            roots=args.root or [],
            fail_on_missing=True,
        )
        resolved_manifest = out_dir / "manifest_resolved.csv"
        rows = read_csv(resolved_manifest)
        if args.limit:
            rows = rows[: args.limit]

        # Pre-load image path check.
        for r in rows:
            paths = select_image_paths(args.task, r)
            if not paths:
                raise RuntimeError(f"No existing image paths for record_id={r.get('record_id')}")

        runner = QwenRunner(args.model_name, args.device, args.allow_cpu_full, args.limit)
        run_status["device_info"] = runner.device_info
        run_status["model_class"] = getattr(runner, "model_class", "")

        results = []
        for i, row in enumerate(rows):
            rid = row.get("record_id", str(i))
            rec = {
                "record_id": rid,
                "task": args.task,
                "parse_ok": False,
                "schema_ok": False,
                "image_load_success": False,
                "runtime_error": "",
                "raw_response": "",
            }
            try:
                image_paths = select_image_paths(args.task, row)
                rec["image_paths"] = image_paths
                if not image_paths:
                    raise RuntimeError("no_existing_image_paths")
                rec["image_load_success"] = True
                prompt = make_prompt(args.task, row)
                if not isinstance(prompt, str):
                    raise RuntimeError("prompt_not_string")
                if len(prompt.strip()) <= 50:
                    raise RuntimeError(f"prompt_too_short:{len(prompt.strip())}")

                # Always store diagnostic heads even without --save-prompts.
                rec["prompt"] = prompt[:1000]
                rec["prompt_len"] = len(prompt)
                rec["model_class"] = getattr(runner, "model_class", "")
                rec["processor_class"] = getattr(runner, "processor_class", "")

                raw, trace = runner.generate(image_paths, prompt, max_new_tokens=args.max_new_tokens)
                rec["rendered_chat_template_text_head"] = trace.get("rendered_chat_template_text_head", "")
                rec["rendered_chat_template_text_len"] = trace.get("rendered_chat_template_text_len", 0)
                rec["image_paths"] = trace.get("image_paths", image_paths)
                rec["image_sizes"] = trace.get("image_sizes", [])
                rec["raw_response"] = raw
                obj, err = extract_json_object(raw)
                rec["parse_error"] = err
                if obj is not None:
                    rec["parse_ok"] = True
                    # Make record_id robust if model omitted it.
                    obj.setdefault("record_id", rid)
                    ok, schema_err = validate_schema(args.task, obj)
                    rec["schema_ok"] = ok
                    rec["schema_error"] = schema_err
                    rec["parsed"] = obj
                else:
                    rec["schema_error"] = "parse_failed"
            except Exception as e:
                rec["runtime_error"] = repr(e)

            append_jsonl(raw_out, rec)
            results.append(rec)
            print(f"[{i+1}/{len(rows)}] {rid} parse={rec['parse_ok']} schema={rec['schema_ok']} err={rec.get('runtime_error') or rec.get('parse_error') or rec.get('schema_error')}")

        n = len(results)
        image_ok = sum(bool(r.get("image_load_success")) for r in results)
        parse_ok = sum(bool(r.get("parse_ok")) for r in results)
        schema_ok = sum(bool(r.get("schema_ok")) for r in results)
        runtime_err = sum(1 for r in results if r.get("runtime_error"))

        summary = {
            "n": n,
            "image_load_success": image_ok,
            "image_load_success_rate": image_ok / n if n else 0,
            "parse_ok": parse_ok,
            "parse_ok_rate": parse_ok / n if n else 0,
            "schema_ok": schema_ok,
            "schema_ok_rate": schema_ok / n if n else 0,
            "runtime_errors": runtime_err,
            "runtime_error_rate": runtime_err / n if n else 0,
        }
        write_csv(out_dir / "validation_summary.csv", [summary])
        # flatten selected failure rows
        failures = []
        for r in results:
            if not (r.get("image_load_success") and r.get("parse_ok") and r.get("schema_ok")):
                failures.append({
                    "record_id": r.get("record_id"),
                    "image_paths": json.dumps(r.get("image_paths", []), ensure_ascii=False),
                    "runtime_error": r.get("runtime_error", ""),
                    "parse_error": r.get("parse_error", ""),
                    "schema_error": r.get("schema_error", ""),
                    "raw_response_head": str(r.get("raw_response", ""))[:300],
                })
        write_csv(out_dir / "first_20_failures.csv", failures[:20])

        valid = (
            summary["image_load_success_rate"] >= args.min_image_success
            and summary["parse_ok_rate"] >= args.min_parse_ok
            and summary["schema_ok_rate"] >= args.min_schema_ok
        )
        run_status.update(summary)
        run_status["valid_for_metrics"] = bool(valid)
        run_status["status"] = "OK" if valid else "INVALID"
        return run_status
    except Exception as e:
        run_status["status"] = "ERROR"
        run_status["errors"].append(repr(e))
        return run_status
    finally:
        run_status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (Path(args.out_dir) / "run_status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["e02", "e05", "e06"])
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--allow-cpu-full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--min-image-success", type=float, default=1.0)
    ap.add_argument("--min-parse-ok", type=float, default=0.8)
    ap.add_argument("--min-schema-ok", type=float, default=0.8)
    ap.add_argument("--save-prompts", action="store_true")
    args = ap.parse_args()

    # Set env early enough for imports in this module path.
    early_env_for_device(args.device)

    status = run(args)
    print(json.dumps(status, indent=2))
    if status.get("status") == "ERROR":
        sys.exit(1)
    if not status.get("valid_for_metrics"):
        sys.exit(2)


if __name__ == "__main__":
    main()
