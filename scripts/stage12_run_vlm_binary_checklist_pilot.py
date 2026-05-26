#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED = {"yes", "no", "uncertain"}
FIELDS = [
    "insulator_visible",
    "crop_quality_issue",
    "partial_or_shifted_crop",
    "missing_fragment_visible",
    "edge_discontinuity_visible",
    "crack_or_break_visible",
    "burn_or_arc_trace_visible",
    "dark_surface_trace_visible",
    "intact_regular_shape_visible",
    "shadow_glare_or_background_confounder",
]

SYSTEM_PROMPT = "You are a strict visual inspection checker. Return JSON only."


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty_response"
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        braced = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if braced:
            text = braced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(parsed, dict):
        return None, "parsed_json_is_not_object"
    return parsed, ""


def schema_ok(obj: dict[str, Any]) -> bool:
    if not isinstance(obj.get("record_id", ""), str):
        return False
    for f in FIELDS:
        if obj.get(f) not in ALLOWED:
            return False
    if not isinstance(obj.get("needs_review"), bool):
        return False
    if not isinstance(obj.get("one_sentence_observation", ""), str):
        return False
    return True


def build_user_prompt(record_id: str, row: dict[str, Any]) -> str:
    return f"""Inspect this insulator crop and answer visual checklist only.
Record id: {record_id}
DINO candidates for context only: top1={row.get('dino_top1')}, top2={row.get('dino_top2')}, top3={row.get('dino_top3')}.

Rules:
- Answer YES only when the cue is directly visible.
- Shadows/glare/background are confounders, not defects.
- If uncertain, answer "uncertain" and set needs_review=true.
- Do not output final defect class.
- Return strict JSON only.

Required JSON:
{{
  "record_id": "{record_id}",
  "insulator_visible": "<yes|no|uncertain>",
  "crop_quality_issue": "<yes|no|uncertain>",
  "partial_or_shifted_crop": "<yes|no|uncertain>",
  "missing_fragment_visible": "<yes|no|uncertain>",
  "edge_discontinuity_visible": "<yes|no|uncertain>",
  "crack_or_break_visible": "<yes|no|uncertain>",
  "burn_or_arc_trace_visible": "<yes|no|uncertain>",
  "dark_surface_trace_visible": "<yes|no|uncertain>",
  "intact_regular_shape_visible": "<yes|no|uncertain>",
  "shadow_glare_or_background_confounder": "<yes|no|uncertain>",
  "needs_review": true,
  "one_sentence_observation": "..."
}}
"""


class QwenHFBackend:
    def __init__(self, model_name: str, max_new_tokens: int = 220) -> None:
        import torch  # type: ignore
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, "float16", "auto"),
            attn_implementation="eager",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            self.process_vision_info = process_vision_info
        except Exception:
            self.process_vision_info = None

    def generate(self, system_prompt: str, user_prompt: str, image_path: Path) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image_path.resolve().as_uri()}, {"type": "text", "text": user_prompt}]},
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.process_vision_info is not None:
            image_inputs, _ = self.process_vision_info(messages)
        else:
            from PIL import Image  # type: ignore

            image_inputs = [Image.open(image_path).convert("RGB")]
        inputs = self.processor(text=[prompt_text], images=image_inputs, padding=True, return_tensors="pt").to(
            self.model.device
        )
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out_ids = gen[:, inputs.input_ids.shape[1] :]
        text = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", default="outputs/stage12/structured_output_v2_binary_real_pilot/pilot_manifest.csv")
    ap.add_argument("--out-dir", default="outputs/stage12/structured_output_v2_binary_real_pilot")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    backend = QwenHFBackend(args.model_name)
    rows = []
    for _, r in df.iterrows():
        rid = str(r["record_id"])
        img = Path(str(r["resolved_image_path"]))
        prompt = build_user_prompt(rid, r.to_dict())
        raw = ""
        parsed: dict[str, Any] | None = None
        parse_error = ""
        ok = False
        try:
            raw = backend.generate(SYSTEM_PROMPT, prompt, img)
            parsed, parse_error = parse_json_object(raw)
            if parsed is not None:
                parsed.setdefault("record_id", rid)
                ok = schema_ok(parsed)
        except Exception as exc:
            parse_error = f"runtime_error: {exc}"
            parsed = None
        rows.append(
            {
                "record_id": rid,
                "raw_response": raw,
                "parse_ok": bool(parsed is not None),
                "schema_ok": bool(ok),
                "parse_error": parse_error,
                "parsed": parsed if parsed is not None else {},
            }
        )
        print(f"{rid}: parse_ok={rows[-1]['parse_ok']} schema_ok={rows[-1]['schema_ok']}")

    # write outputs
    out_jsonl = out / "vlm_binary_checklist_outputs.jsonl"
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            obj = {"record_id": r["record_id"], **(r["parsed"] or {})}
            obj["raw_response"] = r["raw_response"]
            obj["parse_ok"] = r["parse_ok"]
            obj["schema_ok"] = r["schema_ok"]
            obj["parse_error"] = r["parse_error"]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    val = pd.DataFrame(
        [
            {
                "n": len(rows),
                "parse_ok_rate": float(pd.Series([x["parse_ok"] for x in rows]).mean()) if rows else 0.0,
                "schema_ok_rate": float(pd.Series([x["schema_ok"] for x in rows]).mean()) if rows else 0.0,
            }
        ]
    )
    val.to_csv(out / "binary_checklist_validation.csv", index=False)
    (out / "binary_checklist_validation.md").write_text(val.to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
