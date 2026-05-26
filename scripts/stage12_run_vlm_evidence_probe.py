#!/usr/bin/env python3
"""Stage 12 evidence-first VLM probe runner."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_VISIBLE = {"yes", "no", "uncertain"}
ALLOWED_VISIBILITY = {"clear", "partial", "ambiguous", "bad"}
ALLOWED_TAGS = {
    "intact_structure",
    "regular_disc_shape",
    "missing_fragment",
    "edge_discontinuity",
    "burn_like_mark",
    "surface_stain",
    "ambiguous_evidence",
    "surface_damage_mark",
    "blurred_region",
    "partial_view",
    "dark_surface_trace",
    "unclear_boundary",
    "no_visible_break",
    "occluded_region",
    "low_contrast",
}


SYSTEM_PROMPT = (
    "You are a careful visual evidence extractor for insulator crops. "
    "Return strict JSON only."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        tags = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str) and value.strip():
        tags = [value.strip()]
    else:
        tags = []
    deduped: list[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def schema_ok(row: dict[str, Any]) -> bool:
    return (
        row.get("visible_insulator") in ALLOWED_VISIBLE
        and row.get("visibility") in ALLOWED_VISIBILITY
        and isinstance(row.get("needs_review"), bool)
        and isinstance(row.get("evidence_tags"), list)
        and all(isinstance(x, str) for x in row.get("evidence_tags", []))
        and isinstance(row.get("short_canonical_description", ""), str)
    )


def resolve_image_path(raw: str, base_dir: Path) -> Path:
    normalized = str(raw).replace("\\", "/")
    p = Path(normalized)
    candidates = [p]
    if not p.is_absolute():
        candidates.extend(
            [
                base_dir / p,
                Path.cwd() / p,
                Path.cwd() / "images" / p.name,
                Path.cwd().parent / "images" / p.name,
                Path.cwd() / p.name,
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not resolve image: {raw}")


def build_prompt(style: str, record_id: str, row: dict[str, Any]) -> str:
    tag_text = ", ".join(sorted(ALLOWED_TAGS))
    if style == "candidate_support_v1":
        top1 = row.get("dino_top1", "unknown")
        top2 = row.get("dino_top2", "unknown")
        top3 = row.get("dino_top3", "unknown")
        extras = (
            f"DINO candidates (for context only): top1={top1}, top2={top2}, top3={top3}. "
            "Do not blindly trust candidates; output evidence from image."
        )
    elif style == "grounded_report_v1":
        extras = (
            "Additionally provide a short grounded description that names only visibly supported cues. "
            "Avoid conclusions that are not directly observable."
        )
    else:
        extras = "Image-only probe: do not assume class labels or candidate predictions."

    return f"""Analyze this crop image and extract only visible evidence.
Record id: {record_id}
{extras}

Allowed evidence_tags:
[{tag_text}]

Output rules:
- Return exactly one JSON object (no markdown, no prose).
- Do not copy fixed placeholder values; choose values from what is visible in this image.
- If evidence is uncertain, set visibility="ambiguous" or "bad" and needs_review=true.
- Use only tags from the allowed list.

Required JSON schema (fill with image-specific values):
{{
  "record_id": "{record_id}",
  "visible_insulator": "<yes|no|uncertain>",
  "visibility": "<clear|partial|ambiguous|bad>",
  "needs_review": <true|false>,
  "evidence_tags": ["<allowed_tag_1>", "<allowed_tag_2>"],
  "short_canonical_description": "<short image-grounded phrase>"
}}
"""


class MockBackend:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate(self, _system: str, _user: str, _image_path: Path) -> str:
        # Deterministic schema-valid mock for smoke tests only.
        _ = self.rng.random()
        return json.dumps(
            {
                "visible_insulator": "yes",
                "visibility": "clear",
                "needs_review": False,
                "evidence_tags": ["intact_structure", "no_visible_break"],
                "short_canonical_description": "insulator appears intact",
            },
            ensure_ascii=False,
        )


class QwenHFBackend:
    def __init__(self, model_name: str, max_new_tokens: int, max_pixels: int | None) -> None:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import AutoProcessor  # type: ignore

        self.torch = torch
        self.Image = Image
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.process_vision_info = None
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            self.process_vision_info = process_vision_info
        except ImportError:
            self.process_vision_info = None

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        self.model = self._load_model(model_name)
        self.model.eval()

    def _load_model(self, model_name: str) -> Any:
        # T4 on Kaggle is most stable with fp16 + eager attention for Qwen VL.
        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": getattr(self.torch, "float16", "auto"),
            "attn_implementation": "eager",
        }
        errors: list[str] = []
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"Qwen2_5_VLForConditionalGeneration: {exc}")
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForImageTextToText: {exc}")
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            return AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForVision2Seq: {exc}")
        raise RuntimeError("Could not load model: " + " | ".join(errors))

    def _uri(self, p: Path) -> str:
        try:
            return p.resolve().as_uri()
        except ValueError:
            return p.resolve().as_posix()

    def generate(self, system_prompt: str, user_prompt: str, image_path: Path) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "image", "image": self._uri(image_path)}, {"type": "text", "text": user_prompt}],
            },
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images_input: Any = None
        videos_input: Any = None
        if self.process_vision_info is not None:
            try:
                images_input, videos_input = self.process_vision_info(messages)
            except Exception:
                images_input = None
                videos_input = None
        if images_input is None:
            with self.Image.open(image_path) as img:
                images_input = [img.convert("RGB")]

        kwargs: dict[str, Any] = {"text": [prompt_text], "images": images_input, "padding": True, "return_tensors": "pt"}
        if videos_input is not None:
            kwargs["videos"] = videos_input
        inputs = self.processor(**kwargs)
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0] if decoded else ""


def select_rows(
    manifest_rows: list[dict[str, Any]],
    split: str,
    pilot_mode: str,
    limit: int | None,
    eval_reference_csv: str | None,
    seed: int,
) -> list[dict[str, Any]]:
    rows = manifest_rows if split == "all" else [r for r in manifest_rows if str(r.get("split", "")) == split]
    if pilot_mode == "none":
        return rows[:limit] if limit is not None else rows

    if pilot_mode != "dev_errors_plus_controls":
        raise ValueError(f"Unsupported pilot mode: {pilot_mode}")
    if split != "train":
        raise ValueError("pilot_mode=dev_errors_plus_controls is allowed only on split=train")
    if not eval_reference_csv:
        raise ValueError("--eval-reference-csv is required for dev_errors_plus_controls")

    ref = pd.read_csv(eval_reference_csv)
    needed = {"record_id", "split", "dino_top1", "label_coarse_class"}
    if not needed.issubset(set(ref.columns)):
        raise ValueError(f"Eval reference missing required columns: {sorted(needed - set(ref.columns))}")
    ref = ref[ref["split"].astype(str) == "train"].copy()
    ref["is_error"] = ref["dino_top1"].astype(str) != ref["label_coarse_class"].astype(str)
    ref["label_visibility"] = ref.get("label_visibility", "unknown").astype(str)

    by_id = {str(r["record_id"]): r for r in rows}
    errors = [by_id[rid] for rid in ref.loc[ref["is_error"], "record_id"].astype(str) if rid in by_id]
    controls_df = ref.loc[~ref["is_error"]].copy()
    controls_df = controls_df.sample(frac=1.0, random_state=seed)

    # Stratified round-robin by dino_top1 x visibility.
    controls: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[str]] = {}
    for _, r in controls_df.iterrows():
        key = (str(r["dino_top1"]), str(r["label_visibility"]))
        groups.setdefault(key, []).append(str(r["record_id"]))
    for v in groups.values():
        random.Random(seed).shuffle(v)

    target = limit if limit is not None else (len(errors) + 30)
    if target < len(errors):
        target = len(errors)
    added = set(str(r["record_id"]) for r in errors)
    while len(errors) + len(controls) < target:
        progressed = False
        for key in sorted(groups.keys()):
            bucket = groups[key]
            while bucket and bucket[0] in added:
                bucket.pop(0)
            if not bucket:
                continue
            rid = bucket.pop(0)
            row = by_id.get(rid)
            if row is None:
                continue
            controls.append(row)
            added.add(rid)
            progressed = True
            if len(errors) + len(controls) >= target:
                break
        if not progressed:
            break
    selected = errors + controls
    return selected[:target]


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest_jsonl)
    manifest_rows = read_jsonl(manifest_path)
    selected = select_rows(
        manifest_rows=manifest_rows,
        split=args.split,
        pilot_mode=args.pilot_mode,
        limit=args.limit,
        eval_reference_csv=args.eval_reference_csv,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No rows selected.")

    if args.backend == "mock":
        backend: Any = MockBackend(args.seed)
    else:
        backend = QwenHFBackend(args.model_name, args.max_new_tokens, args.max_pixels)

    out_rows: list[dict[str, Any]] = []
    started = time.time()
    for i, row in enumerate(selected, 1):
        rid = str(row["record_id"])
        image = resolve_image_path(str(row["resolved_image_path"]), manifest_path.parent.parent.parent)
        prompt = build_prompt(args.prompt_style, rid, row)
        try:
            raw = backend.generate(SYSTEM_PROMPT, prompt, image)
            parsed, perr = parse_json_object(raw)
            if parsed is None:
                out = {
                    "record_id": rid,
                    "split": str(row.get("split", "")),
                    "visible_insulator": "uncertain",
                    "visibility": "ambiguous",
                    "needs_review": True,
                    "evidence_tags": [],
                    "short_canonical_description": "",
                    "raw_response": raw,
                    "parse_ok": False,
                    "parse_error": perr,
                    "schema_ok": False,
                }
            else:
                vis_ins = str(parsed.get("visible_insulator", "uncertain")).strip()
                visibility = str(parsed.get("visibility", "ambiguous")).strip()
                needs_review = safe_bool(parsed.get("needs_review", visibility in {"ambiguous", "bad"}))
                tags = normalize_tags(parsed.get("evidence_tags", []))
                desc = str(parsed.get("short_canonical_description", "")).strip()
                out = {
                    "record_id": rid,
                    "split": str(row.get("split", "")),
                    "visible_insulator": vis_ins if vis_ins in ALLOWED_VISIBLE else "uncertain",
                    "visibility": visibility if visibility in ALLOWED_VISIBILITY else "ambiguous",
                    "needs_review": needs_review,
                    "evidence_tags": tags,
                    "short_canonical_description": desc,
                    "raw_response": raw,
                    "parse_ok": True,
                    "parse_error": "",
                    "schema_ok": False,
                }
                # Enforce review on weak visibility.
                if out["visibility"] in {"ambiguous", "bad"} or out["visible_insulator"] == "uncertain":
                    out["needs_review"] = True
                out["schema_ok"] = schema_ok(out)
        except Exception as exc:
            out = {
                "record_id": rid,
                "split": str(row.get("split", "")),
                "visible_insulator": "uncertain",
                "visibility": "ambiguous",
                "needs_review": True,
                "evidence_tags": [],
                "short_canonical_description": "",
                "raw_response": "",
                "parse_ok": False,
                "parse_error": f"inference_error: {type(exc).__name__}: {exc}",
                "schema_ok": False,
            }
        out_rows.append(out)
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == len(selected)):
            print(f"[{i}/{len(selected)}] record_id={rid}", flush=True)

    out_path = Path(args.out_jsonl)
    write_jsonl(out_path, out_rows)
    summary = {
        "manifest_jsonl": str(manifest_path),
        "out_jsonl": str(out_path),
        "backend": args.backend,
        "model_name": args.model_name,
        "split": args.split,
        "pilot_mode": args.pilot_mode,
        "limit": args.limit,
        "prompt_style": args.prompt_style,
        "n_rows": len(out_rows),
        "parse_ok": sum(1 for r in out_rows if r["parse_ok"]),
        "schema_ok": sum(1 for r in out_rows if r["schema_ok"]),
        "elapsed_sec": round(time.time() - started, 3),
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.summary.json')}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-jsonl", required=True)
    p.add_argument("--eval-reference-csv", default=None)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="train")
    p.add_argument("--pilot-mode", choices=["none", "dev_errors_plus_controls"], default="none")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--prompt-style",
        choices=["evidence_probe_min_v1", "candidate_support_v1", "grounded_report_v1"],
        default="evidence_probe_min_v1",
    )
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["qwen_hf", "mock"], default="qwen_hf")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--max-pixels", type=int, default=401408)
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
