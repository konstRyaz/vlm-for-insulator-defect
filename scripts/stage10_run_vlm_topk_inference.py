#!/usr/bin/env python3
"""Run Stage 10 VLM top-k checker/reranker inference.

The input manifest must contain only class-neutral image paths and DINOv2
top-k candidates. Ground-truth labels and annotation fields are intentionally
not read by this script.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


ALLOWED_CLASSES = {"insulator_ok", "defect_flashover", "defect_broken", "unknown"}
ALLOWED_RANKS = {"top1", "top2", "top3", "uncertain"}
ALLOWED_SUPPORT = {"yes", "no", "uncertain"}
ALLOWED_EVIDENCE = {"strong", "medium", "weak", "none"}
ALLOWED_QUALITY = {"clear", "partial", "ambiguous", "bad"}


LEAKAGE_FORBIDDEN_FIELDS = {
    "label_coarse_class",
    "label_visual_evidence_tags",
    "label_visibility",
    "label_needs_review",
    "label_short_canonical_description",
    "label_report_snippet",
    "category_name",
    "annotator_notes",
    "label_version",
    "crop_path",
    "bbox_xywh",
}


REQUIRED_MANIFEST_FIELDS = {
    "record_id",
    "split",
    "resolved_image_path",
    "dino_top1",
    "dino_top2",
    "dino_top3",
    "dino_top1_score",
    "dino_top2_score",
    "dino_top3_score",
    "dino_margin",
}


SYSTEM_PROMPT = """You are a careful visual checker for insulator defect crops.
Return strict JSON only. Do not use file names or paths. Use only visible image evidence and the provided DINOv2 candidates.
"""


CLASS_DEFINITIONS = {
    "insulator_ok": (
        "intact or normal-looking insulator crop; no direct visual evidence of flashover "
        "or structural break"
    ),
    "defect_flashover": (
        "direct visible burn, arc trace, carbonization, localized dark electrical trace, "
        "or flashover-like residue on the insulator surface or fitting"
    ),
    "defect_broken": (
        "direct visible structural damage such as broken shed, missing fragment, crack, "
        "gap, discontinuity, or material loss"
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


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


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_output(parsed: dict[str, Any] | None, raw: str, row: dict[str, Any], parse_error: str) -> dict[str, Any]:
    candidate_by_rank = {
        "top1": str(row.get("dino_top1", "")).strip(),
        "top2": str(row.get("dino_top2", "")).strip(),
        "top3": str(row.get("dino_top3", "")).strip(),
    }
    candidates = {v for v in candidate_by_rank.values() if v}

    out: dict[str, Any] = {
        "record_id": str(row.get("record_id", "")),
        "split": str(row.get("split", "")),
        "selected_rank": "uncertain",
        "selected_class": "unknown",
        "supports_top1": "uncertain",
        "supports_top2": "uncertain",
        "supports_top3": "uncertain",
        "evidence_strength": "none",
        "crop_quality": "ambiguous",
        "needs_review": True,
        "evidence_tags": [],
        "contradiction_tags": [],
        "short_reason": "",
        "raw_response": raw,
        "parse_ok": False,
        "parse_error": parse_error,
    }
    if parsed is None:
        return out

    selected_rank = str(parsed.get("selected_rank", "uncertain")).strip()
    selected_class = str(parsed.get("selected_class", "unknown")).strip()
    if selected_rank not in ALLOWED_RANKS:
        selected_rank = "uncertain"
    if selected_class not in ALLOWED_CLASSES:
        selected_class = "unknown"

    # Enforce constrained reranking: a known class must be one of the DINOv2 candidates.
    if selected_rank in {"top1", "top2", "top3"}:
        expected = candidate_by_rank.get(selected_rank, "")
        if selected_class != expected:
            selected_class = expected if expected in ALLOWED_CLASSES else "unknown"
    if selected_class not in candidates and selected_class != "unknown":
        selected_rank = "uncertain"
        selected_class = "unknown"

    supports = {}
    for key in ["supports_top1", "supports_top2", "supports_top3"]:
        val = str(parsed.get(key, "uncertain")).strip().lower()
        supports[key] = val if val in ALLOWED_SUPPORT else "uncertain"

    evidence_strength = str(parsed.get("evidence_strength", "none")).strip().lower()
    if evidence_strength not in ALLOWED_EVIDENCE:
        evidence_strength = "none"
    crop_quality = str(parsed.get("crop_quality", "ambiguous")).strip().lower()
    if crop_quality not in ALLOWED_QUALITY:
        crop_quality = "ambiguous"

    needs_review = safe_bool(parsed.get("needs_review", selected_rank == "uncertain"))
    if selected_rank == "uncertain" or selected_class == "unknown":
        needs_review = True

    out.update(
        {
            "selected_rank": selected_rank,
            "selected_class": selected_class,
            **supports,
            "evidence_strength": evidence_strength,
            "crop_quality": crop_quality,
            "needs_review": needs_review,
            "evidence_tags": list_of_strings(parsed.get("evidence_tags", [])),
            "contradiction_tags": list_of_strings(parsed.get("contradiction_tags", [])),
            "short_reason": str(parsed.get("short_reason", ""))[:500],
            "parse_ok": True,
            "parse_error": "",
        }
    )
    return out


def build_user_prompt(row: dict[str, Any], prompt_style: str = "conservative") -> str:
    top1 = str(row["dino_top1"])
    top2 = str(row["dino_top2"])
    top3 = str(row["dino_top3"])
    s1 = as_float(row.get("dino_top1_score"))
    s2 = as_float(row.get("dino_top2_score"))
    s3 = as_float(row.get("dino_top3_score"))
    margin = as_float(row.get("dino_margin"))
    if prompt_style == "evidence_compare":
        decision_rules = """Decision rules:
- Compare the visible evidence for top1, top2, and top3 directly.
- Keep top1 only when top1 has the strongest visible support.
- Switch to top2 or top3 when that candidate has stronger direct visible evidence than top1.
- If two candidates are visually plausible and evidence is not decisive, choose selected_rank="uncertain", selected_class="unknown", needs_review=true.
- Do not force defect_flashover unless there is direct visible burn, arc, carbonization, or a localized dark electrical trace on the insulator surface/fitting.
- Do not force defect_broken unless there is direct visible structural damage, missing fragment, crack, discontinuity, or material loss.
- Do not introduce any known class outside top1/top2/top3."""
    elif prompt_style == "contrastive_decisive":
        decision_rules = """Decision rules:
- Compare top1 vs top2 first using visible evidence, then consider top3.
- Do not keep top1 by default; keep top1 only if it has stronger direct visible support than top2/top3.
- If top2 or top3 has stronger visible support, switch to that rank.
- Use selected_rank="uncertain" only when image quality is insufficient for reliable comparison.
- Prefer concrete top-rank decisions when evidence is readable.
- Do not force defect_flashover unless there is direct visible burn, arc, carbonization, or a localized dark electrical trace on the insulator surface/fitting.
- Do not force defect_broken unless there is direct visible structural damage, missing fragment, crack, discontinuity, or material loss.
- Do not introduce any known class outside top1/top2/top3."""
    elif prompt_style == "contrastive_decisive_noscore":
        decision_rules = """Decision rules:
- Compare top1 vs top2 first using visible evidence, then consider top3.
- Ignore score magnitudes; use image evidence as the primary signal.
- Do not keep top1 by default; keep top1 only if it has stronger direct visible support than top2/top3.
- If top2 or top3 has stronger visible support, switch to that rank.
- Use selected_rank="uncertain" only when image quality is insufficient for reliable comparison.
- Prefer concrete top-rank decisions when evidence is readable.
- Do not force defect_flashover unless there is direct visible burn, arc, carbonization, or a localized dark electrical trace on the insulator surface/fitting.
- Do not force defect_broken unless there is direct visible structural damage, missing fragment, crack, discontinuity, or material loss.
- Do not introduce any known class outside top1/top2/top3."""
    else:
        decision_rules = """Decision rules:
- If top1 is visually supported, prefer keeping top1.
- Do not force defect_flashover unless there is direct visible burn, arc, carbonization, or a localized dark electrical trace on the insulator surface/fitting.
- Do not force defect_broken unless there is direct visible structural damage, missing fragment, crack, discontinuity, or material loss.
- If evidence is insufficient, choose selected_rank="uncertain", selected_class="unknown", needs_review=true.
- Do not introduce any known class outside top1/top2/top3."""

    score_block = f"""- top1: {top1} (score={s1:.4f})
- top2: {top2} (score={s2:.4f})
- top3: {top3} (score={s3:.4f})
- margin top1-top2: {margin:.4f}"""
    if prompt_style == "contrastive_decisive_noscore":
        score_block = f"""- top1: {top1}
- top2: {top2}
- top3: {top3}"""

    return f"""Analyze the crop image and choose only among the provided DINOv2 candidates.

DINOv2 candidates:
{score_block}

Class definitions:
- insulator_ok: {CLASS_DEFINITIONS["insulator_ok"]}
- defect_flashover: {CLASS_DEFINITIONS["defect_flashover"]}
- defect_broken: {CLASS_DEFINITIONS["defect_broken"]}

{decision_rules}

Return exactly one JSON object. Do not copy option lists or placeholder strings.
For `selected_rank`, write exactly one value: "top1", "top2", "top3", or "uncertain".
For `selected_class`, write exactly one value from the three candidate class names or "unknown".

Use this shape, replacing every example value with one concrete value:
{{
  "record_id": "{row["record_id"]}",
  "selected_rank": "top1",
  "selected_class": "{top1}",
  "supports_top1": "yes",
  "supports_top2": "no",
  "supports_top3": "uncertain",
  "evidence_strength": "strong",
  "crop_quality": "clear",
  "needs_review": true,
  "evidence_tags": ["short_visual_tag"],
  "contradiction_tags": ["short_visual_tag"],
  "short_reason": "one short sentence"
}}
"""


class MockBackend:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def generate(self, _system_prompt: str, _user_prompt: str, _image_path: Path, row: dict[str, Any]) -> str:
        # Deterministic schema smoke-test output. This is not an experimental result.
        _ = self.rng.random()
        return json.dumps(
            {
                "record_id": row["record_id"],
                "selected_rank": "top1",
                "selected_class": row["dino_top1"],
                "supports_top1": "yes",
                "supports_top2": "uncertain",
                "supports_top3": "uncertain",
                "evidence_strength": "medium",
                "crop_quality": "clear",
                "needs_review": False,
                "evidence_tags": [],
                "contradiction_tags": [],
                "short_reason": "Mock output for pipeline validation only.",
            },
            ensure_ascii=False,
        )


class QwenHFBackend:
    def __init__(self, model_name: str, max_new_tokens: int, max_pixels: int | None) -> None:
        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
            from transformers import AutoProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Qwen backend requires transformers, accelerate, Pillow, torch, and qwen-vl-utils."
            ) from exc

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

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = self._load_model(model_name)
        self.model.eval()

    def _load_model(self, model_name: str) -> Any:
        model_kwargs = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
        errors: list[str] = []
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            errors.append(f"Qwen2_5_VLForConditionalGeneration: {exc}")
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            return AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForImageTextToText: {exc}")
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            return AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            errors.append(f"AutoModelForVision2Seq: {exc}")
        raise RuntimeError("Could not load VLM model. " + " | ".join(errors))

    def _image_uri(self, path: Path) -> str:
        try:
            return path.resolve().as_uri()
        except ValueError:
            return path.resolve().as_posix()

    def generate(self, system_prompt: str, user_prompt: str, image_path: Path, _row: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self._image_uri(image_path)},
                    {"type": "text", "text": user_prompt},
                ],
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

        processor_kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "images": images_input,
            "padding": True,
            "return_tensors": "pt",
        }
        if videos_input is not None:
            processor_kwargs["videos"] = videos_input
        inputs = self.processor(**processor_kwargs)
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated)]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Manifest is empty.")
    leaked = sorted(set().union(*(set(r.keys()) for r in rows)) & LEAKAGE_FORBIDDEN_FIELDS)
    if leaked:
        raise ValueError(f"Manifest contains leakage-forbidden fields: {leaked}")
    missing_required = sorted(REQUIRED_MANIFEST_FIELDS - set(rows[0].keys()))
    if missing_required:
        raise ValueError(f"Manifest is missing required fields: {missing_required}")


def resolve_image_path(raw: str, base_dir: Path) -> Path:
    normalized = str(raw).replace("\\", "/")
    path = Path(normalized)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                base_dir / path,
                Path.cwd() / path,
                Path.cwd() / "images" / path.name,
                Path.cwd().parent / "images" / path.name,
                Path.cwd() / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(c) for c in candidates[:10])
    raise FileNotFoundError(f"Could not resolve image path {raw!r}. Tried:\n{tried}")


def select_rows(rows: list[dict[str, Any]], split: str, limit: int | None, seed: int) -> list[dict[str, Any]]:
    if split != "all":
        rows = [r for r in rows if str(r.get("split", "")) == split]
    rows = list(rows)
    # Keep manifest order by default; seed is reserved for reproducible future sampling.
    _ = random.Random(seed)
    if limit is not None:
        rows = rows[:limit]
    return rows


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest_jsonl)
    rows = read_jsonl(manifest_path)
    validate_manifest_rows(rows)
    selected_rows = select_rows(rows, args.split, args.limit, args.seed)
    if not selected_rows:
        raise ValueError(f"No rows selected for split={args.split!r} limit={args.limit!r}.")

    if args.backend == "mock":
        backend: Any = MockBackend(args.seed)
    else:
        backend = QwenHFBackend(args.model_name, args.max_new_tokens, args.max_pixels)

    out_rows: list[dict[str, Any]] = []
    started = time.time()
    for i, row in enumerate(selected_rows, start=1):
        image_path = resolve_image_path(str(row["resolved_image_path"]), manifest_path.parent.parent.parent)
        user_prompt = build_user_prompt(row, args.prompt_style)
        try:
            raw = backend.generate(SYSTEM_PROMPT, user_prompt, image_path, row)
            parsed, parse_error = parse_json_object(raw)
            out = normalize_output(parsed, raw, row, parse_error)
        except Exception as exc:
            out = normalize_output(None, "", row, f"inference_error: {type(exc).__name__}: {exc}")
        out_rows.append(out)
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == len(selected_rows)):
            elapsed = time.time() - started
            print(f"[{i}/{len(selected_rows)}] elapsed={elapsed:.1f}s record_id={row.get('record_id')}", flush=True)

    write_jsonl(Path(args.out_jsonl), out_rows)
    summary_path = Path(args.out_jsonl).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "manifest_jsonl": str(manifest_path),
                "out_jsonl": str(args.out_jsonl),
                "backend": args.backend,
                "model_name": args.model_name,
                "split": args.split,
                "limit": args.limit,
                "n_rows": len(out_rows),
                "parse_ok": sum(1 for r in out_rows if r.get("parse_ok") is True),
                "prompt_style": args.prompt_style,
                "elapsed_sec": round(time.time() - started, 3),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {args.out_jsonl}")
    print(f"Wrote: {summary_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--backend", choices=["qwen_hf", "mock"], default="qwen_hf")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument(
        "--prompt-style",
        choices=["conservative", "evidence_compare", "contrastive_decisive", "contrastive_decisive_noscore"],
        default="conservative",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
