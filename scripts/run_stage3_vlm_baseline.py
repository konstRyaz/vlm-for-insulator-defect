#!/usr/bin/env python3
from __future__ import annotations

"""
Run first Stage 3 baseline:
GT crop dataset -> prompt -> VLM backend -> parsed subset -> vlm_labels_v1 mapping.

This script intentionally keeps Stage 3 minimal:
- no detector integration;
- no report generation layer;
- no model training/fine-tuning.
"""

import argparse
import base64
import uuid
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_vlm_labels_v1 import validate_record  # noqa: E402


COARSE_CLASS_VALUES = {"insulator_ok", "defect_flashover", "defect_broken", "unknown", "other"}
VISIBILITY_VALUES = {"clear", "partial", "ambiguous"}

MODEL_PREDICTED_CORE_FIELDS = [
    "coarse_class",
    "visual_evidence_tags",
    "visibility",
    "short_canonical_description_en",
    "report_snippet_en",
]
MODEL_OPTIONAL_DEBUG_FIELDS = [
    "annotator_notes",
]
PIPELINE_DERIVED_FIELDS = [
    "needs_review = (visibility == 'ambiguous')",
    "short_canonical_description = short_canonical_description_en",
    "report_snippet = report_snippet_en",
]
PIPELINE_COPIED_FIELDS = [
    "record_id",
    "image_id",
    "box_id",
    "source",
    "split",
    "bbox_xywh",
    "crop_path",
    "image_path",
    "score",
    "category_name",
    "label_version",
]


class BackendError(RuntimeError):
    pass


class BackendConfigError(BackendError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 GT-crop VLM baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline/stage3_vlm_gt_baseline.yaml",
        help="Path to baseline YAML config.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Override input dataset JSONL path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run_id.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root directory.",
    )
    parser.add_argument(
        "--backend-mode",
        type=str,
        default=None,
        choices=["mock", "qwen_hf"],
        help="Override backend mode from config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for records processed in this run.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="Optional text file with record_id per line to run a targeted subset.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Override prompt version key from config prompt.versions.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first backend/parse/validation error.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Force resume mode on.",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Force resume mode off.",
    )
    return parser.parse_args()


def now_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:8]}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(payload).__name__}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(payload).__name__}")
            rows.append(payload)
    return rows


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    rows = load_jsonl(path)
    for row in rows:
        rid = row.get("record_id")
        if isinstance(rid, str) and rid.strip():
            processed.add(rid)
    return processed


def load_record_ids_file(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            ids.add(text)
    if not ids:
        raise ValueError(f"No record_ids found in sample ids file: {path}")
    return ids


def render_template(text: str, values: dict[str, Any]) -> str:
    rendered = text
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def resolve_crop_path(record: dict[str, Any], dataset_path: Path, image_root: Path | None) -> Path:
    crop_path_value = record.get("crop_path")
    if not isinstance(crop_path_value, str) or not crop_path_value.strip():
        raise FileNotFoundError("Missing crop_path in record.")

    crop_path = Path(crop_path_value)
    if crop_path.is_absolute() and crop_path.exists():
        return crop_path

    candidates = [(dataset_path.parent / crop_path).resolve()]
    if image_root is not None:
        candidates.append((image_root / crop_path).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve crop_path: {crop_path_value}")


def extract_text_from_chat_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


class MockVLMBackend:
    provider_name = "mock"
    backend_name = "mock"
    model_name = "mock-v1"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        coarse = str(record.get("coarse_class", "unknown")).strip() or "unknown"
        if coarse not in COARSE_CLASS_VALUES:
            coarse = "unknown"

        visibility = str(record.get("visibility", "ambiguous")).strip() or "ambiguous"
        if visibility not in VISIBILITY_VALUES:
            visibility = "ambiguous"

        raw_tags = record.get("visual_evidence_tags", [])
        tags = [str(tag).strip() for tag in raw_tags] if isinstance(raw_tags, list) else []
        tags = [tag for tag in tags if tag][:6]

        desc = str(
            record.get("short_canonical_description_en")
            or record.get("short_canonical_description")
            or "Insulator crop with uncertain visual evidence."
        ).strip()
        if not desc:
            desc = "Insulator crop with uncertain visual evidence."

        snippet = str(
            record.get("report_snippet_en")
            or record.get("report_snippet")
            or "The crop shows uncertain visual evidence and should be reviewed."
        ).strip()
        if not snippet:
            snippet = "The crop shows uncertain visual evidence and should be reviewed."

        mock_obj = {
            "coarse_class": coarse,
            "visual_evidence_tags": tags,
            "visibility": visibility,
            "short_canonical_description_en": desc,
            "report_snippet_en": snippet,
        }
        return {
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "raw_text": json.dumps(mock_obj, ensure_ascii=False),
            "raw_payload": {"mode": "mock", "generated": mock_obj},
        }

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "backend_mode": self.backend_name,
            "model": self.model_name,
            "temperature": None,
            "max_tokens": None,
            "response_format": None,
            "timeout_sec": None,
            "api_base": None,
        }


class QwenHFBackend:
    backend_name = "qwen_hf"

    def __init__(self, cfg: dict[str, Any]) -> None:
        provider_value = str(cfg.get("provider", "huggingface_transformers")).strip().lower()
        if provider_value not in {"huggingface_transformers", "hf_transformers"}:
            raise BackendConfigError(
                "backend.qwen_hf.provider must be 'huggingface_transformers' or 'hf_transformers'."
            )
        self.provider_name = provider_value

        model_id_value = cfg.get("model_id")
        if not isinstance(model_id_value, str) or not model_id_value.strip():
            raise BackendConfigError("backend.qwen_hf.model_id must be a non-empty string.")
        self.model_id = model_id_value.strip()
        self.model_name = self.model_id

        self.trust_remote_code = bool(cfg.get("trust_remote_code", True))
        self.torch_dtype_name = str(cfg.get("torch_dtype", "auto")).strip().lower()
        self.device_map = cfg.get("device_map", "auto")
        self.attn_implementation = cfg.get("attn_implementation", None)
        self.max_new_tokens = int(cfg.get("max_new_tokens", 220))
        if self.max_new_tokens <= 0:
            raise BackendConfigError("backend.qwen_hf.max_new_tokens must be positive.")
        self.do_sample = bool(cfg.get("do_sample", False))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 1.0))
        self.add_generation_prompt = bool(cfg.get("add_generation_prompt", True))
        self.min_pixels = self._optional_positive_int(cfg.get("min_pixels"), "backend.qwen_hf.min_pixels")
        self.max_pixels = self._optional_positive_int(cfg.get("max_pixels"), "backend.qwen_hf.max_pixels")

        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
            from transformers import AutoProcessor  # type: ignore
        except ImportError as exc:
            raise BackendConfigError(
                "Qwen backend requires extra dependencies. Install/upgrade: "
                "transformers, accelerate, and Pillow in Kaggle/Colab."
            ) from exc

        self.torch = torch
        self.Image = Image
        self.AutoProcessor = AutoProcessor
        self._torch_dtype = self._resolve_torch_dtype(self.torch_dtype_name)
        self._process_vision_info = None
        self._uses_qwen_vl_utils = False
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            self._process_vision_info = process_vision_info
            self._uses_qwen_vl_utils = True
        except ImportError:
            self._process_vision_info = None
            self._uses_qwen_vl_utils = False

        processor_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = self.AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
        self.model = self._load_model()
        try:
            self.model.eval()
        except Exception:
            pass

    def _optional_positive_int(self, value: Any, field_name: str) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise BackendConfigError(f"{field_name} must be a positive integer if set.") from exc
        if parsed <= 0:
            raise BackendConfigError(f"{field_name} must be a positive integer if set.")
        return parsed

    def _resolve_torch_dtype(self, dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        mapping = {
            "float16": self.torch.float16,
            "fp16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "bf16": self.torch.bfloat16,
            "float32": self.torch.float32,
            "fp32": self.torch.float32,
        }
        if dtype_name not in mapping:
            raise BackendConfigError(
                f"Unsupported backend.qwen_hf.torch_dtype='{dtype_name}'. "
                "Use one of: auto, float16, bfloat16, float32."
            )
        return mapping[dtype_name]

    def _load_model(self) -> Any:
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self._torch_dtype,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if isinstance(self.attn_implementation, str) and self.attn_implementation.strip():
            model_kwargs["attn_implementation"] = self.attn_implementation.strip()

        load_errors: list[str] = []

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as exc:
            load_errors.append(f"Qwen2_5_VLForConditionalGeneration: {exc}")

        try:
            from transformers import AutoModelForImageTextToText  # type: ignore

            return AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as exc:
            load_errors.append(f"AutoModelForImageTextToText: {exc}")

        try:
            from transformers import AutoModelForVision2Seq  # type: ignore

            return AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as exc:
            load_errors.append(f"AutoModelForVision2Seq: {exc}")

        raise BackendConfigError(
            "Could not load Qwen model. Ensure compatible versions of transformers/accelerate "
            f"and access to model '{self.model_id}'. Details: {' | '.join(load_errors)}"
        )

    def _move_inputs_to_model_device(self, inputs: Any) -> Any:
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            return inputs
        try:
            return inputs.to(model_device)
        except Exception:
            return inputs

    def _build_messages(self, system_prompt: str, user_prompt: str, image_path: Path) -> list[dict[str, Any]]:
        resolved_image_path = image_path.resolve()
        try:
            # Prefer file:// URI for qwen_vl_utils multimodal parsing in notebook runtimes.
            image_uri = resolved_image_path.as_uri()
        except ValueError:
            # Fallback for edge cases where URI conversion is not available.
            image_uri = resolved_image_path.as_posix()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return messages

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
        )
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )

        images_input: Any = None
        videos_input: Any = None
        used_qwen_vl_utils = False

        if self._process_vision_info is not None:
            try:
                images_input, videos_input = self._process_vision_info(messages)
                used_qwen_vl_utils = True
            except Exception:
                images_input = None
                videos_input = None
                used_qwen_vl_utils = False

        if not used_qwen_vl_utils:
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
        inputs = self._move_inputs_to_model_device(inputs)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids", None)
        if input_ids is not None:
            trimmed_ids = []
            for in_ids, out_ids in zip(input_ids, generated_ids):
                trimmed_ids.append(out_ids[len(in_ids) :])
            decoded = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        raw_text = decoded[0] if decoded else ""
        return {
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "raw_text": raw_text,
            "raw_payload": {
                "provider": self.provider_name,
                "model_id": self.model_id,
                "generation_kwargs": generation_kwargs,
                "used_qwen_vl_utils": used_qwen_vl_utils,
            },
        }

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "backend_mode": self.backend_name,
            "model": self.model_name,
            "model_id": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype_name,
            "device_map": self.device_map,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "add_generation_prompt": self.add_generation_prompt,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "uses_qwen_vl_utils": self._uses_qwen_vl_utils,
        }


def extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, n):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def parse_response_text(raw_text: str) -> tuple[str, str | None, dict[str, Any] | None]:
    if not raw_text.strip():
        return "error", "empty_response", None

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return "success", None, payload
    except json.JSONDecodeError:
        pass

    candidates = extract_json_candidates(raw_text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return "success", None, payload
    return "error", "no_json_object_found", None


def normalize_prediction(pred_obj: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    normalized: dict[str, Any] = {}

    coarse_class = pred_obj.get("coarse_class")
    if not isinstance(coarse_class, str) or not coarse_class.strip():
        errors.append("missing_or_invalid:coarse_class")
    else:
        coarse_class_value = coarse_class.strip()
        if coarse_class_value not in COARSE_CLASS_VALUES:
            errors.append("out_of_range:coarse_class")
        normalized["coarse_class"] = coarse_class_value

    tags = pred_obj.get("visual_evidence_tags")
    if not isinstance(tags, list):
        errors.append("missing_or_invalid:visual_evidence_tags")
    else:
        clean_tags: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str):
                errors.append("invalid_item_type:visual_evidence_tags")
                continue
            tag_value = tag.strip()
            if not tag_value:
                continue
            if tag_value in seen:
                continue
            seen.add(tag_value)
            clean_tags.append(tag_value)
        normalized["visual_evidence_tags"] = clean_tags

    visibility = pred_obj.get("visibility")
    if not isinstance(visibility, str) or not visibility.strip():
        errors.append("missing_or_invalid:visibility")
    else:
        visibility_value = visibility.strip()
        if visibility_value not in VISIBILITY_VALUES:
            errors.append("out_of_range:visibility")
        normalized["visibility"] = visibility_value

    short_en = pred_obj.get("short_canonical_description_en")
    if not isinstance(short_en, str) or not short_en.strip():
        errors.append("missing_or_invalid:short_canonical_description_en")
    else:
        normalized["short_canonical_description_en"] = short_en.strip()

    snippet_en = pred_obj.get("report_snippet_en")
    if not isinstance(snippet_en, str) or not snippet_en.strip():
        errors.append("missing_or_invalid:report_snippet_en")
    else:
        normalized["report_snippet_en"] = snippet_en.strip()

    if "annotator_notes" in pred_obj:
        notes = pred_obj.get("annotator_notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            notes = str(notes)
        notes_value = notes.strip()
        if notes_value:
            normalized["annotator_notes"] = notes_value

    if errors:
        return None, errors
    return normalized, []


def map_subset_to_vlm_labels_v1(input_record: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for field in PIPELINE_COPIED_FIELDS:
        if field in input_record:
            output[field] = input_record[field]

    if "label_version" not in output or output["label_version"] is None:
        output["label_version"] = "vlm_labels_v1"

    output["coarse_class"] = pred["coarse_class"]
    output["visual_evidence_tags"] = pred["visual_evidence_tags"]
    output["visibility"] = pred["visibility"]
    output["needs_review"] = pred["visibility"] == "ambiguous"

    output["short_canonical_description_en"] = pred["short_canonical_description_en"]
    output["report_snippet_en"] = pred["report_snippet_en"]
    output["short_canonical_description"] = pred["short_canonical_description_en"]
    output["report_snippet"] = pred["report_snippet_en"]
    output["annotator_notes"] = pred.get("annotator_notes", "")
    return output


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg_copy = dict(cfg)
    cfg_copy.setdefault("input", {})
    cfg_copy.setdefault("output", {})
    cfg_copy.setdefault("run", {})
    cfg_copy.setdefault("backend", {})
    cfg_copy.setdefault("prompt", {})

    if args.input_jsonl:
        cfg_copy["input"]["dataset_jsonl"] = args.input_jsonl
    if args.run_id:
        cfg_copy["run"]["run_id"] = args.run_id
    if args.output_root:
        cfg_copy["output"]["root_dir"] = args.output_root
    if args.backend_mode:
        cfg_copy["backend"]["mode"] = args.backend_mode
    if args.max_samples is not None:
        cfg_copy["run"]["max_samples"] = args.max_samples
    if args.sample_ids_file:
        cfg_copy["run"]["sample_ids_file"] = args.sample_ids_file
    if args.prompt_version:
        cfg_copy["prompt"]["version"] = args.prompt_version
    if args.fail_fast:
        cfg_copy["run"]["fail_fast"] = True
    if args.resume:
        cfg_copy["run"]["resume"] = True
    if args.no_resume:
        cfg_copy["run"]["resume"] = False
    return cfg_copy


def resolve_prompt_paths(prompt_cfg: dict[str, Any]) -> tuple[str | None, Path, Path]:
    selected_version: str | None = None
    versions = prompt_cfg.get("versions")
    requested_version = prompt_cfg.get("version")
    if isinstance(requested_version, str):
        requested_version = requested_version.strip()
    else:
        requested_version = None

    if requested_version:
        if not isinstance(versions, dict):
            raise ValueError("prompt.version is set, but prompt.versions is not an object.")
        version_cfg = versions.get(requested_version)
        if not isinstance(version_cfg, dict):
            raise ValueError(f"prompt version '{requested_version}' is not found in prompt.versions.")
        system_raw = version_cfg.get("system_path", "")
        user_raw = version_cfg.get("user_path", "")
        selected_version = requested_version
    else:
        system_raw = prompt_cfg.get("system_path", "")
        user_raw = prompt_cfg.get("user_path", "")

    system_prompt_path = Path(str(system_raw)).resolve()
    user_prompt_path = Path(str(user_raw)).resolve()
    return selected_version, system_prompt_path, user_prompt_path


def select_backend(cfg: dict[str, Any]) -> tuple[Any, list[str]]:
    backend_cfg = cfg.get("backend", {})
    mode = str(backend_cfg.get("mode", "mock")).strip().lower()
    fallback = bool(backend_cfg.get("fallback_to_mock_if_unavailable", True))
    notices: list[str] = []

    if mode == "mock":
        return MockVLMBackend(), notices

    if mode == "qwen_hf":
        try:
            qwen_cfg = backend_cfg.get("qwen_hf", {})
            if not isinstance(qwen_cfg, dict):
                raise BackendConfigError("backend.qwen_hf config must be an object.")
            return QwenHFBackend(qwen_cfg), notices
        except BackendConfigError:
            raise
        except BackendError as exc:
            if fallback:
                notices.append(f"qwen_hf_unavailable_fallback_to_mock: {exc}")
                return MockVLMBackend(), notices
            raise

    raise ValueError(f"Unsupported backend mode: {mode}")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_yaml(cfg_path)
    cfg = apply_cli_overrides(cfg, args)

    dataset_path = Path(cfg.get("input", {}).get("dataset_jsonl", "")).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset JSONL not found: {dataset_path}")

    image_root_raw = cfg.get("input", {}).get("image_root")
    image_root = Path(image_root_raw).resolve() if isinstance(image_root_raw, str) and image_root_raw.strip() else None

    run_cfg = cfg.get("run", {})
    run_id_cfg = str(run_cfg.get("run_id", "auto")).strip()
    run_id = now_run_id() if run_id_cfg in {"", "auto"} else run_id_cfg

    output_root = Path(cfg.get("output", {}).get("root_dir", "outputs/stage3_vlm_baseline_runs")).resolve()
    run_dir = output_root / run_id
    resume_mode = bool(run_cfg.get("resume", True))
    fail_fast = bool(run_cfg.get("fail_fast", False))
    max_samples = run_cfg.get("max_samples", None)
    sample_ids_file_raw = run_cfg.get("sample_ids_file", None)
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

    sample_ids_path: Path | None = None
    sample_ids_set: set[str] | None = None
    if isinstance(sample_ids_file_raw, str) and sample_ids_file_raw.strip():
        sample_ids_path = Path(sample_ids_file_raw).resolve()
        if not sample_ids_path.exists():
            raise FileNotFoundError(f"sample_ids_file not found: {sample_ids_path}")
        sample_ids_set = load_record_ids_file(sample_ids_path)

    if run_dir.exists() and not resume_mode:
        raise FileExistsError(f"Run dir already exists and resume is disabled: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    backend, backend_notices = select_backend(cfg)
    backend_mode_effective = backend.backend_name
    backend_settings_effective = backend.describe() if hasattr(backend, "describe") else {
        "provider": getattr(backend, "provider_name", "unknown"),
        "backend_mode": backend_mode_effective,
        "model": getattr(backend, "model_name", "unknown"),
    }
    prediction_contract = {
        "mode": str(cfg.get("prediction_contract", {}).get("mode", "reduced_subset_v1")),
        "model_predicted_core_fields": list(MODEL_PREDICTED_CORE_FIELDS),
        "model_optional_debug_fields": list(MODEL_OPTIONAL_DEBUG_FIELDS),
        "pipeline_derived_fields": list(PIPELINE_DERIVED_FIELDS),
        "pipeline_copied_fields": list(PIPELINE_COPIED_FIELDS),
    }

    prompt_cfg = cfg.get("prompt", {})
    if not isinstance(prompt_cfg, dict):
        raise ValueError("prompt config must be an object.")
    selected_prompt_version, system_prompt_path, user_prompt_path = resolve_prompt_paths(prompt_cfg)
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    if not user_prompt_path.exists():
        raise FileNotFoundError(f"User prompt file not found: {user_prompt_path}")

    system_prompt = read_text(system_prompt_path)
    user_prompt_template = read_text(user_prompt_path)
    user_prompt_contains_crop_path_token = "{{crop_path}}" in user_prompt_template or "crop_path" in user_prompt_template

    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts" / "system.txt").write_text(system_prompt, encoding="utf-8")
    (run_dir / "prompts" / "user_template.txt").write_text(user_prompt_template, encoding="utf-8")

    config_snapshot = {
        "config_path": str(cfg_path),
        "resolved_run_id": run_id,
        "backend_mode_effective": backend_mode_effective,
        "backend_settings_effective": backend_settings_effective,
        "backend_notices": backend_notices,
        "prompt_selection": {
            "selected_version": selected_prompt_version,
            "system_prompt_path": str(system_prompt_path),
            "user_prompt_path": str(user_prompt_path),
            "user_prompt_contains_crop_path_token": user_prompt_contains_crop_path_token,
        },
        "prediction_contract": prediction_contract,
        "config": cfg,
    }
    write_json(run_dir / "config_snapshot.json", config_snapshot)

    records = load_jsonl(dataset_path)
    records_total_loaded = len(records)
    record_ids_loaded = {str(r.get("record_id", "")).strip() for r in records if isinstance(r.get("record_id"), str)}
    sample_ids_unmatched_count: int | None = None
    if sample_ids_set is not None:
        sample_ids_unmatched_count = len(sample_ids_set.difference(record_ids_loaded))
        records = [r for r in records if str(r.get("record_id", "")).strip() in sample_ids_set]
    records_after_sample_ids = len(records)
    if max_samples is not None:
        records = records[:max_samples]
    records_after_max_samples = len(records)

    sample_results_path = run_dir / "sample_results.jsonl"
    raw_responses_path = run_dir / "raw_responses.jsonl"
    parsed_predictions_path = run_dir / "parsed_predictions.jsonl"
    predictions_vlm_labels_v1_path = run_dir / "predictions_vlm_labels_v1.jsonl"
    failures_path = run_dir / "failures.jsonl"

    # Keep the run artifact set stable even when the selected dataset is empty.
    for path in [
        sample_results_path,
        raw_responses_path,
        parsed_predictions_path,
        predictions_vlm_labels_v1_path,
        failures_path,
    ]:
        path.touch(exist_ok=True)

    processed_ids = load_processed_ids(sample_results_path) if resume_mode else set()

    counters = {
        "records_total_loaded": records_total_loaded,
        "records_after_sample_ids_filter": records_after_sample_ids,
        "records_after_max_samples": records_after_max_samples,
        "records_skipped_resume": 0,
        "records_attempted": 0,
        "status_ok": 0,
        "status_backend_error": 0,
        "status_parse_error": 0,
        "status_normalization_error": 0,
        "status_validation_error": 0,
    }

    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            counters["status_validation_error"] += 1
            failure = {
                "record_id": "",
                "status": "validation_error",
                "error": "missing_record_id",
            }
            append_jsonl(failures_path, failure)
            if fail_fast:
                break
            continue

        if record_id in processed_ids:
            counters["records_skipped_resume"] += 1
            continue

        counters["records_attempted"] += 1
        status = "ok"
        parse_status = "not_attempted"
        parse_error: str | None = None
        normalization_errors: list[str] = []
        schema_valid = False
        schema_errors: list[str] = []
        mapped_record: dict[str, Any] | None = None
        raw_text = ""
        raw_payload: dict[str, Any] | None = None
        prompt_rendered = ""
        crop_path_resolved = ""
        error_message: str | None = None

        try:
            crop_path = resolve_crop_path(record=record, dataset_path=dataset_path, image_root=image_root)
            crop_path_resolved = str(crop_path)
            prompt_values = {
                "record_id": record_id,
                "split": record.get("split", ""),
                "source": record.get("source", ""),
                "bbox_xywh": record.get("bbox_xywh", ""),
                "crop_path": record.get("crop_path", ""),
            }
            prompt_rendered = render_template(user_prompt_template, prompt_values)

            backend_out = backend.generate(
                system_prompt=system_prompt,
                user_prompt=prompt_rendered,
                image_path=crop_path,
                record=record,
            )
            raw_text = str(backend_out.get("raw_text", "") or "")
            payload_obj = backend_out.get("raw_payload")
            if isinstance(payload_obj, dict):
                raw_payload = payload_obj
            else:
                raw_payload = {"value": payload_obj}

            parse_status, parse_error, parsed_obj = parse_response_text(raw_text)
            parsed_record = {
                "record_id": record_id,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "parsed_object": parsed_obj,
            }

            if parse_status != "success" or parsed_obj is None:
                status = "parse_error"
            else:
                normalized_pred, normalization_errors = normalize_prediction(parsed_obj)
                parsed_record["normalized_prediction"] = normalized_pred
                parsed_record["normalization_errors"] = normalization_errors
                if normalization_errors or normalized_pred is None:
                    status = "normalization_error"
                else:
                    mapped_record = map_subset_to_vlm_labels_v1(record, normalized_pred)
                    schema_errors = validate_record(mapped_record, line_no=1)
                    schema_valid = len(schema_errors) == 0
                    if not schema_valid:
                        status = "validation_error"
                    else:
                        append_jsonl(predictions_vlm_labels_v1_path, mapped_record)

            append_jsonl(parsed_predictions_path, parsed_record)

        except Exception as exc:
            status = "backend_error"
            error_message = str(exc)

        raw_entry = {
            "record_id": record_id,
            "backend_mode_effective": backend_mode_effective,
            "provider_name": backend_settings_effective.get("provider"),
            "model_name": getattr(backend, "model_name", "unknown"),
            "crop_path_resolved": crop_path_resolved,
            "system_prompt_path": str(system_prompt_path),
            "user_prompt_path": str(user_prompt_path),
            "user_prompt_rendered": prompt_rendered,
            "raw_text": raw_text,
            "raw_payload": raw_payload if bool(cfg.get("output", {}).get("save_raw_payload", True)) else None,
            "backend_error": error_message,
        }
        append_jsonl(raw_responses_path, raw_entry)

        sample_result = {
            "record_id": record_id,
            "status": status,
            "backend_mode_effective": backend_mode_effective,
            "provider_name": backend_settings_effective.get("provider"),
            "model_name": getattr(backend, "model_name", "unknown"),
            "crop_path": record.get("crop_path"),
            "crop_path_resolved": crop_path_resolved,
            "parse_status": parse_status,
            "parse_error": parse_error,
            "normalization_errors": normalization_errors,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "backend_error": error_message,
        }
        append_jsonl(sample_results_path, sample_result)

        if status != "ok":
            failure = {
                "record_id": record_id,
                "status": status,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "normalization_errors": normalization_errors,
                "schema_errors": schema_errors,
                "backend_error": error_message,
                "raw_text_preview": raw_text[:500],
            }
            append_jsonl(failures_path, failure)

        if status == "ok":
            counters["status_ok"] += 1
        elif status == "backend_error":
            counters["status_backend_error"] += 1
        elif status == "parse_error":
            counters["status_parse_error"] += 1
        elif status == "normalization_error":
            counters["status_normalization_error"] += 1
        elif status == "validation_error":
            counters["status_validation_error"] += 1

        if fail_fast and status != "ok":
            break

    run_summary = {
        "run_id": run_id,
        "dataset_jsonl": str(dataset_path),
        "backend_mode_effective": backend_mode_effective,
        "backend_settings_effective": backend_settings_effective,
        "backend_notices": backend_notices,
        "prompt_selection": {
            "selected_version": selected_prompt_version,
            "system_prompt_path": str(system_prompt_path),
            "user_prompt_path": str(user_prompt_path),
            "user_prompt_contains_crop_path_token": user_prompt_contains_crop_path_token,
        },
        "prediction_contract": prediction_contract,
        "input_selection": {
            "sample_ids_file": str(sample_ids_path) if sample_ids_path is not None else None,
            "sample_ids_count": len(sample_ids_set) if sample_ids_set is not None else None,
            "sample_ids_unmatched_count": sample_ids_unmatched_count,
            "max_samples": max_samples,
            "ordering": "dataset_jsonl_order",
        },
        "counters": counters,
        "artifacts": {
            "sample_results_jsonl": str(sample_results_path),
            "raw_responses_jsonl": str(raw_responses_path),
            "parsed_predictions_jsonl": str(parsed_predictions_path),
            "predictions_vlm_labels_v1_jsonl": str(predictions_vlm_labels_v1_path),
            "failures_jsonl": str(failures_path),
            "config_snapshot_json": str(run_dir / "config_snapshot.json"),
            "prompt_system_copy": str(run_dir / "prompts" / "system.txt"),
            "prompt_user_template_copy": str(run_dir / "prompts" / "user_template.txt"),
        },
    }
    write_json(run_dir / "run_summary.json", run_summary)

    print(f"Run dir: {run_dir}")
    print(f"Backend mode (effective): {backend_mode_effective}")
    print(f"Provider/model: {backend_settings_effective.get('provider')} / {backend_settings_effective.get('model')}")
    print(f"Processed: {counters['records_attempted']} | OK: {counters['status_ok']} | Failures: {counters['records_attempted'] - counters['status_ok']}")
    print(f"Summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
