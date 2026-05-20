#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


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


class QwenHFBackend:
    def __init__(self, model_name: str, max_new_tokens: int = 220, max_pixels: int | None = 401408) -> None:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore

        self.torch = torch
        self.force_cpu = str(__import__("os").environ.get("VLM_FORCE_CPU", "0")).lower() in {"1", "true", "yes"}
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            max_pixels=max_pixels,
        )
        self.model = self._load_model(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            self.process_vision_info = process_vision_info
        except Exception:
            self.process_vision_info = None

    def _load_model(self, model_name: str) -> Any:
        use_cpu = self.force_cpu or (not self.torch.cuda.is_available())
        dtype = getattr(self.torch, "float32", "auto") if use_cpu else getattr(self.torch, "float16", "auto")
        device_map: str = "cpu" if use_cpu else "auto"
        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "attn_implementation": "eager",
        }
        errors: list[str] = []
        loaders = []
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
            loaders.append(("Qwen2_5_VLForConditionalGeneration", Qwen2_5_VLForConditionalGeneration))
        except Exception:
            pass
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore
            loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except Exception:
            pass
        for name, cls in loaders:
            try:
                return cls.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                # Retry on CPU if CUDA runtime/kernel mismatch happened.
                msg = str(exc).lower()
                if ("no kernel image is available" in msg or "cuda error" in msg or "sm_" in msg) and kwargs.get("device_map") != "cpu":
                    cpu_kwargs = dict(kwargs)
                    cpu_kwargs["device_map"] = "cpu"
                    cpu_kwargs["torch_dtype"] = getattr(self.torch, "float32", "auto")
                    try:
                        return cls.from_pretrained(model_name, **cpu_kwargs)
                    except Exception as exc2:
                        errors.append(f"{name} (cpu_retry): {exc2}")
        raise RuntimeError("Could not load model: " + " | ".join(errors))

    def _uri(self, p: Path) -> str:
        return p.resolve().as_uri()

    def generate(self, system_prompt: str, user_prompt: str, image_paths: list[Path]) -> str:
        content = []
        for p in image_paths:
            content.append({"type": "image", "image": self._uri(p)})
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.process_vision_info is not None:
            image_inputs, videos_inputs = self.process_vision_info(messages)
        else:
            from PIL import Image  # type: ignore

            image_inputs = [Image.open(p).convert("RGB") for p in image_paths]
            videos_inputs = None
        kwargs: dict[str, Any] = {"text": [prompt_text], "images": image_inputs, "padding": True, "return_tensors": "pt"}
        if videos_inputs is not None:
            kwargs["videos"] = videos_inputs
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
