from __future__ import annotations

import json
import uuid
from pathlib import Path

import torch

ROOT_PATH = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: Path | str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path | str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def now_timestamp() -> str:
    return uuid.uuid4().hex[:12]
