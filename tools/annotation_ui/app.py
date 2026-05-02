#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_file


EDITABLE_FIELDS = {
    "visual_evidence_tags",
    "visibility",
    "short_canonical_description_en",
    "report_snippet_en",
    "annotator_notes",
}

VISIBILITY_VALUES = {"clear", "partial", "ambiguous"}

DEFAULT_TAG_OPTIONS = [
    "intact_structure",
    "regular_disc_shape",
    "no_visible_break",
    "no_visible_burn_mark",
    "clean_surface",
    "uniform_appearance",
    "dark_surface_trace",
    "burn_like_mark",
    "surface_stain",
    "localized_darkening",
    "flashover_like_trace",
    "surface_damage_mark",
    "missing_fragment",
    "edge_discontinuity",
    "broken_profile",
    "structural_gap",
    "shape_irregularity",
    "material_loss",
    "low_contrast",
    "blurred_region",
    "partial_view",
    "occluded_region",
    "unclear_boundary",
    "ambiguous_evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal local UI annotator for vlm_labels_v1 JSONL files.")
    parser.add_argument("--input", required=True, type=str, help="Path to source JSONL file.")
    parser.add_argument("--host", default="127.0.0.1", type=str, help="Host (default: 127.0.0.1).")
    parser.add_argument("--port", default=8501, type=int, help="Port (default: 8501).")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_no}, got {type(record).__name__}")
            records.append(record)
    return records


def ensure_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def normalize_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        tags = [ensure_string(v).strip() for v in value]
    elif isinstance(value, str):
        tags = [part.strip() for part in value.split(",")]
    else:
        tags = []

    unique: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        unique.append(tag)
    return unique


def is_completed(record: dict[str, Any]) -> bool:
    visibility = ensure_string(record.get("visibility")).strip()
    tags = normalize_tags(record.get("visual_evidence_tags"))

    text_fields = [
        ensure_string(record.get("short_canonical_description_en")).strip(),
        ensure_string(record.get("report_snippet_en")).strip(),
    ]
    return visibility in VISIBILITY_VALUES and len(tags) > 0 and all(text_fields)


class AnnotationStore:
    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.resolve()
        annotated_suffix = f".annotated{self.input_path.suffix}"
        if self.input_path.name.endswith(annotated_suffix):
            self.sidecar_path = self.input_path
        else:
            self.sidecar_path = self.input_path.with_name(f"{self.input_path.stem}.annotated{self.input_path.suffix}")
        self.base_dir = self.input_path.parent.resolve()
        self._lock = threading.Lock()

        self.records: list[dict[str, Any]] = []
        self.record_id_to_index: dict[str, int] = {}
        self._loaded_version: tuple[int | None, int | None] = (None, None)
        self._reload_from_disk(force=True)

    def _path_mtime_ns(self, path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def _compute_version(self) -> tuple[int | None, int | None]:
        return (
            self._path_mtime_ns(self.input_path),
            self._path_mtime_ns(self.sidecar_path),
        )

    def _reload_from_disk(self, force: bool = False) -> None:
        current_version = self._compute_version()
        if not force and current_version == self._loaded_version:
            return

        records = self._load_records()
        self.records = [self._normalize_record(rec) for rec in records]
        self.record_id_to_index = {
            ensure_string(rec.get("record_id")).strip(): idx for idx, rec in enumerate(self.records)
        }
        self._loaded_version = current_version

    def refresh_if_stale(self) -> None:
        with self._lock:
            self._reload_from_disk(force=False)

    def _load_records(self) -> list[dict[str, Any]]:
        source_records = load_jsonl(self.input_path)
        if not self.sidecar_path.exists():
            return source_records

        sidecar_records = load_jsonl(self.sidecar_path)
        sidecar_by_id: dict[str, dict[str, Any]] = {}
        for rec in sidecar_records:
            rid = ensure_string(rec.get("record_id")).strip()
            if rid:
                sidecar_by_id[rid] = rec

        merged: list[dict[str, Any]] = []
        for rec in source_records:
            rid = ensure_string(rec.get("record_id")).strip()
            if rid and rid in sidecar_by_id:
                merged.append(sidecar_by_id[rid])
            else:
                merged.append(rec)
        return merged

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        rec = dict(record)
        rec.setdefault("visual_evidence_tags", [])
        rec["visual_evidence_tags"] = normalize_tags(rec.get("visual_evidence_tags"))

        rec["visibility"] = ensure_string(rec.get("visibility")).strip()
        rec["annotator_notes"] = ensure_string(rec.get("annotator_notes"))

        legacy_desc = ensure_string(rec.get("short_canonical_description"))
        legacy_snippet = ensure_string(rec.get("report_snippet"))

        desc_en = ensure_string(rec.get("short_canonical_description_en"), default=legacy_desc).strip()
        snippet_en = ensure_string(rec.get("report_snippet_en"), default=legacy_snippet).strip()

        # If EN fields exist but are blank, fall back to legacy values from JSON.
        if not desc_en:
            desc_en = legacy_desc.strip()
        if not snippet_en:
            snippet_en = legacy_snippet.strip()

        rec["short_canonical_description_en"] = desc_en
        rec["report_snippet_en"] = snippet_en

        # Keep legacy fields synchronized with EN to preserve old pipeline compatibility.
        rec["short_canonical_description"] = rec["short_canonical_description_en"]
        rec["report_snippet"] = rec["report_snippet_en"]

        if "label_version" not in rec or rec["label_version"] is None:
            rec["label_version"] = "vlm_labels_v1"

        rec["record_id"] = ensure_string(rec.get("record_id"))
        rec["box_id"] = ensure_string(rec.get("box_id"))
        rec["category_name"] = ensure_string(rec.get("category_name"))
        rec["source"] = ensure_string(rec.get("source"))
        rec["split"] = ensure_string(rec.get("split"))
        return rec

    def get_tag_options(self) -> list[str]:
        self.refresh_if_stale()
        tags = set(DEFAULT_TAG_OPTIONS)
        for rec in self.records:
            tags.update(normalize_tags(rec.get("visual_evidence_tags")))
        return sorted(tag for tag in tags if tag)

    def get_categories(self) -> list[str]:
        self.refresh_if_stale()
        categories = {ensure_string(rec.get("category_name")).strip() for rec in self.records}
        return sorted(cat for cat in categories if cat)

    def list_record_summaries(self) -> list[dict[str, Any]]:
        self.refresh_if_stale()
        summaries: list[dict[str, Any]] = []
        for idx, rec in enumerate(self.records):
            summaries.append(
                {
                    "index": idx,
                    "record_id": rec.get("record_id"),
                    "category_name": rec.get("category_name"),
                    "visibility": rec.get("visibility"),
                    "completed": is_completed(rec),
                }
            )
        return summaries

    def get_record(self, index: int) -> dict[str, Any]:
        self.refresh_if_stale()
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Record index out of range: {index}")
        rec = dict(self.records[index])
        rec["_index"] = index
        rec["_completed"] = is_completed(rec)
        rec["_image_url"] = f"/api/image?crop_path={rec.get('crop_path', '')}"
        return rec

    def update_record(self, record_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.refresh_if_stale()
        if record_id not in self.record_id_to_index:
            raise KeyError(f"Unknown record_id: {record_id}")

        idx = self.record_id_to_index[record_id]
        rec = self.records[idx]

        for key in EDITABLE_FIELDS:
            if key not in payload:
                continue
            if key == "visual_evidence_tags":
                rec[key] = normalize_tags(payload.get(key))
            elif key == "visibility":
                visibility = ensure_string(payload.get(key)).strip()
                rec[key] = visibility if visibility in VISIBILITY_VALUES else ""
            else:
                rec[key] = ensure_string(payload.get(key))

        rec["short_canonical_description"] = ensure_string(rec.get("short_canonical_description_en")).strip()
        rec["report_snippet"] = ensure_string(rec.get("report_snippet_en")).strip()
        rec["needs_review"] = rec.get("visibility") == "ambiguous"
        rec["label_version"] = ensure_string(rec.get("label_version"), default="vlm_labels_v1")

        self.save_sidecar()
        updated = self.get_record(idx)
        return updated

    def save_sidecar(self) -> None:
        with self._lock:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.sidecar_path.with_name(f"{self.sidecar_path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for rec in self.records:
                    to_write = dict(rec)
                    # Keep primary annotation file EN-centric for compatibility.
                    to_write.pop("short_canonical_description_ru", None)
                    to_write.pop("report_snippet_ru", None)
                    f.write(json.dumps(to_write, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.sidecar_path)
            self._loaded_version = self._compute_version()

    def resolve_crop_path(self, crop_path_value: str) -> Path:
        crop_path = Path(crop_path_value)
        candidate = (self.base_dir / crop_path).resolve()

        try:
            candidate.relative_to(self.base_dir)
        except ValueError as exc:
            raise ValueError("crop_path escapes allowed base directory") from exc

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Crop image not found: {candidate}")
        return candidate


def create_app(store: AnnotationStore) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["store"] = store

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/config")
    def api_config():
        state = {
            "input_path": str(store.input_path),
            "sidecar_path": str(store.sidecar_path),
            "total_records": len(store.records),
            "categories": store.get_categories(),
            "tag_options": store.get_tag_options(),
            "visibility_values": sorted(VISIBILITY_VALUES),
        }
        return jsonify(state)

    @app.get("/api/records")
    def api_records():
        return jsonify({"records": store.list_record_summaries()})

    @app.get("/api/record/<int:index>")
    def api_record(index: int):
        try:
            rec = store.get_record(index)
        except IndexError:
            abort(404, description=f"Record index not found: {index}")
        return jsonify(rec)

    @app.post("/api/update/<record_id>")
    def api_update(record_id: str):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="Expected JSON object body")
        try:
            updated = store.update_record(record_id=record_id, payload=payload)
        except KeyError:
            abort(404, description=f"Record not found: {record_id}")
        return jsonify(updated)

    @app.post("/api/save")
    def api_save():
        store.save_sidecar()
        return jsonify({"status": "ok", "sidecar_path": str(store.sidecar_path)})

    @app.get("/api/image")
    def api_image():
        crop_path = request.args.get("crop_path", "")
        if not crop_path:
            abort(400, description="Missing crop_path query parameter")
        try:
            image_path = store.resolve_crop_path(crop_path)
        except (ValueError, FileNotFoundError) as exc:
            abort(404, description=str(exc))
        return send_file(image_path)

    return app


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    store = AnnotationStore(input_path=input_path)
    app = create_app(store)

    print(f"Input JSONL:  {store.input_path}")
    print(f"Sidecar JSONL:{store.sidecar_path}")
    print(f"Open in browser: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
