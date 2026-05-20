# VLM Labels v1 Specification

## Purpose

`vlm_labels_v1` is a structured annotation schema for region-level (`bbox/crop`) supervision.
It exists to make Stage 3 measurable, parseable, and robust for analysis.

Why structured labels instead of free text:

- stable parsing and easier QA;
- explicit supervision targets for VLM slot extraction;
- cleaner comparison between `GT bbox -> VLM` and `pred bbox -> VLM`;
- reproducible handoff to report generation.

## Why Stage 3 Starts with GT BBox

We start with `GT bbox -> VLM` first to isolate VLM behavior from detector noise.

This gives:

- a cleaner upper-bound estimate of VLM quality;
- faster debugging of prompts/schema/output parsing;
- a reliable baseline before introducing detection errors.

After GT baseline is stable, we move to `pred bbox -> VLM`.

## Record Fields

Minimum required fields:

- `record_id`
- `image_id`
- `box_id`
- `source` (`gt` or `pred`)
- `split`
- `bbox_xywh`
- `coarse_class`
- `visual_evidence_tags` (array of strings)
- `visibility` (`clear`, `partial`, `ambiguous`)
- `needs_review` (boolean)
- `short_canonical_description`
- `report_snippet`

Recommended optional fields:

- `crop_path`
- `image_path`
- `score` (nullable for GT)
- `category_name`
- `annotator_notes`
- `label_version`

## Annotation Guidelines

1. Describe only visually observable evidence.
2. Do not invent causes or hidden physical mechanisms.
3. Explicitly separate clear defect cases from ambiguous cases.
4. Keep split integrity by image (`train/val/test`) rather than by crop.
5. Keep wording concise and consistent across similar visual patterns.

## Role of `unknown`

`unknown` is treated as an uncertainty/review bucket in baseline policy.
It can be labeled when present, but should not become the main semantic target for descriptive claims.

Guideline:

- if evidence is insufficient, mark `needs_review: true` and use `visibility: ambiguous`;
- avoid strong defect claims when region quality is low.

## `short_canonical_description` vs `report_snippet`

- `short_canonical_description`
  - compact, normalized phrase for analysis and consistency.
  - intended for slot-level comparison.

- `report_snippet`
  - human-readable sentence fragment that can be inserted into a final report.
  - may include slightly richer phrasing, but must remain evidence-grounded.

## Testid Example Records

```json
{
  "record_id": "val_img100017_ann42",
  "image_id": 100017,
  "box_id": "ann42",
  "source": "gt",
  "split": "val",
  "bbox_xywh": [413.0, 212.0, 91.0, 186.0],
  "coarse_class": "defect_flashover",
  "visual_evidence_tags": ["surface_burn_trace", "dark_streak", "localized_damage"],
  "visibility": "clear",
  "needs_review": false,
  "short_canonical_description": "Flashover-like surface damage is visible on the insulator unit.",
  "report_snippet": "Detected insulator region shows clear flashover-like surface damage.",
  "crop_path": "stage3_gt_crops/crops/val/defect_flashover/val_img100017_ann42.jpg",
  "image_path": "val/images/100017.jpg",
  "score": null,
  "category_name": "defect_flashover",
  "annotator_notes": "High confidence visual evidence.",
  "label_version": "vlm_labels_v1"
}
```

```json
{
  "record_id": "val_img986_ann18",
  "image_id": 986,
  "box_id": "ann18",
  "source": "gt",
  "split": "val",
  "bbox_xywh": [210.0, 148.0, 72.0, 158.0],
  "coarse_class": "defect_broken",
  "visual_evidence_tags": ["missing_fragment", "edge_discontinuity"],
  "visibility": "partial",
  "needs_review": false,
  "short_canonical_description": "Possible broken section with missing material is visible.",
  "report_snippet": "The selected insulator segment appears broken with visible material loss.",
  "score": null,
  "category_name": "defect_broken",
  "label_version": "vlm_labels_v1"
}
```

```json
{
  "record_id": "val_img1268_ann7",
  "image_id": 1268,
  "box_id": "ann7",
  "source": "pred",
  "split": "val",
  "bbox_xywh": [522.0, 260.0, 54.0, 130.0],
  "coarse_class": "unknown",
  "visual_evidence_tags": ["low_contrast", "unclear_boundary"],
  "visibility": "ambiguous",
  "needs_review": true,
  "short_canonical_description": "Region appearance is uncertain and not confidently classifiable.",
  "report_snippet": "One region is uncertain and should be reviewed manually.",
  "score": 0.18,
  "category_name": "unknown",
  "label_version": "vlm_labels_v1"
}
```
