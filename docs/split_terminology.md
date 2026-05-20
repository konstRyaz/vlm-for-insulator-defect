# Split Terminology

The repository keeps historical file and run identifiers such as `train`,
`val`, and `val_v2` for reproducibility. These names should not always be read
literally in the methodological interpretation.

For frozen VLM experiments, including zero-shot, prompt-based, verifier,
review-gate, retrieval-prompt, and bad-crop checker runs, model weights are not
updated. In those experiments, the first labeled crop split is used as a
development/validation split for prompt variants, crop-context choices, and
decision-policy selection. The second labeled crop split, historically stored
under `val` or `val_v2`, is used as the test split for final reported metrics.

For supervised no-VLM baselines, where a classifier is trained on top of frozen
visual features such as DINOv2 or CLIP, the first labeled crop split is a
training split. The historical `val`/`val_v2` split is still interpreted as the
test split.

Therefore, paths such as
`outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl` are kept
unchanged as artifact names, but report text should describe them as the test
split unless it explicitly refers to an internal file path.
