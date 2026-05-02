#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def src(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(True)


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def build_stage3_clean_onepass() -> None:
    nb = load_nb(NOTEBOOKS_DIR / "stage3_qwen_kaggle_onepass.ipynb")
    nb["cells"][0]["source"] = src(
        "# Stage 3 Qwen One-Pass Clean (Kaggle)\n\n"
        "Leakage-free GT-crop rerun notebook. It uses a `_nocroppath` prompt version, runs preflight + full val + eval + visuals, and packs the full artifact set into one archive.\n"
    )
    nb["cells"][1]["source"] = src(
        "import json\n"
        "import os\n"
        "import shutil\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        'REPO_URL = "https://github.com/konstRyaz/vlm-for-insulator-defect-detection.git"\n'
        'REPO_DIR = Path("/kaggle/working/vlm-for-insulator-defect-detection")\n\n'
        "DATASET_ROOT_CANDIDATES = [\n"
        '    Path("/kaggle/input/datasets/kostyaryazanov/idid-coco-v3"),\n'
        '    Path("/kaggle/input/idid-coco-v3"),\n'
        "]\n\n"
        'JSONL_REL = Path("stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")\n\n'
        'BACKEND_MODE = "qwen_hf"\n'
        'PROMPT_VERSION = "qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath"\n'
        'RUN_ID = "stage3_qwen_val_v2_clean_final"\n\n'
        "DO_PREFLIGHT = True\n"
        "PREFLIGHT_SAMPLES = 1\n"
        "DO_FULL_RUN = True\n"
        "MAX_VISIBILITY_ERROR_SAMPLES = 18\n\n"
        'print("RUN_ID:", RUN_ID)\n'
    )
    nb["cells"][3]["source"] = src(
        "# 1) Setup: clean clone + hard clean-prompt checks + deps + GPU check\n"
        "import yaml\n\n"
        "if REPO_DIR.exists():\n"
        '    print("Removing existing repo to avoid stale prompts:", REPO_DIR)\n'
        "    shutil.rmtree(REPO_DIR)\n\n"
        'sh(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")\n\n'
        'git_head = subprocess.check_output("git rev-parse --short HEAD", shell=True, cwd=str(REPO_DIR), text=True).strip()\n'
        'print("git_head:", git_head)\n\n'
        'cfg_path = REPO_DIR / "configs/pipeline/stage3_vlm_gt_baseline.yaml"\n'
        "if not cfg_path.exists():\n"
        '    raise FileNotFoundError(f"Missing required repo file: {cfg_path}")\n'
        'cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))\n'
        'versions = cfg["prompt"]["versions"]\n'
        "if PROMPT_VERSION not in versions:\n"
        '    raise RuntimeError(f"Prompt version not found in config: {PROMPT_VERSION}")\n'
        'if not PROMPT_VERSION.endswith("_nocroppath"):\n'
        '    raise RuntimeError("Selected prompt version is not a clean _nocroppath variant")\n'
        'prompt_system_rel = Path(versions[PROMPT_VERSION]["system_path"])\n'
        'prompt_user_rel = Path(versions[PROMPT_VERSION]["user_path"])\n'
        "for rel in [prompt_system_rel, prompt_user_rel]:\n"
        "    abs_path = REPO_DIR / rel\n"
        "    if not abs_path.exists():\n"
        '        raise FileNotFoundError(f"Missing required repo file: {abs_path}")\n\n'
        'sys_text = (REPO_DIR / prompt_system_rel).read_text(encoding="utf-8")\n'
        'usr_text = (REPO_DIR / prompt_user_rel).read_text(encoding="utf-8")\n'
        'if "Visibility policy:" not in sys_text:\n'
        '    raise RuntimeError("Unexpected system prompt content")\n'
        'if "crop_path" in usr_text:\n'
        '    raise RuntimeError("Selected clean user prompt still exposes crop_path")\n\n'
        'print("Resolved system prompt:", prompt_system_rel)\n'
        'print("Resolved user prompt  :", prompt_user_rel)\n'
        'print("Repo clean-prompt checks: OK")\n\n'
        'sh("python -m pip install -q -U pip")\n'
        'sh(f"python -m pip install -q -r {REPO_DIR / \'requirements.txt\'}")\n'
        'sh("python -m pip install -q -U transformers accelerate qwen-vl-utils")\n\n'
        'sh("nvidia-smi", check=False)\n'
        'print("cwd:", REPO_DIR)\n'
    )
    nb["cells"][7]["source"] = src(
        "# 5) Print key metrics in notebook output\n"
        "import pandas as pd\n\n"
        'METRICS_PATH = EVAL_DIR / "metrics.json"\n'
        'RUN_SUMMARY_PATH = RUN_DIR / "run_summary.json"\n\n'
        'metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))\n'
        'run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))\n\n'
        'rates = metrics.get("rates", {})\n'
        'f1 = metrics.get("f1", {})\n'
        'counts = metrics.get("counts", {})\n'
        'prompt_flags = run_summary.get("prompt_selection", {})\n\n'
        "summary = {\n"
        '    "run_id": RUN_ID,\n'
        '    "prompt_version": PROMPT_VERSION,\n'
        '    "backend": run_summary.get("backend_mode_effective"),\n'
        '    "model": run_summary.get("backend_settings_effective", {}).get("model"),\n'
        '    "user_prompt_contains_crop_path_token": prompt_flags.get("user_prompt_contains_crop_path_token"),\n'
        '    "records_attempted": run_summary.get("counters", {}).get("records_attempted"),\n'
        '    "status_ok": run_summary.get("counters", {}).get("status_ok"),\n'
        '    "status_backend_error": run_summary.get("counters", {}).get("status_backend_error"),\n'
        '    "parse_success_rate": rates.get("parse_success_rate"),\n'
        '    "schema_valid_rate": rates.get("schema_valid_rate"),\n'
        '    "coarse_class_accuracy": rates.get("coarse_class_accuracy"),\n'
        '    "coarse_class_macro_f1": f1.get("coarse_class_macro_f1"),\n'
        '    "visibility_accuracy": rates.get("visibility_accuracy"),\n'
        '    "visibility_macro_f1": f1.get("visibility_macro_f1"),\n'
        '    "needs_review_accuracy": rates.get("needs_review_accuracy"),\n'
        '    "tag_exact_match_rate": rates.get("tag_exact_match_rate"),\n'
        '    "tag_mean_jaccard": rates.get("tag_mean_jaccard"),\n'
        '    "pred_ambiguous_rate": rates.get("pred_ambiguous_rate"),\n'
        '    "gt_ambiguous_rate": rates.get("gt_ambiguous_rate"),\n'
        '    "samples_evaluated": counts.get("samples_evaluated") or counts.get("evaluated_total"),\n'
        "}\n\n"
        'print("=== STAGE3 CLEAN RUN SUMMARY ===")\n'
        "for k, v in summary.items():\n"
        '    print(f"{k}: {v}")\n\n'
        'print("\\nmetrics.json:", METRICS_PATH)\n'
        'print("run_summary.json:", RUN_SUMMARY_PATH)\n\n'
        "summary_df = pd.DataFrame([summary]).T\n"
        'summary_df.columns = ["value"]\n'
        "summary_df\n"
    )
    nb["cells"][10]["source"] = src(
        "# 8) Collect deliverables + pack archive\n"
        'DELIVER_DIR = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID}")\n'
        "if DELIVER_DIR.exists():\n"
        "    shutil.rmtree(DELIVER_DIR)\n"
        "DELIVER_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "to_copy = [\n"
        '    RUN_DIR / "run_summary.json",\n'
        '    RUN_DIR / "config_snapshot.json",\n'
        '    RUN_DIR / "predictions_vlm_labels_v1.jsonl",\n'
        '    RUN_DIR / "parsed_predictions.jsonl",\n'
        '    RUN_DIR / "raw_responses.jsonl",\n'
        '    RUN_DIR / "sample_results.jsonl",\n'
        '    RUN_DIR / "failures.jsonl",\n'
        '    EVAL_DIR / "metrics.json",\n'
        '    EVAL_DIR / "confusion_coarse_class.csv",\n'
        '    EVAL_DIR / "confusion_visibility.csv",\n'
        '    EVAL_DIR / "review_table.csv",\n'
        '    EVAL_DIR / "failures.jsonl",\n'
        '    EVAL_DIR / "visuals" / "report.md",\n'
        '    EVAL_DIR / "visuals" / "kpi_overview.png",\n'
        '    EVAL_DIR / "visuals" / "confusion_coarse_class.png",\n'
        '    EVAL_DIR / "visuals" / "confusion_visibility.png",\n'
        '    EVAL_DIR / "visuals" / "visibility_errors_top.csv",\n'
        '    EVAL_DIR / "visuals" / "visibility_errors_gallery.png",\n'
        "]\n\n"
        "for src_path in to_copy:\n"
        "    if src_path.exists():\n"
        "        rel = src_path.relative_to(RUN_DIR)\n"
        "        dst = DELIVER_DIR / rel\n"
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copy2(src_path, dst)\n\n"
        'summary_md = DELIVER_DIR / "RESULT_SUMMARY.md"\n'
        "summary_md.write_text(\n"
        '    "\\n".join([\n'
        '        f"# Stage 3 Clean One-Pass Result: {RUN_ID}",\n'
        '        "",\n'
        '        f"- prompt_version: {PROMPT_VERSION}",\n'
        '        f"- backend: {summary[\'backend\']}",\n'
        '        f"- model: {summary[\'model\']}",\n'
        '        f"- user_prompt_contains_crop_path_token: {summary[\'user_prompt_contains_crop_path_token\']}",\n'
        '        f"- records_attempted: {summary[\'records_attempted\']}",\n'
        '        f"- status_ok: {summary[\'status_ok\']}",\n'
        '        f"- status_backend_error: {summary[\'status_backend_error\']}",\n'
        '        "",\n'
        '        "## Main metrics",\n'
        '        f"- parse_success_rate: {summary[\'parse_success_rate\']}",\n'
        '        f"- schema_valid_rate: {summary[\'schema_valid_rate\']}",\n'
        '        f"- coarse_class_accuracy: {summary[\'coarse_class_accuracy\']}",\n'
        '        f"- coarse_class_macro_f1: {summary[\'coarse_class_macro_f1\']}",\n'
        '        f"- visibility_accuracy: {summary[\'visibility_accuracy\']}",\n'
        '        f"- visibility_macro_f1: {summary[\'visibility_macro_f1\']}",\n'
        '        f"- needs_review_accuracy: {summary[\'needs_review_accuracy\']}",\n'
        '        f"- tag_exact_match_rate: {summary[\'tag_exact_match_rate\']}",\n'
        '        f"- tag_mean_jaccard: {summary[\'tag_mean_jaccard\']}",\n'
        '        f"- pred_ambiguous_rate: {summary[\'pred_ambiguous_rate\']}",\n'
        '        f"- gt_ambiguous_rate: {summary[\'gt_ambiguous_rate\']}",\n'
        '        "",\n'
        '        f"Ground truth JSONL: {VAL_JSONL}",\n'
        '        f"Run dir: {RUN_DIR}",\n'
        "    ]),\n"
        '    encoding="utf-8",\n'
        ")\n\n"
        'ARCHIVE_BASE = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID}")\n'
        'ARCHIVE_PATH = shutil.make_archive(str(ARCHIVE_BASE), "gztar", root_dir=DELIVER_DIR)\n\n'
        'print("DELIVER_DIR:", DELIVER_DIR)\n'
        'print("ARCHIVE_PATH:", ARCHIVE_PATH)\n'
        'print("\\nFiles in deliverables:")\n'
        'for p in sorted(DELIVER_DIR.rglob("*")):\n'
        "    if p.is_file():\n"
        '        print("-", p.relative_to(DELIVER_DIR))\n'
    )
    nb["cells"][11]["source"] = src(
        "## Artifacts\n\n"
        "Run outputs: `outputs/stage3_vlm_baseline_runs/stage3_qwen_val_v2_clean_final/`\n"
        "Packed archive: `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_clean_final.tar.gz`\n"
    )
    save_nb(NOTEBOOKS_DIR / "stage3_qwen_kaggle_clean_onepass.ipynb", nb)


def build_stage3_clean_sweep() -> None:
    nb = load_nb(NOTEBOOKS_DIR / "stage3_prompt_sweep_visibility_v6.ipynb")
    nb["cells"][0]["source"] = src(
        "# Stage 3 Prompt Sweep Clean (Visibility v6 Family, Kaggle)\n\n"
        "Leakage-free full-val prompt sweep on Qwen2.5-VL-3B-Instruct. All prompt variants use `_nocroppath` user templates, and the notebook fails fast if any selected prompt still exposes `crop_path`.\n"
    )
    nb["cells"][1]["source"] = src(
        "import json\n"
        "import shutil\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        'REPO_URL = "https://github.com/konstRyaz/vlm-for-insulator-defect-detection.git"\n'
        'REPO_DIR = Path("/kaggle/working/vlm-for-insulator-defect-detection")\n\n'
        "DATASET_ROOT_CANDIDATES = [\n"
        '    Path("/kaggle/input/datasets/kostyaryazanov/idid-coco-v3"),\n'
        '    Path("/kaggle/input/idid-coco-v3"),\n'
        "]\n"
        'JSONL_REL = Path("stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")\n\n'
        'BACKEND_MODE = "qwen_hf"\n'
        'RUN_ID_PREFIX = "stage3_qwen_val_v2_sweep_v6_clean"\n\n'
        "PROMPT_VERSIONS = [\n"
        '    "qwen_vlm_labels_v1_prompt_v3_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v4_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v6a_notaglock_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v6c_balanced_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v6e_partial_geometry_nocroppath",\n'
        "]\n\n"
        'CONTROL_VERSION = "qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best_nocroppath"\n\n'
        'print("Prompt sweep size:", len(PROMPT_VERSIONS))\n'
        "from IPython.display import display\n"
    )
    nb["cells"][5]["source"] = src(
        "# 3) Hard preflight checks: registry wiring + clean content fingerprints per version\n"
        "import yaml\n\n"
        'cfg_path = REPO_DIR / "configs" / "pipeline" / "stage3_vlm_gt_baseline.yaml"\n'
        'cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))\n'
        'versions = cfg["prompt"]["versions"]\n\n'
        "expected_paths = {\n"
        '    "qwen_vlm_labels_v1_prompt_v3_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v3_visibility_tag_calibrated.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v3_visibility_tag_calibrated_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v4_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v4_visibility_recalibrated.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v4_visibility_recalibrated_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v5a_visibility_gate_best.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v5a_visibility_gate_best_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v6a_notaglock_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6a_notaglock.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6a_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6b_positive_ambiguous.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6b_positive_ambiguous_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v6c_balanced_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6c_balanced.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6c_balanced_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6d_balanced_notaglock.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v6e_partial_geometry_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6e_partial_geometry.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6e_partial_geometry_nocroppath.txt",\n'
        "    ),\n"
        "}\n\n"
        "positive_sentence = 'If the visible evidence itself cannot be trusted due to blur, glare, washout, low contrast, heavy shadow, or unstable boundaries, prefer visibility=\"ambiguous\" even when a conservative coarse_class guess is still possible.'\n"
        "anti_sentence = 'Do not use visibility=\"ambiguous\" only because defect evidence is weak, absent, or conservative while the visible region remains readable.'\n"
        "taglock_line = 'If visibility=\"ambiguous\", ensure the tags explain why.'\n\n"
        "pattern_rules = {\n"
        '    "qwen_vlm_labels_v1_prompt_v5a_visibility_gate_best_nocroppath": {\n'
        '        "sys_must": [\n'
        '            "Treat visibility as a property of visual interpretability and view completeness,",\n'
        '            "Step 1: readability of the visible region",\n'
        "        ],\n"
        '        "usr_must": [taglock_line],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": ["crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v6a_notaglock_nocroppath": {\n'
        '        "sys_must": [],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v6b_positive_ambiguous_nocroppath": {\n'
        '        "sys_must": [positive_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [anti_sentence],\n'
        '        "usr_must_not": ["crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v6c_balanced_nocroppath": {\n'
        '        "sys_must": [positive_sentence, anti_sentence],\n'
        '        "usr_must": [taglock_line],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": ["crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath": {\n'
        '        "sys_must": [positive_sentence, anti_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v6e_partial_geometry_nocroppath": {\n'
        '        "sys_must": [\n'
        '            "clearly substantial part",\n'
        '            "missing geometry or context",\n'
        '            "Use visibility=\\"partial\\" only for readable but materially incomplete geometry/view; do not use partial for image-quality uncertainty.",\n'
        '            "Readable fragment view with materially missing geometry is partial, not ambiguous.",\n'
        "        ],\n"
        '        "usr_must": [taglock_line],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": ["crop_path"],\n'
        "    },\n"
        "}\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    if pv not in versions:\n"
        '        raise RuntimeError(f"Prompt version missing in config: {pv}")\n\n'
        "    expected_sys, expected_usr = expected_paths[pv]\n"
        '    actual_sys = versions[pv]["system_path"]\n'
        '    actual_usr = versions[pv]["user_path"]\n\n'
        "    if actual_sys != expected_sys or actual_usr != expected_usr:\n"
        "        raise RuntimeError(\n"
        '            f"Registry mismatch for {pv}: expected ({expected_sys}, {expected_usr}), got ({actual_sys}, {actual_usr})"\n'
        "        )\n\n"
        "    sys_abs = REPO_DIR / actual_sys\n"
        "    usr_abs = REPO_DIR / actual_usr\n"
        '    require_path(sys_abs, f"system prompt for {pv}")\n'
        '    require_path(usr_abs, f"user prompt for {pv}")\n\n'
        '    sys_text = sys_abs.read_text(encoding="utf-8")\n'
        '    usr_text = usr_abs.read_text(encoding="utf-8")\n'
        '    if "crop_path" in usr_text:\n'
        '        raise RuntimeError(f"Clean user prompt still exposes crop_path: {pv}")\n\n'
        "    rules = pattern_rules.get(pv)\n"
        "    if rules:\n"
        '        for pattern in rules.get("sys_must", []):\n'
        "            if pattern not in sys_text:\n"
        '                raise RuntimeError(f"Pattern missing in system prompt {pv}: {pattern}")\n'
        '        for pattern in rules.get("usr_must", []):\n'
        "            if pattern not in usr_text:\n"
        '                raise RuntimeError(f"Pattern missing in user prompt {pv}: {pattern}")\n'
        '        for pattern in rules.get("sys_must_not", []):\n'
        "            if pattern in sys_text:\n"
        '                raise RuntimeError(f"Forbidden pattern in system prompt {pv}: {pattern}")\n'
        '        for pattern in rules.get("usr_must_not", []):\n'
        "            if pattern in usr_text:\n"
        '                raise RuntimeError(f"Forbidden pattern in user prompt {pv}: {pattern}")\n\n'
        'print("Clean prompt preflight checks passed for all sweep variants.")\n'
    )
    nb["cells"][8]["source"] = src(
        "# 6) Package sweep deliverables\n"
        'DELIVER_DIR = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        "if DELIVER_DIR.exists():\n"
        "    shutil.rmtree(DELIVER_DIR)\n"
        "DELIVER_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "for p in [AGG_DIR / 'prompt_sweep_comparison.csv', AGG_DIR / 'prompt_sweep_comparison.md']:\n"
        "    if p.exists():\n"
        '        dst = DELIVER_DIR / "aggregate" / p.name\n'
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copy2(p, dst)\n\n"
        "run_artifacts_rel = [\n"
        '    "run_summary.json",\n'
        '    "config_snapshot.json",\n'
        '    "predictions_vlm_labels_v1.jsonl",\n'
        '    "parsed_predictions.jsonl",\n'
        '    "raw_responses.jsonl",\n'
        '    "sample_results.jsonl",\n'
        '    "failures.jsonl",\n'
        '    "eval/metrics.json",\n'
        '    "eval/confusion_coarse_class.csv",\n'
        '    "eval/confusion_visibility.csv",\n'
        '    "eval/review_table.csv",\n'
        '    "eval/failures.jsonl",\n'
        '    "eval/visuals/report.md",\n'
        '    "eval/visuals/kpi_overview.png",\n'
        '    "eval/visuals/confusion_coarse_class.png",\n'
        '    "eval/visuals/confusion_visibility.png",\n'
        '    "eval/visuals/visibility_errors_top.csv",\n'
        '    "eval/visuals/visibility_errors_gallery.png",\n'
        "]\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    run_dir = run_dirs[pv]\n"
        "    for rel in run_artifacts_rel:\n"
        "        src_path = run_dir / rel\n"
        "        if src_path.exists():\n"
        '            dst = DELIVER_DIR / "runs" / pv / rel\n'
        "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "            shutil.copy2(src_path, dst)\n\n"
        'summary_md = DELIVER_DIR / "RESULT_SUMMARY.md"\n'
        "summary_md.write_text(\n"
        '    "\\n".join([\n'
        '        f"# Stage 3 Clean Prompt Sweep Result: {RUN_ID_PREFIX}",\n'
        '        "",\n'
        '        "- model: Qwen/Qwen2.5-VL-3B-Instruct",\n'
        '        f"- backend: {BACKEND_MODE}",\n'
        '        f"- control_version: {CONTROL_VERSION}",\n'
        '        f"- prompt_count: {len(PROMPT_VERSIONS)}",\n'
        '        f"- best_prompt_version: {best[\'prompt_version\']}",\n'
        '        f"- best_run_id: {best[\'run_id\']}",\n'
        '        f"- best_verdict: {best[\'verdict\']}",\n'
        '        "",\n'
        '        f"Ground truth JSONL: {VAL_JSONL}",\n'
        '        f"Repo commit: {git_head}",\n'
        "    ]),\n"
        '    encoding="utf-8",\n'
        ")\n\n"
        'ARCHIVE_BASE = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        'ARCHIVE_PATH = shutil.make_archive(str(ARCHIVE_BASE), "gztar", root_dir=DELIVER_DIR)\n\n'
        'print("DELIVER_DIR:", DELIVER_DIR)\n'
        'print("ARCHIVE_PATH:", ARCHIVE_PATH)\n'
        'print("\\nFiles in deliverables:")\n'
        'for p in sorted(DELIVER_DIR.rglob("*")):\n'
        "    if p.is_file():\n"
        '        print("-", p.relative_to(DELIVER_DIR))\n'
    )
    nb["cells"][9]["source"] = src(
        "## Artifacts\n\n"
        "Per-run outputs: `outputs/stage3_vlm_baseline_runs/<RUN_ID_PREFIX>...`\n"
        "Aggregate files: `.../<RUN_ID_PREFIX>_aggregate/`\n"
        "Packed archive: `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_sweep_v6_clean.tar.gz`\n"
    )
    save_nb(NOTEBOOKS_DIR / "stage3_prompt_sweep_visibility_v6_clean.ipynb", nb)


def build_stage3_clean_micro() -> None:
    nb = load_nb(NOTEBOOKS_DIR / "stage3_final_micro_ablation_v6d_vs_v6f.ipynb")
    nb["cells"][0]["source"] = src(
        "# Stage 3 Final Micro-Ablation Clean: v6d vs v6f (Kaggle)\n\n"
        "Leakage-free full-val comparison between the clean v6d control and the clean v6f micro-ablation. Both variants use `_nocroppath` user templates.\n"
    )
    nb["cells"][1]["source"] = src(
        "import json\n"
        "import shutil\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n"
        "from IPython.display import display\n\n"
        'REPO_URL = "https://github.com/konstRyaz/vlm-for-insulator-defect-detection.git"\n'
        'REPO_DIR = Path("/kaggle/working/vlm-for-insulator-defect-detection")\n\n'
        "DATASET_ROOT_CANDIDATES = [\n"
        '    Path("/kaggle/input/datasets/kostyaryazanov/idid-coco-v3"),\n'
        '    Path("/kaggle/input/idid-coco-v3"),\n'
        "]\n"
        'JSONL_REL = Path("stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")\n\n'
        'BACKEND_MODE = "qwen_hf"\n'
        'RUN_ID_PREFIX = "stage3_qwen_val_v2_v6d_vs_v6f_clean"\n\n'
        'CONTROL_VERSION = "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath"\n'
        'CANDIDATE_VERSION = "qwen_vlm_labels_v1_prompt_v6f_balanced_notaglock_soft_ambiguous_raise_nocroppath"\n'
        "PROMPT_VERSIONS = [CONTROL_VERSION, CANDIDATE_VERSION]\n"
    )
    nb["cells"][5]["source"] = src(
        "# 3) Hard preflight checks for clean v6d/v6f registry + prompt content\n"
        "import yaml\n\n"
        'cfg_path = REPO_DIR / "configs" / "pipeline" / "stage3_vlm_gt_baseline.yaml"\n'
        'cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))\n'
        'versions = cfg["prompt"]["versions"]\n\n'
        "expected_paths = {\n"
        "    CONTROL_VERSION: (\n"
        '        "configs/pipeline/prompts/stage3_vlm_system_v6d_balanced_notaglock.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        "    CANDIDATE_VERSION: (\n"
        '        "configs/pipeline/prompts/stage3_vlm_system_v6f_balanced_notaglock_soft_ambiguous_raise.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6f_balanced_notaglock_soft_ambiguous_raise_nocroppath.txt",\n'
        "    ),\n"
        "}\n\n"
        "positive_sentence = 'If the visible evidence itself cannot be trusted due to blur, glare, washout, low contrast, heavy shadow, or unstable boundaries, prefer visibility=\"ambiguous\" even when a conservative coarse_class guess is still possible.'\n"
        "anti_sentence = 'Do not use visibility=\"ambiguous\" only because defect evidence is weak, absent, or conservative while the visible region remains readable.'\n"
        "soft_raise_sentence = 'If the visible evidence is borderline readable and a suspicious mark or boundary could plausibly be explained by image quality rather than object structure, prefer visibility=\"ambiguous\" over visibility=\"clear\".'\n"
        "taglock_line = 'If visibility=\"ambiguous\", ensure the tags explain why.'\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    if pv not in versions:\n"
        '        raise RuntimeError(f"Prompt version missing in config: {pv}")\n'
        "    expected_sys, expected_usr = expected_paths[pv]\n"
        '    actual_sys = versions[pv]["system_path"]\n'
        '    actual_usr = versions[pv]["user_path"]\n'
        "    if actual_sys != expected_sys or actual_usr != expected_usr:\n"
        "        raise RuntimeError(\n"
        '            f"Registry mismatch for {pv}: expected ({expected_sys}, {expected_usr}), got ({actual_sys}, {actual_usr})"\n'
        "        )\n\n"
        "    sys_abs = REPO_DIR / actual_sys\n"
        "    usr_abs = REPO_DIR / actual_usr\n"
        '    require_path(sys_abs, f"system prompt for {pv}")\n'
        '    require_path(usr_abs, f"user prompt for {pv}")\n'
        '    sys_text = sys_abs.read_text(encoding="utf-8")\n'
        '    usr_text = usr_abs.read_text(encoding="utf-8")\n'
        '    if "crop_path" in usr_text:\n'
        '        raise RuntimeError(f"Clean user prompt still exposes crop_path: {pv}")\n'
        "    if taglock_line in usr_text:\n"
        '        raise RuntimeError(f"Unexpected tag-lock line in clean notaglock prompt: {pv}")\n'
        "    if positive_sentence not in sys_text or anti_sentence not in sys_text:\n"
        '        raise RuntimeError(f"Missing balanced visibility rules in {pv}")\n'
        "    if pv == CANDIDATE_VERSION and soft_raise_sentence not in sys_text:\n"
        '        raise RuntimeError("Missing soft ambiguous raise sentence in clean v6f")\n'
        "    if pv == CONTROL_VERSION and soft_raise_sentence in sys_text:\n"
        '        raise RuntimeError("Soft ambiguous raise sentence leaked into clean v6d control")\n\n'
        'print("Clean v6d/v6f prompt checks passed.")\n'
    )
    nb["cells"][8]["source"] = src(
        "# 6) Package deliverables\n"
        'DELIVER_DIR = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        "if DELIVER_DIR.exists():\n"
        "    shutil.rmtree(DELIVER_DIR)\n"
        "DELIVER_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "aggregate_files = [AGG_DIR / 'v6d_vs_v6f_comparison.csv', AGG_DIR / 'v6d_vs_v6f_comparison.md']\n"
        "for p in aggregate_files:\n"
        "    if p.exists():\n"
        '        dst = DELIVER_DIR / "aggregate" / p.name\n'
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copy2(p, dst)\n\n"
        "run_artifacts_rel = [\n"
        '    "run_summary.json",\n'
        '    "config_snapshot.json",\n'
        '    "predictions_vlm_labels_v1.jsonl",\n'
        '    "parsed_predictions.jsonl",\n'
        '    "raw_responses.jsonl",\n'
        '    "sample_results.jsonl",\n'
        '    "failures.jsonl",\n'
        '    "eval/metrics.json",\n'
        '    "eval/confusion_coarse_class.csv",\n'
        '    "eval/confusion_visibility.csv",\n'
        '    "eval/review_table.csv",\n'
        '    "eval/failures.jsonl",\n'
        '    "eval/visuals/report.md",\n'
        '    "eval/visuals/kpi_overview.png",\n'
        '    "eval/visuals/confusion_coarse_class.png",\n'
        '    "eval/visuals/confusion_visibility.png",\n'
        '    "eval/visuals/visibility_errors_top.csv",\n'
        '    "eval/visuals/visibility_errors_gallery.png",\n'
        "]\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    run_dir = run_dirs[pv]\n"
        "    for rel in run_artifacts_rel:\n"
        "        src_path = run_dir / rel\n"
        "        if src_path.exists():\n"
        '            dst = DELIVER_DIR / "runs" / pv / rel\n'
        "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "            shutil.copy2(src_path, dst)\n\n"
        'summary_md = DELIVER_DIR / "RESULT_SUMMARY.md"\n'
        "summary_md.write_text(\n"
        '    "\\n".join([\n'
        '        f"# Stage 3 Clean Micro-Ablation Result: {RUN_ID_PREFIX}",\n'
        '        "",\n'
        '        f"- control_version: {CONTROL_VERSION}",\n'
        '        f"- candidate_version: {CANDIDATE_VERSION}",\n'
        '        f"- selected_version: {selected_version}",\n'
        '        f"- decision: {decision}",\n'
        '        "",\n'
        '        f"Ground truth JSONL: {VAL_JSONL}",\n'
        '        f"Repo commit: {git_head}",\n'
        "    ]),\n"
        '    encoding="utf-8",\n'
        ")\n\n"
        'ARCHIVE_BASE = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        'ARCHIVE_PATH = shutil.make_archive(str(ARCHIVE_BASE), "gztar", root_dir=DELIVER_DIR)\n\n'
        'print("DELIVER_DIR:", DELIVER_DIR)\n'
        'print("ARCHIVE_PATH:", ARCHIVE_PATH)\n'
    )
    nb["cells"][9]["source"] = src(
        "## Artifacts\n\n"
        "Per-run outputs: `outputs/stage3_vlm_baseline_runs/<RUN_ID_PREFIX>...`\n"
        "Packed archive: `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_v6d_vs_v6f_clean.tar.gz`\n"
    )
    save_nb(NOTEBOOKS_DIR / "stage3_final_micro_ablation_v6d_vs_v6f_clean.ipynb", nb)


def build_stage3_coarse_recovery_sweep() -> None:
    nb = load_nb(NOTEBOOKS_DIR / "stage3_prompt_sweep_visibility_v6_clean.ipynb")
    nb["cells"][0]["source"] = src(
        "# Stage 3 Prompt Sweep Clean (Coarse Recovery v7, Kaggle)\n\n"
        "Leakage-free full-val coarse-class recovery sweep on Qwen2.5-VL-3B-Instruct. "
        "It keeps the clean `_nocroppath` path, uses `v6d_nocroppath` as control, and tests four small coarse-policy variants aimed at recovering defect-vs-ok discrimination without sacrificing the improved visibility calibration.\n"
    )
    nb["cells"][1]["source"] = src(
        "import json\n"
        "import shutil\n"
        "import subprocess\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        'REPO_URL = "https://github.com/konstRyaz/vlm-for-insulator-defect-detection.git"\n'
        'REPO_DIR = Path("/kaggle/working/vlm-for-insulator-defect-detection")\n\n'
        "DATASET_ROOT_CANDIDATES = [\n"
        '    Path("/kaggle/input/datasets/kostyaryazanov/idid-coco-v3"),\n'
        '    Path("/kaggle/input/idid-coco-v3"),\n'
        "]\n"
        'JSONL_REL = Path("stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl")\n\n'
        'BACKEND_MODE = "qwen_hf"\n'
        'RUN_ID_PREFIX = "stage3_qwen_val_v2_sweep_v7_coarse_clean"\n\n'
        "PROMPT_VERSIONS = [\n"
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v7a_ok_requires_positive_intact_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v7b_flashover_object_bound_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v7c_broken_fragment_support_nocroppath",\n'
        '    "qwen_vlm_labels_v1_prompt_v7d_balanced_coarse_recovery_nocroppath",\n'
        "]\n\n"
        'CONTROL_VERSION = "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath"\n\n'
        'print("Prompt sweep size:", len(PROMPT_VERSIONS))\n'
        "from IPython.display import display\n"
    )
    nb["cells"][2]["source"] = src(
        "def sh(cmd: str, cwd: Path | None = None, check: bool = True):\n"
        "    print(f\"$ {cmd}\")\n"
        "    p = subprocess.run(\n"
        "        cmd,\n"
        "        shell=True,\n"
        "        cwd=str(cwd) if cwd else None,\n"
        "        text=True,\n"
        "        capture_output=True,\n"
        "    )\n"
        "    if p.stdout:\n"
        "        print(p.stdout)\n"
        "    if p.stderr:\n"
        "        print(p.stderr)\n"
        "    if check and p.returncode != 0:\n"
        "        raise RuntimeError(f\"Command failed ({p.returncode}): {cmd}\")\n"
        "    return p\n\n"
        "def require_path(path: Path, label: str):\n"
        "    if not path.exists():\n"
        "        raise FileNotFoundError(f\"{label} not found: {path}\")\n"
        "    return path\n\n"
        "def read_json(path: Path):\n"
        "    return json.loads(path.read_text(encoding=\"utf-8\"))\n\n"
        "def metrics_row(prompt_version: str, run_id: str, run_dir: Path, eval_dir: Path) -> dict:\n"
        "    metrics = read_json(eval_dir / \"metrics.json\")\n"
        "    run_summary = read_json(run_dir / \"run_summary.json\")\n"
        "    rates = metrics.get(\"rates\", {})\n"
        "    f1 = metrics.get(\"f1\", {})\n\n"
        "    row = {\n"
        "        \"prompt_version\": prompt_version,\n"
        "        \"run_id\": run_id,\n"
        "        \"parse_success\": float(rates.get(\"parse_success_rate\", 0.0)),\n"
        "        \"schema_valid\": float(rates.get(\"schema_valid_rate\", 0.0)),\n"
        "        \"coarse_acc\": float(rates.get(\"coarse_class_accuracy\", 0.0)),\n"
        "        \"coarse_macro_f1\": float(f1.get(\"coarse_class_macro_f1\", 0.0)),\n"
        "        \"visibility_acc\": float(rates.get(\"visibility_accuracy\", 0.0)),\n"
        "        \"visibility_macro_f1\": float(f1.get(\"visibility_macro_f1\", 0.0)),\n"
        "        \"needs_review_acc\": float(rates.get(\"needs_review_accuracy\", 0.0)),\n"
        "        \"tag_exact\": float(rates.get(\"tag_exact_match_rate\", 0.0)),\n"
        "        \"tag_mean_jaccard\": float(rates.get(\"tag_mean_jaccard\", 0.0)),\n"
        "        \"pred_ambiguous_rate\": float(rates.get(\"pred_ambiguous_rate\", 0.0)),\n"
        "        \"gt_ambiguous_rate\": float(rates.get(\"gt_ambiguous_rate\", 0.0)),\n"
        "        \"backend\": run_summary.get(\"backend_mode_effective\"),\n"
        "        \"model\": run_summary.get(\"backend_settings_effective\", {}).get(\"model\"),\n"
        "    }\n"
        "    row[\"abs_ambiguous_gap\"] = abs(row[\"pred_ambiguous_rate\"] - row[\"gt_ambiguous_rate\"])\n"
        "    return row\n\n"
        "def pass_rule(row: dict, control_row: dict) -> bool:\n"
        "    return (\n"
        "        row[\"parse_success\"] == 1.0\n"
        "        and row[\"schema_valid\"] == 1.0\n"
        "        and row[\"coarse_macro_f1\"] >= (control_row[\"coarse_macro_f1\"] + 0.04)\n"
        "        and row[\"coarse_acc\"] >= (control_row[\"coarse_acc\"] + 0.02)\n"
        "        and row[\"visibility_macro_f1\"] >= (control_row[\"visibility_macro_f1\"] - 0.03)\n"
        "        and row[\"tag_mean_jaccard\"] >= (control_row[\"tag_mean_jaccard\"] - 0.015)\n"
        "    )\n\n"
        "def soft_pass_rule(row: dict, control_row: dict) -> bool:\n"
        "    return (\n"
        "        row[\"parse_success\"] == 1.0\n"
        "        and row[\"schema_valid\"] == 1.0\n"
        "        and row[\"coarse_macro_f1\"] > (control_row[\"coarse_macro_f1\"] + 0.015)\n"
        "        and row[\"coarse_acc\"] >= (control_row[\"coarse_acc\"] - 0.02)\n"
        "        and row[\"visibility_macro_f1\"] >= (control_row[\"visibility_macro_f1\"] - 0.05)\n"
        "    )\n"
    )
    nb["cells"][5]["source"] = src(
        "# 3) Hard preflight checks: registry wiring + clean content fingerprints per version\n"
        "import yaml\n\n"
        'cfg_path = REPO_DIR / "configs" / "pipeline" / "stage3_vlm_gt_baseline.yaml"\n'
        'cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))\n'
        'versions = cfg["prompt"]["versions"]\n\n'
        "expected_paths = {\n"
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v6d_balanced_notaglock.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v7a_ok_requires_positive_intact_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v7a_ok_requires_positive_intact.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v7b_flashover_object_bound_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v7b_flashover_object_bound.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v7c_broken_fragment_support_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v7c_broken_fragment_support.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        '    "qwen_vlm_labels_v1_prompt_v7d_balanced_coarse_recovery_nocroppath": (\n'
        '        "configs/pipeline/prompts/stage3_vlm_system_v7d_balanced_coarse_recovery.txt",\n'
        '        "configs/pipeline/prompts/stage3_vlm_user_v6d_balanced_notaglock_nocroppath.txt",\n'
        "    ),\n"
        "}\n\n"
        "ok_gate_sentence = 'Absence of obvious defect alone is not sufficient for insulator_ok.'\n"
        "flash_sentence = 'A localized object-bound dark trace can support defect_flashover even if the rest of the crop looks regular.'\n"
        "broken_sentence = 'A direct edge or profile discontinuity on the visible fragment can support defect_broken even when only part of the object is shown.'\n"
        "taglock_line = 'If visibility=\"ambiguous\", ensure the tags explain why.'\n\n"
        "pattern_rules = {\n"
        '    "qwen_vlm_labels_v1_prompt_v6d_balanced_notaglock_nocroppath": {\n'
        '        "sys_must": [\n'
        '            "Do not use visibility=\\"ambiguous\\" only because defect evidence is weak, absent, or conservative while the visible region remains readable.",\n'
        '            "If the visible evidence itself cannot be trusted due to blur, glare, washout, low contrast, heavy shadow, or unstable boundaries, prefer visibility=\\"ambiguous\\" even when a conservative coarse_class guess is still possible.",\n'
        "        ],\n"
        '        "usr_must": [],\n'
        '        "sys_must_not": [ok_gate_sentence, flash_sentence, broken_sentence],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v7a_ok_requires_positive_intact_nocroppath": {\n'
        '        "sys_must": [ok_gate_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [flash_sentence, broken_sentence],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v7b_flashover_object_bound_nocroppath": {\n'
        '        "sys_must": [flash_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [ok_gate_sentence, broken_sentence],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v7c_broken_fragment_support_nocroppath": {\n'
        '        "sys_must": [broken_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [ok_gate_sentence, flash_sentence],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        '    "qwen_vlm_labels_v1_prompt_v7d_balanced_coarse_recovery_nocroppath": {\n'
        '        "sys_must": [ok_gate_sentence, flash_sentence, broken_sentence],\n'
        '        "usr_must": [],\n'
        '        "sys_must_not": [],\n'
        '        "usr_must_not": [taglock_line, "crop_path"],\n'
        "    },\n"
        "}\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    if pv not in versions:\n"
        '        raise RuntimeError(f"Prompt version missing in config: {pv}")\n\n'
        "    expected_sys, expected_usr = expected_paths[pv]\n"
        '    actual_sys = versions[pv]["system_path"]\n'
        '    actual_usr = versions[pv]["user_path"]\n\n'
        "    if actual_sys != expected_sys or actual_usr != expected_usr:\n"
        "        raise RuntimeError(\n"
        '            f"Registry mismatch for {pv}: expected ({expected_sys}, {expected_usr}), got ({actual_sys}, {actual_usr})"\n'
        "        )\n\n"
        "    sys_abs = REPO_DIR / actual_sys\n"
        "    usr_abs = REPO_DIR / actual_usr\n"
        '    require_path(sys_abs, f"system prompt for {pv}")\n'
        '    require_path(usr_abs, f"user prompt for {pv}")\n\n'
        '    sys_text = sys_abs.read_text(encoding="utf-8")\n'
        '    usr_text = usr_abs.read_text(encoding="utf-8")\n'
        '    if "crop_path" in usr_text:\n'
        '        raise RuntimeError(f"Clean user prompt still exposes crop_path: {pv}")\n\n'
        "    rules = pattern_rules[pv]\n"
        '    for pattern in rules.get("sys_must", []):\n'
        "        if pattern not in sys_text:\n"
        '            raise RuntimeError(f"Pattern missing in system prompt {pv}: {pattern}")\n'
        '    for pattern in rules.get("usr_must", []):\n'
        "        if pattern not in usr_text:\n"
        '            raise RuntimeError(f"Pattern missing in user prompt {pv}: {pattern}")\n'
        '    for pattern in rules.get("sys_must_not", []):\n'
        "        if pattern in sys_text:\n"
        '            raise RuntimeError(f"Forbidden pattern in system prompt {pv}: {pattern}")\n'
        '    for pattern in rules.get("usr_must_not", []):\n'
        "        if pattern in usr_text:\n"
        '            raise RuntimeError(f"Forbidden pattern in user prompt {pv}: {pattern}")\n\n'
        'print("Coarse-recovery clean prompt preflight checks passed for all sweep variants.")\n'
    )
    nb["cells"][7]["source"] = src(
        "# 5) Aggregate + rank + PASS/SOFT_PASS/FAIL\n"
        "if not all_rows:\n"
        "    raise RuntimeError('No sweep rows collected.')\n\n"
        "df = pd.DataFrame(all_rows)\n\n"
        "control_matches = df[df['prompt_version'] == CONTROL_VERSION]\n"
        "if control_matches.empty:\n"
        "    raise RuntimeError(f'Control version not found in sweep rows: {CONTROL_VERSION}')\n"
        "control_row = control_matches.iloc[0].to_dict()\n\n"
        "verdicts = []\n"
        "for _, r in df.iterrows():\n"
        "    row = r.to_dict()\n"
        "    if pass_rule(row, control_row):\n"
        "        verdicts.append('PASS')\n"
        "    elif soft_pass_rule(row, control_row):\n"
        "        verdicts.append('SOFT_PASS')\n"
        "    else:\n"
        "        verdicts.append('FAIL')\n\n"
        "df['verdict'] = verdicts\n\n"
        "# Ranking rule:\n"
        "# 1) highest coarse_macro_f1\n"
        "# 2) highest coarse_acc\n"
        "# 3) highest tag_mean_jaccard\n"
        "# 4) highest visibility_macro_f1\n"
        "ranked = df.sort_values(\n"
        "    by=['coarse_macro_f1', 'coarse_acc', 'tag_mean_jaccard', 'visibility_macro_f1'],\n"
        "    ascending=[False, False, False, False],\n"
        ").reset_index(drop=True)\n"
        "ranked['rank'] = ranked.index + 1\n\n"
        "best = ranked.iloc[0].to_dict()\n\n"
        "AGG_DIR = REPO_DIR / 'outputs' / 'stage3_vlm_baseline_runs' / f'{RUN_ID_PREFIX}_aggregate'\n"
        "AGG_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "agg_csv = AGG_DIR / 'prompt_sweep_comparison.csv'\n"
        "agg_md = AGG_DIR / 'prompt_sweep_comparison.md'\n"
        "ranked.to_csv(agg_csv, index=False)\n\n"
        "def to_markdown_table(df_in: pd.DataFrame) -> str:\n"
        "    cols = list(df_in.columns)\n"
        "    lines = []\n"
        "    lines.append('| ' + ' | '.join(cols) + ' |')\n"
        "    lines.append('| ' + ' | '.join(['---'] * len(cols)) + ' |')\n"
        "    for _, rr in df_in.iterrows():\n"
        "        vals = []\n"
        "        for c in cols:\n"
        "            v = rr[c]\n"
        "            if isinstance(v, float):\n"
        "                vals.append(f'{v:.6f}')\n"
        "            else:\n"
        "                vals.append(str(v))\n"
        "        lines.append('| ' + ' | '.join(vals) + ' |')\n"
        "    return '\\n'.join(lines)\n\n"
        "md_lines = [\n"
        "    '# Stage 3 Prompt Sweep Comparison (Coarse Recovery v7)',\n"
        "    '',\n"
        "    f'- control_version: `{CONTROL_VERSION}`',\n"
        "    f'- best_prompt_version: `{best[\"prompt_version\"]}`',\n"
        "    f'- best_run_id: `{best[\"run_id\"]}`',\n"
        "    f'- best_verdict: `{best[\"verdict\"]}`',\n"
        "    '',\n"
        "    to_markdown_table(ranked),\n"
        "]\n"
        "agg_md.write_text('\\n'.join(md_lines), encoding='utf-8')\n\n"
        "print('Aggregate CSV:', agg_csv)\n"
        "print('Aggregate MD:', agg_md)\n"
        "print('\\n=== BEST CANDIDATE ===')\n"
        "for k in ['prompt_version', 'run_id', 'verdict', 'coarse_macro_f1', 'coarse_acc', 'tag_mean_jaccard', 'visibility_macro_f1']:\n"
        "    print(f'{k}: {best[k]}')\n\n"
        "display(ranked)\n"
    )
    nb["cells"][8]["source"] = src(
        "# 6) Package sweep deliverables\n"
        'DELIVER_DIR = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        "if DELIVER_DIR.exists():\n"
        "    shutil.rmtree(DELIVER_DIR)\n"
        "DELIVER_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "for p in [AGG_DIR / 'prompt_sweep_comparison.csv', AGG_DIR / 'prompt_sweep_comparison.md']:\n"
        "    if p.exists():\n"
        '        dst = DELIVER_DIR / "aggregate" / p.name\n'
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        shutil.copy2(p, dst)\n\n"
        "run_artifacts_rel = [\n"
        '    "run_summary.json",\n'
        '    "config_snapshot.json",\n'
        '    "predictions_vlm_labels_v1.jsonl",\n'
        '    "parsed_predictions.jsonl",\n'
        '    "raw_responses.jsonl",\n'
        '    "sample_results.jsonl",\n'
        '    "failures.jsonl",\n'
        '    "eval/metrics.json",\n'
        '    "eval/confusion_coarse_class.csv",\n'
        '    "eval/confusion_visibility.csv",\n'
        '    "eval/review_table.csv",\n'
        '    "eval/failures.jsonl",\n'
        '    "eval/visuals/report.md",\n'
        '    "eval/visuals/kpi_overview.png",\n'
        '    "eval/visuals/confusion_coarse_class.png",\n'
        '    "eval/visuals/confusion_visibility.png",\n'
        '    "eval/visuals/visibility_errors_top.csv",\n'
        '    "eval/visuals/visibility_errors_gallery.png",\n'
        "]\n\n"
        "for pv in PROMPT_VERSIONS:\n"
        "    run_dir = run_dirs[pv]\n"
        "    for rel in run_artifacts_rel:\n"
        "        src_path = run_dir / rel\n"
        "        if src_path.exists():\n"
        '            dst = DELIVER_DIR / "runs" / pv / rel\n'
        "            dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "            shutil.copy2(src_path, dst)\n\n"
        'summary_md = DELIVER_DIR / "RESULT_SUMMARY.md"\n'
        "summary_md.write_text(\n"
        '    "\\n".join([\n'
        '        f"# Stage 3 Clean Coarse-Recovery Sweep Result: {RUN_ID_PREFIX}",\n'
        '        "",\n'
        '        "- model: Qwen/Qwen2.5-VL-3B-Instruct",\n'
        '        f"- backend: {BACKEND_MODE}",\n'
        '        f"- control_version: {CONTROL_VERSION}",\n'
        '        f"- prompt_count: {len(PROMPT_VERSIONS)}",\n'
        '        f"- best_prompt_version: {best[\'prompt_version\']}",\n'
        '        f"- best_run_id: {best[\'run_id\']}",\n'
        '        f"- best_verdict: {best[\'verdict\']}",\n'
        '        "",\n'
        '        f"Ground truth JSONL: {VAL_JSONL}",\n'
        '        f"Repo commit: {git_head}",\n'
        "    ]),\n"
        '    encoding="utf-8",\n'
        ")\n\n"
        'ARCHIVE_BASE = Path(f"/kaggle/working/stage3_deliverables_{RUN_ID_PREFIX}")\n'
        'ARCHIVE_PATH = shutil.make_archive(str(ARCHIVE_BASE), "gztar", root_dir=DELIVER_DIR)\n\n'
        'print("DELIVER_DIR:", DELIVER_DIR)\n'
        'print("ARCHIVE_PATH:", ARCHIVE_PATH)\n'
        'print("\\nFiles in deliverables:")\n'
        'for p in sorted(DELIVER_DIR.rglob("*")):\n'
        "    if p.is_file():\n"
        '        print("-", p.relative_to(DELIVER_DIR))\n'
    )
    nb["cells"][9]["source"] = src(
        "## Artifacts\n\n"
        "Per-run outputs: `outputs/stage3_vlm_baseline_runs/<RUN_ID_PREFIX>...`\n"
        "Aggregate files: `.../<RUN_ID_PREFIX>_aggregate/`\n"
        "Packed archive: `/kaggle/working/stage3_deliverables_stage3_qwen_val_v2_sweep_v7_coarse_clean.tar.gz`\n"
    )
    save_nb(NOTEBOOKS_DIR / "stage3_prompt_sweep_coarse_v7_clean.ipynb", nb)


def patch_stage4_notebook() -> None:
    nb = load_nb(NOTEBOOKS_DIR / "stage4_detector_to_vlm_kaggle_run.ipynb")
    nb["cells"][0]["source"] = src(
        "# Stage 4 Detector -> VLM (Kaggle, Clean Prompt Path)\n\n"
        "This notebook runs the full Stage 4 pipeline on predicted detector boxes with the clean `_nocroppath` Stage 3 prompt path.\n\n"
        "`val images -> detector -> pred crops -> Stage3 VLM -> Stage4 eval -> tar.gz`\n\n"
        "The final archive is saved to `/kaggle/working`.\n"
    )
    nb["cells"][1]["source"] = src(
        "from pathlib import Path\n"
        "import os\n"
        "import sys\n"
        "import subprocess\n"
        "import json\n\n"
        "DATASET_ROOT_CANDIDATES = [\n"
        "    Path('/kaggle/input/datasets/kostyaryazanov/idid-coco-v3'),\n"
        "    Path('/kaggle/input/idid-coco-v3'),\n"
        "]\n"
        "JSONL_REL = Path('stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl')\n\n"
        "DATA_ROOT = None\n"
        "for root in DATASET_ROOT_CANDIDATES:\n"
        "    if (root / JSONL_REL).exists():\n"
        "        DATA_ROOT = root\n"
        "        break\n\n"
        "if DATA_ROOT is None:\n"
        "    raise FileNotFoundError(\n"
        "        'DATA_ROOT not found: stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl'\n"
        "    )\n\n"
        "REPO_DIR = Path('/kaggle/working/vlm-for-insulator-defect-detection')\n"
        "REPO_URL = 'https://github.com/konstRyaz/vlm-for-insulator-defect-detection.git'\n"
        "RUN_NAME = 'stage4_detector_to_vlm_pred_val_kaggle'\n"
        "STAGE4_PROMPT_VERSION = 'qwen_vlm_labels_v1_prompt_v7f_flashover_unclear_to_unknown_nocroppath'\n\n"
        "print('DATA_ROOT:', DATA_ROOT)\n"
        "print('REPO_DIR :', REPO_DIR)\n"
        "print('RUN_NAME :', RUN_NAME)\n"
        "print('STAGE4_PROMPT_VERSION:', STAGE4_PROMPT_VERSION)\n"
    )
    nb["cells"][3]["source"] = src(
        "import yaml\n\n"
        "def pick_existing(candidates):\n"
        "    for c in candidates:\n"
        "        p = Path(c)\n"
        "        if p.exists():\n"
        "            return p\n"
        '    raise FileNotFoundError("Not found among candidates:\\n" + "\\n".join(str(x) for x in candidates))\n\n'
        "# Stage 3 GT labels\n"
        "stage3_gt_jsonl = pick_existing([\n"
        "    DATA_ROOT / 'stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl',\n"
        "    DATA_ROOT / 'outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl',\n"
        "    REPO_DIR / 'outputs/stage3_regrouped_v2/val/vlm_labels_v1_val_v2.annotated.jsonl',\n"
        "])\n\n"
        "# Detector assets root\n"
        "DETECTOR_DATASET_BASE = Path('/kaggle/input/datasets/kostyaryazanov/idid-detector-assets')\n"
        "DETECTOR_ASSETS_ROOT = (\n"
        "    DETECTOR_DATASET_BASE / 'idid-detector-assets'\n"
        "    if (DETECTOR_DATASET_BASE / 'idid-detector-assets').exists()\n"
        "    else DETECTOR_DATASET_BASE\n"
        ")\n\n"
        "if not DETECTOR_ASSETS_ROOT.exists():\n"
        '    raise FileNotFoundError(f"DETECTOR_ASSETS_ROOT not found: {DETECTOR_ASSETS_ROOT}")\n\n'
        "detector_input_dir = DETECTOR_ASSETS_ROOT / 'data/processed/val/images'\n"
        "coco_json = DETECTOR_ASSETS_ROOT / 'data/processed/val/annotations.json'\n"
        "weights_path = DETECTOR_ASSETS_ROOT / 'outputs/train/detector_baseline/best.pth'\n"
        "gt_jsonl = pick_existing([\n"
        "    DETECTOR_ASSETS_ROOT / 'analysis/stage4_gt_remapped.jsonl',\n"
        "    stage3_gt_jsonl,\n"
        "])\n\n"
        "for p in [detector_input_dir, coco_json, weights_path]:\n"
        "    if not p.exists():\n"
        '        raise FileNotFoundError(f"Missing path: {p}")\n\n'
        "print('stage3_gt_jsonl     :', stage3_gt_jsonl)\n"
        "print('gt_jsonl            :', gt_jsonl)\n"
        "print('DETECTOR_ASSETS_ROOT:', DETECTOR_ASSETS_ROOT)\n"
        "print('detector_input_dir  :', detector_input_dir)\n"
        "print('coco_json           :', coco_json)\n"
        "print('weights_path        :', weights_path)\n\n"
        "ceiling_run_dir = None\n"
        "ceiling_candidates = [\n"
        "    DATA_ROOT / 'outputs/stage3_vlm_baseline_runs/stage3_qwen_val_v2_clean_final',\n"
        "    REPO_DIR / 'outputs/stage3_vlm_baseline_runs/stage3_qwen_val_v2_clean_final',\n"
        "]\n"
        "for found in Path('/kaggle/input').rglob('stage3_qwen_val_v2_clean_final/predictions_vlm_labels_v1.jsonl'):\n"
        "    ceiling_candidates.append(found.parent)\n"
        "for c in ceiling_candidates:\n"
        "    if Path(c).exists():\n"
        "        ceiling_run_dir = Path(c)\n"
        "        break\n\n"
        "cfg = {\n"
        "    'version': 1,\n"
        "    'name': 'stage4_detector_to_vlm_pred_val_kaggle',\n"
        "    'stage4': {\n"
        "        'run_name': RUN_NAME,\n"
        "        'split': 'val',\n"
        "        'output_root': str(REPO_DIR / 'outputs/stage4'),\n"
        "    },\n"
        "    'detector': {\n"
        "        'experiment': 'detector_baseline',\n"
        "        'input_dir': str(detector_input_dir),\n"
        "        'config_path': str(REPO_DIR / 'src/configs/infer.yaml'),\n"
        "        'weights_path': str(weights_path),\n"
        "        'conf_threshold': 0.05,\n"
        "        'iou_threshold': 0.5,\n"
        "        'max_detections_per_image': 100,\n"
        "        'vis_samples': 8,\n"
        "        'device': 'auto',\n"
        "    },\n"
        "    'crop_export': {\n"
        "        'coco_json': str(coco_json),\n"
        "        'images_dir': str(detector_input_dir),\n"
        "        'include_categories': None,\n"
        "        'padding_ratio': 0.15,\n"
        "        'manifest_name': 'pred_manifest.jsonl',\n"
        "        'summary_name': 'pred_manifest_summary.json',\n"
        "        'limit': None,\n"
        "    },\n"
        "    'vlm': {\n"
        "        'backend_mode': 'qwen_hf',\n"
        "        'model_id': 'Qwen/Qwen2.5-VL-3B-Instruct',\n"
        "        'prompt_version': STAGE4_PROMPT_VERSION,\n"
        "        'stage3_runner_config': str(REPO_DIR / 'configs/pipeline/stage3_vlm_gt_baseline.yaml'),\n"
        "        'run_id': 'stage4_pred_vlm',\n"
        "    },\n"
        "    'analysis': {\n"
        "        'ground_truth_jsonl': str(gt_jsonl),\n"
        "        'match_iou_threshold': 0.5,\n"
        "        'good_crop_iou_threshold': 0.7,\n"
        "    },\n"
        "}\n\n"
        "if ceiling_run_dir is not None:\n"
        "    cfg['analysis']['ceiling_run_dir'] = str(ceiling_run_dir)\n\n"
        "cfg_path = REPO_DIR / 'configs/stage4_kaggle_run.yaml'\n"
        "cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False), encoding='utf-8')\n\n"
        "print('ceiling_run_dir     :', ceiling_run_dir)\n"
        "print('Config path:', cfg_path)\n"
        "print(cfg_path.read_text(encoding='utf-8'))\n"
    )
    nb["cells"][4]["source"] = src(
        "required_paths = [\n"
        "    REPO_DIR / 'scripts/run_stage4_detector_to_vlm.py',\n"
        "    REPO_DIR / 'configs/pipeline/stage3_vlm_gt_baseline.yaml',\n"
        "    detector_input_dir,\n"
        "    coco_json,\n"
        "    weights_path,\n"
        "    gt_jsonl,\n"
        "]\n"
        "for p in required_paths:\n"
        "    if not Path(p).exists():\n"
        '        raise FileNotFoundError(f"Missing required path: {p}")\n\n'
        "stage3_cfg = yaml.safe_load((REPO_DIR / 'configs/pipeline/stage3_vlm_gt_baseline.yaml').read_text(encoding='utf-8'))\n"
        "versions = stage3_cfg['prompt']['versions']\n"
        "selected_prompt = cfg['vlm']['prompt_version']\n"
        "if selected_prompt not in versions:\n"
        '    raise RuntimeError(f"Prompt version missing in Stage 3 config: {selected_prompt}")\n'
        "prompt_user_rel = versions[selected_prompt]['user_path']\n"
        "prompt_user_abs = REPO_DIR / prompt_user_rel\n"
        "if not prompt_user_abs.exists():\n"
        '    raise FileNotFoundError(f"Missing selected prompt file: {prompt_user_abs}")\n'
        "prompt_user_text = prompt_user_abs.read_text(encoding='utf-8')\n"
        'if "crop_path" in prompt_user_text:\n'
        '    raise RuntimeError("Selected Stage 4 prompt still exposes crop_path")\n\n'
        "print('Selected clean prompt file:', prompt_user_abs)\n"
        "print('Clean prompt preflight OK.')\n"
    )
    nb["cells"][5]["source"] = src(
        "os.chdir(REPO_DIR)\n\n"
        "run_cmd = [sys.executable, 'scripts/run_stage4_detector_to_vlm.py', '--config', str(cfg_path)]\n"
        "print('Running:', ' '.join(run_cmd))\n\n"
        "try:\n"
        "    subprocess.run(run_cmd, check=True)\n"
        "    print('Stage 4 run finished.')\n"
        "except subprocess.CalledProcessError as exc:\n"
        '    print(f"run_stage4 failed: code={exc.returncode}")\n'
        "    run_root = REPO_DIR / 'outputs/stage4' / RUN_NAME\n"
        "    vlm_run_id = cfg['vlm']['run_id']\n"
        "    log_paths = [\n"
        "        run_root / '01_detector' / 'run_detector.log',\n"
        "        run_root / '02_pred_crops' / 'run_export_pred_crops.log',\n"
        "        run_root / '03_vlm_pred' / 'run_stage3_pred.log',\n"
        "        run_root / '04_eval' / 'run_eval_stage4.log',\n"
        "    ]\n"
        "    for lp in log_paths:\n"
        "        if lp.exists():\n"
        '            print(f"\\n===== tail: {lp} =====")\n'
        "            txt = lp.read_text(encoding='utf-8', errors='replace')\n"
        "            print(txt[-5000:])\n"
        "    pred_vlm_run_dir = run_root / '03_vlm_pred' / vlm_run_id\n"
        "    pred_vlm_jsonl = pred_vlm_run_dir / 'predictions_vlm_labels_v1.jsonl'\n"
        "    if not pred_vlm_jsonl.exists():\n"
        "        pred_vlm_run_dir.mkdir(parents=True, exist_ok=True)\n"
        "        pred_vlm_jsonl.write_text('', encoding='utf-8')\n"
        '        print(f"Created empty file: {pred_vlm_jsonl}")\n'
        "    eval_cmd = [\n"
        "        sys.executable,\n"
        "        'scripts/eval_stage4_detector_to_vlm.py',\n"
        "        '--gt-jsonl', str(cfg['analysis']['ground_truth_jsonl']),\n"
        "        '--pred-manifest-jsonl', str(run_root / '02_pred_crops' / cfg['crop_export']['manifest_name']),\n"
        "        '--pred-vlm-run-dir', str(pred_vlm_run_dir),\n"
        "        '--detector-predictions-json', str(run_root / '01_detector' / 'predictions.json'),\n"
        "        '--coco-json', str(cfg['crop_export']['coco_json']),\n"
        "        '--match-iou-threshold', str(cfg['analysis']['match_iou_threshold']),\n"
        "        '--good-crop-iou-threshold', str(cfg['analysis']['good_crop_iou_threshold']),\n"
        "        '--output-dir', str(run_root / '04_eval'),\n"
        "    ]\n"
        "    if 'ceiling_run_dir' in cfg['analysis']:\n"
        "        eval_cmd.extend(['--ceiling-run-dir', str(cfg['analysis']['ceiling_run_dir'])])\n"
        "    print('Retry eval:', ' '.join(eval_cmd))\n"
        "    subprocess.run(eval_cmd, check=True)\n"
        "    print('Stage 4 eval finished (fallback path).')\n"
    )
    save_nb(NOTEBOOKS_DIR / "stage4_detector_to_vlm_kaggle_run.ipynb", nb)


def main() -> None:
    build_stage3_clean_onepass()
    build_stage3_clean_sweep()
    build_stage3_clean_micro()
    build_stage3_coarse_recovery_sweep()
    patch_stage4_notebook()
    print("Clean rerun notebooks generated.")


if __name__ == "__main__":
    main()
