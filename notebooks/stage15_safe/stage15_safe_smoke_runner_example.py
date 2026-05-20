# This is a Kaggle notebook cell-style script.
# Paste into Kaggle notebook after copying scripts into the repo.

from pathlib import Path
import subprocess, sys, json, os

repo = Path("/kaggle/working/repo")
os.chdir(repo)

REF = "/kaggle/input/datasets/kostyaryazanov/stage10-vlm-topk-manifest-bundle/stage10/vlm_topk_inference_manifest/stage10_vlm_eval_reference.csv"
IMG_ROOT = "/kaggle/input/datasets/kostyaryazanov/stage10-vlm-topk-manifest-bundle/stage10/vlm_topk_inference_manifest"
OUT_ROOT = "/kaggle/working/stage15_safe_smoke"

def run(cmd):
    print("CMD:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, text=True, capture_output=True)
    print("STDOUT:", p.stdout[-4000:])
    print("STDERR:", p.stderr[-4000:])
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return p

run([
    sys.executable, "scripts/stage15_make_smoke_manifests.py",
    "--reference-csv", REF,
    "--image-root", IMG_ROOT,
    "--out-root", OUT_ROOT,
    "--limit", "8",
])

for task, subdir in [("e02", "E02_flashover"), ("e05", "E05_claim"), ("e06", "E06_multiview")]:
    run([
        sys.executable, "scripts/stage15_safe_vlm_runner.py",
        "--task", task,
        "--manifest-csv", f"{OUT_ROOT}/{subdir}/manifest.csv",
        "--out-dir", f"{OUT_ROOT}/{subdir}/run",
        "--model-name", "Qwen/Qwen2-VL-2B-Instruct",
        "--device", "auto",
        "--limit", "5",
    ])
    print(json.loads(Path(f"{OUT_ROOT}/{subdir}/run/run_status.json").read_text()))
