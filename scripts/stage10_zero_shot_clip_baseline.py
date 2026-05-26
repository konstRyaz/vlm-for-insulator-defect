import argparse
from pathlib import Path
from collections import Counter

import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import CLIPProcessor, CLIPModel

CLASS_PROMPTS = {
    "insulator_ok": [
        "a photo of a normal high-voltage insulator without visible defects",
        "an intact electrical insulator with clean normal surface",
    ],
    "defect_flashover": [
        "a photo of an electrical insulator with flashover burn marks and dark arc traces",
        "an insulator damaged by flashover with carbonization and dark surface track",
    ],
    "defect_broken": [
        "a photo of a broken electrical insulator with missing or cracked parts",
        "an insulator with mechanical damage, broken shed or missing fragment",
    ],
}

def collect_val(data_root: Path):
    val_root = data_root / "val" / "crops" / "val"
    rows = []
    for cls_dir in sorted(val_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpeg")):
            rows.append({"record_id": img.stem, "gt": cls, "path": str(img)})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = collect_val(data_root)
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    labels = list(CLASS_PROMPTS.keys())
    text_prompts = []
    prompt_to_label = []
    for label, prompts in CLASS_PROMPTS.items():
        for p in prompts:
            text_prompts.append(p)
            prompt_to_label.append(label)

    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    preds = []
    for _, row in df.iterrows():
        image = Image.open(row["path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ text_features.T).squeeze(0).detach().cpu().numpy()

        # average prompts per class
        scores = {}
        for label in labels:
            idx = [i for i, l in enumerate(prompt_to_label) if l == label]
            scores[label] = float(np.mean(sims[idx]))
        pred = max(scores, key=scores.get)
        preds.append({**row.to_dict(), "pred": pred, **{f"score_{k}": v for k, v in scores.items()}})

    pred_df = pd.DataFrame(preds)
    acc = accuracy_score(pred_df["gt"], pred_df["pred"])
    macro_f1 = f1_score(pred_df["gt"], pred_df["pred"], average="macro", zero_division=0)

    pred_df.to_csv(out / "stage10_clip_zero_shot_predictions.csv", index=False)
    pd.DataFrame([{"accuracy": acc, "macro_f1": macro_f1, "n": len(pred_df), "model": args.model}]).to_csv(out / "stage10_clip_zero_shot_metrics.csv", index=False)

    print("class counts:", Counter(pred_df["gt"]))
    print("accuracy:", acc, "macro_f1:", macro_f1)
    print(classification_report(pred_df["gt"], pred_df["pred"], zero_division=0))

if __name__ == "__main__":
    main()
