"""
Step 1: Download Kvasir-VQA and convert to the conversation format
that Mobile-O's own training code (post_train.py) uses.

Mobile-O's format per sample (from post_train.py __getitem__):
  conversations = [
      {"from": "human", "value": "<image>\n<question>"},
      {"from": "gpt",   "value": "<answer>"},
  ]
  und_image = PIL Image  (the understanding image)

We store this as JSONL with image paths so our custom dataset can load it.

Output:
  data/images/          <- kvasir images as .jpg
  data/smoke_test.jsonl <- 200 samples for smoke test
  data/train.jsonl      <- all ~58k QA pairs

Run: python step1_prepare_data.py
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR      = Path("data")
IMG_DIR       = DATA_DIR / "images"
TRAIN_JSONL   = DATA_DIR / "train.jsonl"
SMOKE_JSONL   = DATA_DIR / "smoke_test.jsonl"
SMOKE_SAMPLES = 200

IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Kvasir-VQA from HuggingFace...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")
print(f"Total QA pairs: {len(ds)}")

print("Saving images...")
seen = set()
for row in tqdm(ds, desc="images"):
    img_id = row["img_id"]
    if img_id not in seen:
        row["image"].save(str(IMG_DIR / f"{img_id}.jpg"))
        seen.add(img_id)
print(f"Saved {len(seen)} unique images")

print("Building JSONL...")
records = []
for row in tqdm(ds, desc="QA pairs"):
    img_path = str((IMG_DIR / f"{row['img_id']}.jpg").resolve())
    records.append({
        "image": img_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{row['question']}"},
            {"from": "gpt",   "value": row["answer"]},
        ]
    })

with open(TRAIN_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"Full train: {len(records)} samples → {TRAIN_JSONL}")

with open(SMOKE_JSONL, "w") as f:
    for r in records[:SMOKE_SAMPLES]:
        f.write(json.dumps(r) + "\n")
print(f"Smoke test: {SMOKE_SAMPLES} samples → {SMOKE_JSONL}")

print("\nSample:")
print(json.dumps(records[0], indent=2))