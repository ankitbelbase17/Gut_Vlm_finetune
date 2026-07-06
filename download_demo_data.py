"""
Download 6 curated VQA demo pairs from Kvasir-VQA using streaming mode.

Saves to:  demo_data/vqa/
  images/    JPEG files named demo_1.jpg ... demo_6.jpg
  questions.json  {image, question, expected_answer, category, source}

Run from the project root:
    pip install datasets pillow
    python download_demo_data.py
"""

import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR   = Path("demo_data/vqa")
IMG_DIR   = OUT_DIR / "images"
QUESTIONS = OUT_DIR / "questions.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)

# Real source names found in the dataset
TARGET_SOURCES = {
    "Polyps":             "Polyp detection",
    "Esophagitis":        "Esophagitis",
    "Ulcerative Colitis": "Ulcerative colitis",
    "Instrument":         "Instrument presence",
}

# Skip these — trivial or position-grid answers, not good for a live demo
SKIP_ANSWERS  = {"none", "not relevant", "yes", "no", "n/a", "unknown", "normal"}
SKIP_QUESTIONS = {"where in the image", "is this finding easy", "how many images"}

def is_good(row):
    ans = row["answer"].strip().lower()
    q   = row["question"].strip().lower()
    if ans in SKIP_ANSWERS or len(ans) < 4:
        return False
    if ans.count(";") > 3:          # position-grid answer (too many semicolons)
        return False
    if any(k in q for k in SKIP_QUESTIONS):
        return False
    return True

def score(row):
    """Longer, more descriptive answers score higher."""
    return len(row["answer"])

print("Scanning Kvasir-VQA in streaming mode (~58k records) ...")
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw", streaming=True)

# best[source] = highest-scoring good row seen so far
best     = {}
seen     = 0
last_src = None

for row in ds:
    seen += 1
    src = row.get("source", "")

    if src != last_src:
        print(f"  [record {seen:>6}] category: {src!r}")
        last_src = src

    if src not in TARGET_SOURCES:
        continue
    if not is_good(row):
        continue

    if src not in best or score(row) > score(best[src]):
        best[src] = row

print(f"\nScanned {seen} records. Found {len(best)} categories: {list(best.keys())}")

DEMO_ORDER = ["Polyps", "Esophagitis", "Ulcerative Colitis", "Instrument"]

records = []
for rank, src in enumerate(DEMO_ORDER, start=1):
    if src not in best:
        print(f"  MISSING: {src}")
        continue
    row      = best[src]
    label    = TARGET_SOURCES[src]
    img_name = f"demo_{rank}.jpg"
    img_path = IMG_DIR / img_name

    row["image"].convert("RGB").save(img_path, "JPEG", quality=92)

    records.append({
        "demo_id":         rank,
        "image":           str(img_path),
        "question":        row["question"],
        "expected_answer": row["answer"],
        "category":        label,
        "source":          src,
        "original_img_id": row.get("img_id", ""),
    })

QUESTIONS.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nSaved {len(records)} demo pairs to {OUT_DIR}/")
print()
print("=" * 62)
print("CUE CARDS FOR LIVE DEMO")
print("=" * 62)
for r in records:
    print(f"\n[Demo {r['demo_id']}]  {r['category']}")
    print(f"  Image    : demo_data/vqa/images/{Path(r['image']).name}")
    print(f"  Question : {r['question']}")
    print(f"  Expected : {r['expected_answer']}")
