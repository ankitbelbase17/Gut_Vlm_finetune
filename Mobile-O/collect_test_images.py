"""
Copy the 366 Gut-VLM test images from kvasir-v2-flat into one directory.

Run on Clariden from ~/Mobile-O:
    python collect_test_images.py

Adjust the paths below if your layout differs.
"""

import json
import shutil
from pathlib import Path

TEST_JSON   = Path("../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json")
IMAGES_SRC  = Path("/iopsstor/scratch/cscs/dbartaula/FT/kvasir-v2-flat")
OUT_DIR     = Path("data/gut_vlm/test_images")

OUT_DIR.mkdir(parents=True, exist_ok=True)

img_ids = list(json.loads(TEST_JSON.read_text()).keys())
print(f"Found {len(img_ids)} images in test.json")

ok, missing = 0, []
for img_id in img_ids:
    src = IMAGES_SRC / img_id
    dst = OUT_DIR / img_id
    if src.exists():
        shutil.copy2(src, dst)
        ok += 1
    else:
        missing.append(img_id)

print(f"Copied  : {ok}")
print(f"Missing : {len(missing)}")
if missing:
    print("First few missing:", missing[:5])
print(f"Output  : {OUT_DIR.resolve()}")
