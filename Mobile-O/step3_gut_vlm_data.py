"""
Step 3 (data prep): Convert the Gut-VLM hallucination-aware annotations into
Mobile-O's multi-turn `conversations` JSONL format.

Source: Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/{train,test}.json
  {
    "<uuid>.jpg": {
      "original_text": "<VLM-generated report, possibly hallucinated>",
      "corrections":    "<expert-corrected report>",
      "annotations": {
        "<start>-<end>": {"text": "<sentence>", "type": "correct"|"incorrect"}
      }
    },
    ...
  }

This mirrors the 2-turn structure used in the paper's own
`training_style/hallucinated_aware_train.json` (ms-swift format), translated
into Mobile-O's {"from": "human"/"gpt", "value": ...} turns:

  human: <image> Caption: <original_text>\n\n Can you detect which sentences
         are hallucinated in the given caption?
  gpt:   <sentence_1> <non-hallucinated>\n<sentence_2> <hallucinated>\n...
  human: \n Can you please correct any hallucinated sentences and generate a
         modified response?
  gpt:   \n Modified caption: <corrections>

Annotation spans are sorted by `start` so sentence order matches the
original report regardless of dict insertion order in the source JSON.

Images are NOT bundled in the Hallucination-Aware-VLM repo (see its README) --
you must separately download kvasir-v2.zip and pass its image directory via
--images_dir. Filenames in the annotation JSON (e.g. "53a543e6-....jpg") must
match files in that directory.

Run:
    cd ~/Mobile-O
    python step3_gut_vlm_data.py \
        --hal_repo /path/to/Hallucination-Aware-VLM \
        --images_dir /path/to/kvasir-v2/images \
        --out_dir data/gut_vlm
"""

import argparse
import json
from pathlib import Path

SYSTEM_PREFIX = (
    "A chat between a user and an artificial intelligence assistant expert "
    "in Gastrointestinal endoscopic images. The assistant is tasked with "
    "detecting hallucinated sentences in a given caption. Hallucination "
    "occurs when the caption is incorrect, misleading, or non-existent "
    "information that is not grounded in the input image or context. "
    "Hallucinations include factual errors, misidentification of anatomy, "
    "false detection of abnormalities, incorrect reasoning, and nonexistent "
    "instruments or conditions like bleeding, infection, or inflammation.\n\n"
)

TAG = {"correct": "<non-hallucinated>", "incorrect": "<hallucinated>"}


def build_record(img_id, entry, images_dir):
    img_path = images_dir / img_id
    if not img_path.exists():
        return None

    spans = sorted(entry["annotations"].values(), key=lambda s: s["start"])
    detect_lines = [
        f"{s['text']} {TAG.get(s['type'], '<non-hallucinated>')}" for s in spans
    ]
    detect_response = "\n".join(detect_lines)

    return {
        "image": str(img_path.resolve()),
        "conversations": [
            {
                "from": "human",
                "value": (
                    SYSTEM_PREFIX
                    + f"<image>\nCaption: {entry['original_text']}\n\n"
                    "Can you detect which sentences are hallucinated in the "
                    "given caption?"
                ),
            },
            {"from": "gpt", "value": detect_response},
            {
                "from": "human",
                "value": (
                    "Can you please correct any hallucinated sentences and "
                    "generate a modified response?"
                ),
            },
            {"from": "gpt", "value": f"Modified caption: {entry['corrections']}"},
        ],
    }


def convert(src_json, images_dir, out_jsonl):
    data = json.loads(Path(src_json).read_text())
    records, skipped = [], 0
    for img_id, entry in data.items():
        rec = build_record(img_id, entry, images_dir)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"{src_json} -> {out_jsonl}: {len(records)} written, {skipped} skipped (image not found)")
    return records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hal_repo", required=True, help="Path to Hallucination-Aware-VLM clone")
    p.add_argument("--images_dir", required=True, help="Path to raw Kvasir-v2 images (uuid filenames)")
    p.add_argument("--out_dir", default="data/gut_vlm")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hal_repo = Path(args.hal_repo)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    split_dir = hal_repo / "dataset" / "Gut-VLM" / "train_test_split"
    train_recs = convert(split_dir / "train.json", images_dir, out_dir / "train.jsonl")
    test_recs = convert(split_dir / "test.json", images_dir, out_dir / "test.jsonl")

    if train_recs:
        print("\nSample record:")
        print(json.dumps(train_recs[0], indent=2)[:1500])
