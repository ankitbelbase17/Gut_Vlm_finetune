"""
Generate hallucination-correction predictions on all 366 Gut-VLM test images.

Run from ~/Mobile-O on Clariden:
    python step4_generate_predictions.py \\
        --model_path checkpoints/vlm_gutvlm_hal/epoch_4 \\
        --test_json  ../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json \\
        --images_dir kvasir-v2-flat \\
        --output_dir results/step4

Saves (written incrementally — safe to Ctrl-C and resume with --resume):
    results/step4/predictions.json   — [{"image_path", "response"}]  corrected captions
    results/step4/groundtruth.json   — [{"images", "response"}]       gold corrections (R-Sim ref)
    results/step4/detections.json    — [{"image_path", "detection"}]  per-sentence tags (detection F1)
"""

import sys, os, json, argparse
from pathlib import Path

MOBILEO_REPO = os.environ.get("MOBILEO_PATH", ".")
if MOBILEO_REPO not in sys.path:
    sys.path.insert(0, MOBILEO_REPO)

from tqdm import tqdm
from inference import load_model, detect_hallucinations


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="e.g. checkpoints/vlm_gutvlm_hal/epoch_4")
    p.add_argument("--test_json",
                   default="../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json")
    p.add_argument("--images_dir", default="kvasir-v2-flat",
                   help="Flat kvasir-v2 directory where uuid.jpg files live")
    p.add_argument("--output_dir", default="results/step4")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--resume", action="store_true",
                   help="Skip images already present in predictions.json")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred_path = out / "predictions.json"
    gt_path   = out / "groundtruth.json"
    det_path  = out / "detections.json"

    with open(args.test_json) as f:
        test_data = json.load(f)

    images_dir = Path(args.images_dir)

    # Build complete ground truth once (all 366 images regardless of resume state)
    groundtruth = [
        {"images": [img_id], "response": entry["corrections"]}
        for img_id, entry in test_data.items()
    ]
    _save(gt_path, groundtruth)
    print(f"Ground truth saved: {gt_path}")

    # Load existing results if resuming
    predictions, detections, done = [], [], set()
    if args.resume and pred_path.exists():
        with open(pred_path) as f:
            predictions = json.load(f)
        with open(det_path) as f:
            detections = json.load(f)
        done = {e["image_path"] for e in predictions}
        print(f"[resume] {len(done)}/{len(test_data)} already done, skipping.")

    model, tokenizer, image_processor = load_model(args.model_path, args.device)

    items   = [(img_id, e) for img_id, e in test_data.items() if img_id not in done]
    skipped = 0

    for img_id, entry in tqdm(items, desc="Generating predictions"):
        img_path = images_dir / img_id
        if not img_path.exists():
            print(f"[skip] image not found: {img_path}")
            skipped += 1
            continue

        try:
            detection, correction = detect_hallucinations(
                model, tokenizer, image_processor,
                str(img_path),
                entry["original_text"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as ex:
            print(f"[error] {img_id}: {ex}")
            skipped += 1
            continue

        predictions.append({"image_path": img_id, "response": correction})
        detections.append({
            "image_path":    img_id,
            "detection":     detection,
            "original_text": entry["original_text"],
        })

        # Incremental save so we can Ctrl-C and resume safely
        _save(pred_path, predictions)
        _save(det_path,  detections)

    print(f"\nDone. {len(predictions)} predictions, {skipped} skipped.")
    print(f"Outputs:\n  {pred_path}\n  {gt_path}\n  {det_path}")


if __name__ == "__main__":
    main()
