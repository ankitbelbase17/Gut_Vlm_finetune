"""
Compute ROUGE-L, BLEU-1/2/4, METEOR, and hallucination-detection F1
from the files produced by step4_generate_predictions.py.

No GPU and no OpenAI key required.

Install once:
    pip install rouge-score nltk sacrebleu

Usage:
    python step4_eval_local.py \\
        --predictions results/step4/predictions.json \\
        --groundtruth results/step4/groundtruth.json \\
        --detections  results/step4/detections.json \\
        --test_json   ../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json

Can also be run locally after scp-ing the three JSON files off Clariden.
"""

import json, re, argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def compute_rouge_l(predictions, references):
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise SystemExit("pip install rouge-score")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_bleu(predictions, references, n):
    try:
        import sacrebleu
    except ImportError:
        raise SystemExit("pip install sacrebleu")
    refs_wrapped = [[r] for r in references]
    metric = sacrebleu.BLEU(max_ngram_order=n)
    result = metric.corpus_score(predictions, list(zip(*refs_wrapped)))
    return result.score / 100.0   # sacrebleu returns 0–100, normalise to 0–1


def compute_meteor(predictions, references):
    try:
        import nltk
    except ImportError:
        raise SystemExit("pip install nltk")
    for resource in ("punkt_tab", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    from nltk.translate.meteor_score import meteor_score
    scores = [
        meteor_score([ref.split()], pred.split())
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Hallucination-detection F1
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<(hallucinated|non-hallucinated)>")


def _parse_prediction_tags(text):
    """Extract ordered list of tag strings from model detection output."""
    return [m.group(1) for m in _TAG_RE.finditer(text)]


def _gt_labels_from_annotations(annotations):
    """
    Derive per-sentence ground truth labels from test.json annotations.
    Sorts spans by start offset (same order as step3_gut_vlm_data.py).
    """
    spans = sorted(annotations.values(), key=lambda x: x["start"])
    return [
        "hallucinated" if s["type"] == "incorrect" else "non-hallucinated"
        for s in spans
    ]


def compute_detection_f1(detections, test_data):
    """
    Token-level binary F1 for the sentence-level hallucination detection task.
    Aligns prediction and GT label lists by position (truncates to min length,
    treats extra/missing predicted labels as FP/FN respectively).
    """
    tp = fp = fn = 0
    skipped = 0
    for entry in detections:
        img_id = entry["image_path"]
        if img_id not in test_data:
            skipped += 1
            continue
        pred_labels = _parse_prediction_tags(entry.get("detection", ""))
        gt_labels   = _gt_labels_from_annotations(test_data[img_id]["annotations"])
        n = min(len(pred_labels), len(gt_labels))
        for p, g in zip(pred_labels[:n], gt_labels[:n]):
            if   p == "hallucinated" and g == "hallucinated":   tp += 1
            elif p == "hallucinated" and g != "hallucinated":   fp += 1
            elif p != "hallucinated" and g == "hallucinated":   fn += 1
        for p in pred_labels[n:]:           # extra predictions → FP
            if p == "hallucinated": fp += 1
        for g in gt_labels[n:]:             # missing predictions → FN
            if g == "hallucinated": fn += 1

    if skipped:
        print(f"[detection F1] skipped {skipped} images (not in test_json).")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="results/step4/predictions.json")
    p.add_argument("--groundtruth", default="results/step4/groundtruth.json")
    p.add_argument("--detections",  default="results/step4/detections.json")
    p.add_argument("--test_json",
                   default="../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json")
    p.add_argument("--skip_meteor", action="store_true",
                   help="Skip METEOR if WordNet download is unavailable")
    return p.parse_args()


def main():
    args = parse_args()

    preds = load_json(args.predictions)
    gts   = load_json(args.groundtruth)
    dets  = load_json(args.detections) if Path(args.detections).exists() else []

    # Build image_id → ground truth correction map
    gt_map = {}
    for e in gts:
        for img in e["images"]:
            gt_map[img] = e["response"]

    # Align predictions with references
    pred_texts, ref_texts = [], []
    for e in preds:
        img_id = e["image_path"]
        if img_id in gt_map:
            pred_texts.append(e["response"])
            ref_texts.append(gt_map[img_id])

    n = len(pred_texts)
    if n == 0:
        raise SystemExit("No matched predictions found — check file paths.")
    print(f"Evaluating {n} images ...\n")

    rouge_l = compute_rouge_l(pred_texts, ref_texts)
    bleu1   = compute_bleu(pred_texts, ref_texts, 1)
    bleu2   = compute_bleu(pred_texts, ref_texts, 2)
    bleu4   = compute_bleu(pred_texts, ref_texts, 4)

    if args.skip_meteor:
        meteor = None
    else:
        try:
            meteor = compute_meteor(pred_texts, ref_texts)
        except Exception as ex:
            print(f"[METEOR skipped] {ex}")
            meteor = None

    # Detection F1
    test_data = load_json(args.test_json) if Path(args.test_json).exists() else {}
    if dets and test_data:
        prec, rec, f1_det = compute_detection_f1(dets, test_data)
    else:
        prec = rec = f1_det = None

    # Print summary table
    print("=" * 52)
    print(f"{'Metric':<28} {'Our Mobile-O':>12}")
    print("-" * 52)
    print(f"{'ROUGE-L':<28} {rouge_l*100:>11.2f}%")
    print(f"{'BLEU-1':<28} {bleu1*100:>11.2f}%")
    print(f"{'BLEU-2':<28} {bleu2*100:>11.2f}%")
    print(f"{'BLEU-4':<28} {bleu4*100:>11.2f}%")
    if meteor is not None:
        print(f"{'METEOR':<28} {meteor*100:>11.2f}%")
    else:
        print(f"{'METEOR':<28} {'(skipped)':>12}")
    if f1_det is not None:
        print("-" * 52)
        print(f"{'Detection Precision':<28} {prec*100:>11.2f}%")
        print(f"{'Detection Recall':<28} {rec*100:>11.2f}%")
        print(f"{'Detection F1':<28} {f1_det*100:>11.2f}%")
    print("=" * 52)
    print("\nNote: QAAS and R-Sim require an OpenAI key.")
    print("Run step4_eval_qaas.py and step4_eval_rsim.py for those.")


if __name__ == "__main__":
    main()
