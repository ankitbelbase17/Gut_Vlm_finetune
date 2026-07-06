"""
R-Sim (Report Similarity) evaluation — GPT-4o rates how well our corrected
caption matches the gold correction on a 1–5 scale.

Requires:
    OPENAI_API_KEY env var  (or --api_key flag)
    pip install openai

Approximate OpenAI cost: ~370 GPT-4o calls × $0.015 ≈ $5.50

Usage:
    export OPENAI_API_KEY=sk-...
    python step4_eval_rsim.py \\
        --predictions results/step4/predictions.json \\
        --groundtruth results/step4/groundtruth.json \\
        --output_dir  results/step4/rsim

Can run locally after downloading predictions.json and groundtruth.json from Clariden.
"""

import json, os, argparse, time
from pathlib import Path


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _gpt4o(client, prompt, max_tokens=200):
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [GPT-4o retry {attempt+1}] {e} — waiting {wait}s")
            time.sleep(wait)
    return ""


RATING_PROMPT_TEMPLATE = """
You are an expert in medical image analysis.
Compare two descriptions of a gastrointestinal endoscopy image and rate their similarity.

Rating scale:
  5 - (Very Good)  Nearly identical, all findings correctly described.
  4 - (Good)       Minor differences, clinically acceptable.
  3 - (Alright)    Some differences, but overall meaning preserved.
  2 - (Not Good)   Significant differences affecting diagnosis.
  1 - (Poor)       Completely incorrect or misleading.

Consider: anatomical landmark, abnormality color/location, polyps, instruments,
inflammation, bleeding, infection, foreign bodies.

Description 1 (model output):
{desc1}

Description 2 (ground truth):
{desc2}

Return ONLY:
- Match?: Yes/No
- Similarity Rating: <1-5>
- Brief Justification: <one sentence>
"""


def score_pair(client, pred_text, gt_text):
    prompt = RATING_PROMPT_TEMPLATE.format(desc1=pred_text, desc2=gt_text)
    raw = _gpt4o(client, prompt)
    # Extract the first single digit 1-5
    score = next(
        (int(tok) for tok in raw.split() if tok.isdigit() and 1 <= int(tok) <= 5),
        None
    )
    return score, raw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="results/step4/predictions.json")
    p.add_argument("--groundtruth", default="results/step4/groundtruth.json")
    p.add_argument("--output_dir",  default="results/step4/rsim")
    p.add_argument("--api_key",     default=os.environ.get("OPENAI_API_KEY", ""))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY env var or pass --api_key.")

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "rsim_results.json"

    preds = load_json(args.predictions)
    gts   = load_json(args.groundtruth)

    gt_map = {}
    for e in gts:
        for img in e["images"]:
            gt_map[img] = e["response"]

    # Resume support: reload any already-scored entries
    done = {}
    if results_path.exists():
        partial = load_json(results_path)
        done = {e["image"]: e["score"] for e in partial
                if isinstance(e, dict) and "image" in e and "score" in e}
        print(f"[resume] {len(done)} images already scored.")

    results = []
    scores  = []
    total   = len(preds)

    for i, entry in enumerate(preds, 1):
        img_id   = entry["image_path"]
        pred_txt = entry["response"]
        gt_txt   = gt_map.get(img_id, "")

        if not gt_txt:
            print(f"  [skip] no GT for {img_id}")
            continue

        if img_id in done:
            s = done[img_id]
            results.append({"image": img_id, "score": s})
            scores.append(s)
            continue

        score, raw = score_pair(client, pred_txt, gt_txt)
        if score is None:
            print(f"  [warn] could not parse score for {img_id}: {raw[:80]}")
            score = 0
        results.append({"image": img_id, "score": score, "justification": raw})
        scores.append(score)

        if i % 10 == 0:
            avg = sum(scores) / len(scores)
            print(f"  {i}/{total} — running R-Sim avg: {avg:.3f}/5")
            save_json(results_path, results)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    results.append({"average_score": avg_score})
    save_json(results_path, results)

    # Score distribution
    dist = {s: scores.count(s) for s in range(1, 6)}

    print("\n" + "=" * 45)
    print(f"  R-Sim (avg): {avg_score:.3f} / 5.0")
    print("=" * 45)
    print("\nScore distribution:")
    for s, n in sorted(dist.items()):
        bar = "█" * n
        print(f"  {s}: {n:3d}  {bar}")
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
