"""
QAAS (Question Answering Accuracy Score) evaluation — paper's primary metric.

Full pipeline:
  1. Convert each corrected caption to 12 structured QA answers (GPT-4o)
  2. Normalize our answers to canonical categories (GPT-4o)
  3. Normalize ground-truth VQA answers to categories (local lookup — no extra GPT-4o calls)
  4. Compute per-question accuracy across all 366 test images

Each intermediate file is cached so you can restart mid-run without re-spending API budget.

Requires:
    OPENAI_API_KEY env var  (or --api_key flag)
    pip install openai

Approximate OpenAI cost: ~750 GPT-4o calls × $0.015 ≈ $11

Usage:
    export OPENAI_API_KEY=sk-...
    python step4_eval_qaas.py \\
        --predictions results/step4/predictions.json \\
        --vqa_gt      ../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/VQA_format_testset_only.json \\
        --output_dir  results/step4/qaas

Paper reference (Table 2):
    Hallucination-aware FT (Qwen2-VL):  90.89% QAAS
    Standard FT (Qwen2-VL):             83.07% QAAS
"""

import json, os, ast, argparse, time
from pathlib import Path


# ---------------------------------------------------------------------------
# The 12 QAAS questions (verbatim from paper)
# ---------------------------------------------------------------------------

QUESTIONS = [
    "Which anatomical landmark or organ does the image belong to among colon, cecum, pylorus, or z-line? Just select one of the following if it's present in the context text. If not, return N/A.",
    "If the color of the anatomical landmark is explicitly mentioned, just answer in a single or two words maximum. If it's not mentioned, return N/A.",
    "If the location or position of the anatomical landmark is explicitly mentioned, where is it located? Just answer with a location or position describing word. Absolutely limit your answer to a single or two words at maximum. If it's not mentioned, return N/A.",
    "Is there any abnormality present in the image? If yes, return Yes. If not, return No.",
    "If the color of the abnormality is explicitly mentioned, just answer in a single or two words maximum. If it's not mentioned, return N/A.",
    "If the location or position of the abnormality is explicitly mentioned, just answer in a single or two words maximum. If it's not mentioned, return N/A.",
    "Are there any polyps present? Just answer how many polyps are there? Possible answers are (Zero, Single, Multiple). If it's not mentioned, return N/A.",
    "Are there any instruments visible in the image? If yes, return Yes. If not, return No. If it's not mentioned, return N/A.",
    "Are there any signs of inflammation present in the image? If yes, return Yes. If not, return No. If it's not mentioned, return N/A.",
    "Is there evidence of bleeding in the image? If yes, return Yes. If not, return No. If it's not mentioned, return N/A.",
    "Are there any foreign bodies present in the image? If yes, return Yes. If not, return No. If it's not mentioned, return N/A.",
    "Are there any signs of infection present in the image? If yes, return Yes. If not, return No. If it's not mentioned, return N/A.",
]

# Question index (1-based) → category type
Q_CATEGORY = {
    1: "landmark", 2: "color", 3: "location", 4: "yes_no",
    5: "color",    6: "location", 7: "polyp",
    8: "yes_no", 9: "yes_no", 10: "yes_no", 11: "yes_no", 12: "yes_no",
}

# Category maps for local GT normalization (from parse_vqa.py)
CATEGORIES = {
    "color": {
        "red":    ["red","reddish","reddish pink","reddish brown","dark red","light red","pinkish red","red pink","pink orange","red orange","pinkish orange","pinkish brown","brownish red","brick red","scarlet","crimson","maroon","rose red"],
        "pink":   ["pink","pinkish","pinkish yellow","pinkish brown","pinkish orange","light pink","pink light","reddish pink","red pink","yellowish pink","light pinkish","dark pinkish","pinkish red","blush pink","rose pink","coral pink","hot pink","pastel pink","peach pink","baby pink","flesh","light flesh","dark flesh","skin tone","peach flesh","flesh colored","beige flesh","rosy flesh","normal","flesh tone","flesh color","flesh shade","normal color","normal shade","normal hue","normal tone","normal pigment","normal coloration","light","lighter","lightest","lightest pink","lightest shade","same color","typical color","typical shade","typical hue","typical tone","similar color","similar shade","similar hue","similar tone","similar pigment"],
        "orange": ["orange","yellowish orange","orange yellow","yellow orange","pinkish orange","dark orange","light orange","brownish orange","tangerine","amber","burnt orange","apricot","coral orange","golden orange","peach orange"],
        "yellow": ["yellow","yellowish","yellowish pink","yellowish brown","brownish yellow","yellowish white","white yellow","light yellow","dark yellow","light yellowish","dark yellowish","pinkish yellow","yellow-orange","golden yellow","mustard yellow","lemon yellow","sunflower yellow","pale yellow","cream yellow"],
        "brown":  ["brown","brownish","brownish yellow","brownish red","brownish pink","brownish orange","reddish brown","yellowish brown","light brown","light brownish","dark brownish","pinkish brown","tan","beige","chocolate brown","coffee brown","caramel brown","rust brown"],
        "blue":   ["blue","bluish","dark blue","blue dark","bluish green","light bluish","dark bluish","greenish blue","blue green","green bluish","sky blue","navy blue","royal blue","aqua blue","teal blue","light blueish","dark blueish","baby blue","electric blue","cobalt blue","cerulean blue","blueish","blueish green","green blueish","blue-colored","bluish hue","blueish hue","blueish color","blueish shade","blueish tone","blue-green","green-blue"],
        "green":  ["green","greenish","bluish green","green bluish","greenish blue","blue green","light green","light greenish","dark greenish","lime green","forest green","emerald green","olive green","mint green","sea green","sage green","pastel green","blueish green"],
        "white":  ["white","off white","ivory","cream","light gray"],
        "dark":   ["black","charcoal","gray","dark gray","light gray","dark"],
        "n/a":    ["n/a","none","not mentioned","not specified","not provided"],
    },
    "location": {
        "center":      ["center","middle","centre","centrally","central","mid center","center mid","mid","midpoint","centered"],
        "left":        ["left","center left","left center","top left","upper left","bottom left","lower left","mid left","left mid","left top","left bottom","9'o clock","9 o'clock"],
        "right":       ["right","center right","right center","top right","upper right","bottom right","lower right","mid right","right mid","right top","right bottom","3'o clock","3 o'clock"],
        "top":         ["top","upper","above","top center","upper center","center top","top left","upper left","top right","upper right","mid top","top mid","12'o clock","12 o'clock"],
        "bottom":      ["bottom","lower","below","bottom center","lower center","center bottom","bottom left","lower left","bottom right","lower right","mid bottom","bottom mid","6'o clock","6 o'clock"],
        "surrounding": ["surrounding","around","near","nearby","peripheral","adjacent","bordering","encompassing","encircling","flanking"],
        "background":  ["scattered","throughout","background","distributed","dispersed","spread out","sporadic","all over","widespread","consistent","surface"],
        "n/a":         ["n/a","none","not mentioned","not specified","not provided"],
    },
    "landmark": {
        "colon":   ["colon"],
        "cecum":   ["cecum"],
        "z-line":  ["z-line","z line"],
        "pylorus": ["pylorus"],
        "n/a":     ["n/a","none","not mentioned","not specified","not provided"],
    },
    "polyp": {
        "zero":     ["zero","0"],
        "single":   ["single","1"],
        "multiple": ["multiple","more than one","several","numerous","many"],
        "n/a":      ["n/a","none","not mentioned","not specified","not provided"],
    },
    "yes_no": {
        "yes": ["yes"],
        "no":  ["no"],
        "n/a": ["n/a","none","not mentioned","not specified","not provided"],
    },
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# GPT-4o helper (with retry)
# ---------------------------------------------------------------------------

def _gpt4o(client, prompt, max_tokens=400):
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


# ---------------------------------------------------------------------------
# Step 1: Caption → 12 raw answers  (GPT-4o)
# ---------------------------------------------------------------------------

def caption_to_vqa(client, text):
    if not text.strip():
        return ["N/A"] * 12
    prompt = (
        "You are given this text from a medical image report:\n\n"
        f"------\n{text}\n------\n\n"
        "Answer the following 12 questions STRICTLY based on the provided text.\n"
        "Keep answers 1-2 words max. Return N/A if information is missing.\n"
        "Prefix each answer A1: A2: ... A12: in order.\n\n"
    )
    for i, q in enumerate(QUESTIONS, 1):
        prompt += f"{i}. {q}\nA{i}:\n"
    raw = _gpt4o(client, prompt, max_tokens=350)
    answers = []
    for i in range(12):
        line = next((l for l in raw.splitlines() if l.strip().startswith(f"A{i+1}:")), "")
        answers.append(line.replace(f"A{i+1}:", "").strip() or "N/A")
    return answers


# ---------------------------------------------------------------------------
# Step 2a: Normalize our predictions via GPT-4o
# ---------------------------------------------------------------------------

def normalize_via_gpt(client, answers):
    """Return list of lists with canonical category values for 12 answers."""
    prompt = f"""
You are a medical QA expert. Map each of the 12 answers below to its canonical category.

Expected distributions:
- Answer 1: landmark  (colon/cecum/pylorus/z-line/n/a)
- Answers 2,5: color  (red/pink/orange/yellow/brown/blue/green/white/dark/n/a)
- Answers 3,6: location (center/left/right/top/bottom/surrounding/background/n/a)
- Answers 4,8,9,10,11,12: yes_no (yes/no/n/a)
- Answer 7: polyp  (zero/single/multiple/n/a)

Categories for reference: {json.dumps(CATEGORIES)}

Answers: {answers}

Return ONLY a Python list of lists, one inner list per answer, e.g.:
[["cecum"], ["pink"], ["n/a"], ["yes"], ["blue-green"], ["center"], ["zero"], ["no"], ["no"], ["no"], ["no"], ["no"]]
No code blocks, no extra text.
"""
    raw = _gpt4o(client, prompt, max_tokens=300)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and all(isinstance(x, list) for x in parsed):
            return parsed
    except Exception:
        pass
    print(f"  [parse fallback] raw: {raw[:120]}")
    return [["n/a"]] * 12


# ---------------------------------------------------------------------------
# Step 2b: Normalize ground truth locally (no extra GPT-4o calls)
# ---------------------------------------------------------------------------

def _local_normalize(answer, q_idx):
    """Map a canonical GT answer string to a chosen-answer list using local lookup."""
    cat_name = Q_CATEGORY.get(q_idx, "yes_no")
    cat = CATEGORIES[cat_name]
    a_lower = answer.lower().strip().rstrip(".")
    for canon, variants in cat.items():
        if a_lower in [v.lower().strip().rstrip(".") for v in variants]:
            return [canon]
    return ["n/a"]


def normalize_gt_vqa_local(raw_vqa):
    """
    Convert VQA_format_testset_only.json {'image_id': [{'question', 'answer'}]}
    to {'image_id': [{'question', 'answer', 'chosen answer'}]} using local lookup.
    GT answers are already canonical strings so GPT-4o is not needed here.
    """
    out = {}
    for img_id, qa_list in raw_vqa.items():
        out[img_id] = []
        for i, qa in enumerate(qa_list):
            out[img_id].append({
                "question":      qa["question"],
                "answer":        qa["answer"],
                "chosen answer": _local_normalize(qa["answer"], i + 1),
            })
    return out


# ---------------------------------------------------------------------------
# Step 3: QAAS accuracy
# ---------------------------------------------------------------------------

def compute_qaas(pred_parsed, gt_parsed):
    total_correct = total_q = 0
    for img_id, pred_qa in pred_parsed.items():
        if img_id not in gt_parsed:
            continue
        for p_qa, g_qa in zip(pred_qa, gt_parsed[img_id]):
            p_ans = set(p_qa.get("chosen answer", []))
            g_ans = set(g_qa.get("chosen answer", []))
            if not g_ans:
                continue
            total_correct += 1 if (p_ans & g_ans) else 0
            total_q       += 1
    acc = total_correct / total_q * 100 if total_q else 0.0
    return acc, total_correct, total_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="results/step4/predictions.json")
    p.add_argument("--vqa_gt",
                   default="../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/VQA_format_testset_only.json")
    p.add_argument("--output_dir", default="results/step4/qaas")
    p.add_argument("--api_key",    default=os.environ.get("OPENAI_API_KEY", ""))
    return p.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY env var or pass --api_key.")

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    preds      = load_json(args.predictions)
    raw_gt_vqa = load_json(args.vqa_gt)

    # ---- Step 1: caption → raw VQA answers ----
    vqa_raw_path = out / "pred_vqa_raw.json"
    if vqa_raw_path.exists():
        print("Loading cached pred_vqa_raw.json ...")
        pred_vqa_raw = load_json(vqa_raw_path)
    else:
        print(f"Step 1/3: caption → VQA answers via GPT-4o ({len(preds)} images) ...")
        pred_vqa_raw = {}
        for i, entry in enumerate(preds, 1):
            img_id = entry["image_path"]
            answers = caption_to_vqa(client, entry["response"])
            pred_vqa_raw[img_id] = [
                {"question": q, "answer": a}
                for q, a in zip(QUESTIONS, answers)
            ]
            if i % 20 == 0:
                print(f"  {i}/{len(preds)} done ...")
                save_json(vqa_raw_path, pred_vqa_raw)
        save_json(vqa_raw_path, pred_vqa_raw)
        print(f"  Saved {vqa_raw_path}")

    # ---- Step 2a: normalize our answers via GPT-4o ----
    pred_parsed_path = out / "pred_vqa_parsed.json"
    if pred_parsed_path.exists():
        print("Loading cached pred_vqa_parsed.json ...")
        pred_parsed = load_json(pred_parsed_path)
    else:
        print(f"Step 2/3: normalizing prediction answers via GPT-4o ({len(pred_vqa_raw)} images) ...")
        pred_parsed = {}
        for i, (img_id, qa_list) in enumerate(pred_vqa_raw.items(), 1):
            raw_answers = [qa["answer"] for qa in qa_list]
            chosen = normalize_via_gpt(client, raw_answers)
            pred_parsed[img_id] = [
                {"question": qa["question"], "given answer": qa["answer"], "chosen answer": c}
                for qa, c in zip(qa_list, chosen)
            ]
            if i % 20 == 0:
                print(f"  {i}/{len(pred_vqa_raw)} normalized ...")
                save_json(pred_parsed_path, pred_parsed)
        save_json(pred_parsed_path, pred_parsed)
        print(f"  Saved {pred_parsed_path}")

    # ---- Step 2b: normalize GT answers locally (free) ----
    gt_parsed_path = out / "gt_vqa_parsed.json"
    if gt_parsed_path.exists():
        print("Loading cached gt_vqa_parsed.json ...")
        gt_parsed = load_json(gt_parsed_path)
    else:
        print("Step 2b: normalizing ground truth locally (no GPT-4o) ...")
        gt_parsed = normalize_gt_vqa_local(raw_gt_vqa)
        save_json(gt_parsed_path, gt_parsed)
        print(f"  Saved {gt_parsed_path}")

    # ---- Step 3: QAAS ----
    acc, correct, total_q = compute_qaas(pred_parsed, gt_parsed)
    result = {"QAAS_%": acc, "correct": correct, "total_questions": total_q}
    save_json(out / "qaas_results.json", result)

    print("\n" + "=" * 52)
    print(f"  QAAS: {acc:.2f}%  ({correct}/{total_q} questions correct)")
    print("=" * 52)
    print("\nPaper reference (Table 2, Qwen2-VL backbone):")
    print("  Hallucination-aware FT:  90.89%  ← our training recipe")
    print("  Standard FT:             83.07%")
    print("  No FT (zero-shot):       ~75%    (approx)")
    print(f"\n  Our Mobile-O (hal-aware FT): {acc:.2f}%")


if __name__ == "__main__":
    main()
