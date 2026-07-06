# Evaluation: How We Benchmark Our Finetuned Mobile-O

This document explains **what we are evaluating, how the paper evaluates it, how we replicate that
evaluation, and what the numbers mean** — step by step, with concrete examples from the actual data.

---

## 1. What Is the Task We Are Evaluating?

The Gut-VLM paper trains a model to do **hallucination-aware caption correction** on gastrointestinal
endoscopy images. Here is exactly what that means:

1. A VLM (a different vision-language model, not ours) is shown an endoscopy image and writes a
   short medical caption. These AI-generated captions contain **hallucinations** — sentences that
   are factually wrong, e.g., identifying the wrong organ.

2. A **human expert** then reads the same image and annotates which sentences in the AI caption are
   hallucinated and which are correct. They also write the corrected version of the caption.

3. Our job is to **train a model that, given the image + the hallucinated AI caption, can (a) identify
   which sentences are wrong and (b) produce a corrected caption.**

### Concrete Example (from the actual test set)

**Image:** `389c76a0-...jpg` (a colon endoscopy image)

**Original AI caption (hallucinated input):**
```
An image of pylorus. The abnormality is blue-green in color, and occupies a central
position in the image. An anatomical landmark is visible, characterized by dark blue
coloring. No polyps, instruments, signs of inflammation, infection, bleeding, or
foreign bodies are present in the image.
```

**Human annotation:**
| Sentence | Label |
|---|---|
| "An image of pylorus." | ❌ HALLUCINATED (it is actually a colon, not pylorus) |
| "The abnormality is blue-green..." | ✅ correct |
| "An anatomical landmark is visible..." | ✅ correct |
| "No polyps, instruments..." | ✅ correct |

**Gold corrected caption (ground truth):**
```
An image of colon. The abnormality is blue-green in color, and occupies a central
position in the image. An anatomical landmark is visible, characterized by dark blue
coloring. No polyps, instruments, signs of inflammation, infection, bleeding, or
foreign bodies are present in the image.
```
*(Only one word changed: "pylorus" → "colon")*

### What Our Model Outputs (Two Turns)

**Turn 1 — Detection:**
```
An image of pylorus. <hallucinated>
The abnormality is blue-green in color, and occupies a central position in the image. <non-hallucinated>
An anatomical landmark is visible, characterized by dark blue coloring. <non-hallucinated>
No polyps, instruments, signs of inflammation, infection, bleeding, or foreign bodies are present in the image. <non-hallucinated>
```

**Turn 2 — Correction:**
```
Modified caption: An image of colon. The abnormality is blue-green in color, and
occupies a central position in the image. An anatomical landmark is visible,
characterized by dark blue coloring. No polyps, instruments, signs of inflammation,
infection, bleeding, or foreign bodies are present in the image.
```

The correction from Turn 2 is what gets evaluated against the gold corrected caption.

### Test Set Scale

- **366 images** (the paper's official test split, never used in training)
- **1,330 total sentences** across those images
- **359 sentences hallucinated** (27% of all sentences)
- **971 sentences correct** (73%)

So roughly 1 in 4 sentences in any given caption is hallucinated.

---

## 2. The Evaluation Metrics

The paper uses **six metrics** across two categories. We implement all of them.

---

### Category A: Text Quality Metrics (ROUGE-L, BLEU, METEOR)

These metrics measure how similar our corrected caption is to the gold correction,
purely as text strings. They do NOT understand medical meaning — they just look at
word overlap.

#### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence)

ROUGE-L finds the **longest common subsequence** of words between our output and the reference.
It rewards getting the words in the same order, even if there are other words in between.

**Example:**
- Our output:   `"An image of colon. The abnormality is blue-green..."`
- Gold:         `"An image of colon. The abnormality is blue-green..."`
- ROUGE-L ≈ 1.0 (near-perfect match)

If we output `"An image of pylorus."` (wrong, same as input):
- Our output:   `"An image of pylorus."`
- Gold:         `"An image of colon."`
- Common words: "An image of" → LCS = 3 words
- ROUGE-L ≈ 0.60 (partial match on the non-hallucinated words)

**Score range:** 0.0 (nothing in common) to 1.0 (perfect match). Higher is better.
**Typical good scores** in medical report generation: 0.60–0.85.

---

#### BLEU-1, BLEU-2, BLEU-4 (Bilingual Evaluation Understudy)

BLEU counts how many **n-grams** (sequences of 1, 2, or 4 consecutive words) from our
output appear in the reference. It adds a penalty if our output is shorter than the reference.

- **BLEU-1** counts single-word matches ("colon" in both → +1)
- **BLEU-2** counts two-word matches ("image of", "of colon" → +1 each)
- **BLEU-4** counts four-word matches ("An image of colon" → +1)

BLEU-4 is the strictest because entire 4-word chunks must match exactly. It is most
sensitive to errors in phrasing.

**Score range:** 0.0 to 1.0 (reported as percentages). Higher is better.
If the model outputs near-identical text to the gold standard, all BLEU scores will be high.
If the model rephrases correctly (same meaning, different words), BLEU may be low even
when the correction is medically right — this is a known weakness of BLEU.

---

#### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

METEOR is similar to BLEU but also considers:
- **Stemming** ("corrected" matches "correct")
- **Synonyms** (via WordNet)
- **Word order** (penalizes scrambled words)

It tends to correlate better with human judgment than BLEU for short medical texts.

**Score range:** 0.0 to 1.0. Higher is better.

---

### Category B: Medical Knowledge Metrics (QAAS and R-Sim)

These two metrics go beyond word overlap and check whether the model understood the
**clinical content** correctly. Both use GPT-4o as a judge.

---

### QAAS — Question Answering Accuracy Score (Primary Metric)

This is the **main metric** in the paper. The key insight: rather than comparing the
entire corrected caption as text, QAAS extracts 12 specific clinical facts from the
caption and checks if they are correct.

#### The 12 Questions

For every image, GPT-4o answers these 12 questions about the corrected caption:

| # | Question | Type | Example answer |
|---|---|---|---|
| 1 | Which organ? (colon/cecum/pylorus/z-line) | landmark | `colon` |
| 2 | Color of the landmark? | color | `dark blue` |
| 3 | Position of the landmark? | location | `N/A` |
| 4 | Is there an abnormality? | yes/no | `Yes` |
| 5 | Color of the abnormality? | color | `blue-green` |
| 6 | Position of the abnormality? | location | `central` |
| 7 | How many polyps? (Zero/Single/Multiple) | polyp count | `Zero` |
| 8 | Any instruments visible? | yes/no | `No` |
| 9 | Signs of inflammation? | yes/no | `No` |
| 10 | Evidence of bleeding? | yes/no | `No` |
| 11 | Any foreign bodies? | yes/no | `No` |
| 12 | Signs of infection? | yes/no | `No` |

These 12 questions are applied to **both** our corrected caption (model output) and the
gold corrected caption (ground truth). Then the answers are compared.

#### The QAAS Pipeline: 4 Steps

```
Our corrected caption
        │
        ▼
Step 1: GPT-4o extracts 12 raw answers
        │
        ▼
Step 2: GPT-4o normalizes answers to canonical categories
        │   ("dark blue" → "blue", "blue-green" → "blue",
        │    "central" → "center", "colon" → "colon")
        ▼
        Our normalized answers
        [["colon"], ["blue"], ["n/a"], ["yes"], ["blue"], ["center"],
         ["zero"], ["no"], ["no"], ["no"], ["no"], ["no"]]
        │
        ▼
Step 3: Compare with ground truth normalized answers
        GT: [["colon"], ["blue"], ["n/a"], ["yes"], ["blue"], ["center"],
             ["zero"], ["no"], ["no"], ["no"], ["no"], ["no"]]
        │
        ▼
Step 4: Count matches → accuracy
```

#### Why This Is a Better Metric Than BLEU

Consider these two corrected captions for the pylorus/colon example:
- **Our output:** `"An image of colon. The abnormality is blue-green..."`
- **Alternative phrasing:** `"The image shows the colon region. A blue-green abnormality is seen centrally..."`

Both are medically CORRECT corrections (same facts), but BLEU-4 would score the second
one much lower because the phrasing is different. QAAS scores both the same: Q1=colon ✓,
Q5=blue ✓, etc.

#### Answer Normalization — Why It Is Needed

GPT-4o might extract "dark blue" for Q2, while the gold answer is "blue". The normalization
step maps both to the canonical category "blue" before comparing, so they count as a match.

| Raw Answer | Canonical Category |
|---|---|
| "dark blue", "navy blue", "bluish", "blue-green" | `blue` |
| "normal", "flesh", "pinkish", "typical color" | `pink` |
| "center", "middle", "centrally" | `center` |
| "Yes", "yes", "yes." | `yes` |
| "N/A", "none", "not mentioned" | `n/a` |

For our model's predictions: we use GPT-4o for both steps (extraction + normalization).
For the ground truth: the answers in `VQA_format_testset_only.json` are already canonical
strings like "cecum", "No", "Zero" — so we do the normalization **locally** with a lookup
table (no extra GPT-4o calls needed).

#### Scoring

```
QAAS = (number of questions answered correctly across all 366 images) 
       ÷ (total questions across all 366 images) × 100
     = correct / (366 × 12) × 100        [approximately]
```

If a gold answer is `N/A`, that question is skipped (the model can't be wrong about
something the caption doesn't mention). So the denominator is slightly less than 366 × 12.

**Paper results (Table 2):**
| Model | QAAS |
|---|---|
| Hallucination-aware FT (Qwen2-VL, 7B) | **90.89%** |
| Standard FT (Qwen2-VL, 7B) | 83.07% |
| No FT zero-shot | ~75% |

Our Mobile-O is a **0.5B parameter** model (14× smaller than the paper's 7B Qwen2-VL).
We should expect a lower number, but the important comparison is: does our hallucination-
aware finetuning give a bigger improvement over no-finetuning than the 7.82 percentage
point gap the paper shows (90.89 - 83.07)?

---

### R-Sim — Report Similarity Score

R-Sim asks GPT-4o to directly compare our corrected caption to the gold corrected caption
and rate them on a 1–5 scale. It is a holistic clinical similarity score.

**The rating scale:**
| Score | Label | Meaning |
|---|---|---|
| 5 | Very Good | Nearly identical, all findings correctly described |
| 4 | Good | Minor differences, clinically acceptable |
| 3 | Alright | Some differences, but overall meaning preserved |
| 2 | Not Good | Significant differences affecting diagnosis |
| 1 | Poor | Completely incorrect or misleading |

GPT-4o is prompted to compare the two reports focusing on:
- Which anatomical landmark (colon, cecum, pylorus, z-line)
- Color and location of the landmark and any abnormalities
- Presence of polyps, instruments, inflammation, bleeding, infection, foreign bodies

Then all per-image scores are averaged to get the final R-Sim score.

**For the pylorus/colon example:**
- If our model correctly changes "pylorus" to "colon" (and everything else is unchanged):
  → GPT-4o rates this 5/5 (nearly identical after correction)
- If our model leaves "pylorus" unchanged (failed to detect/correct):
  → GPT-4o rates this 1-2/5 (wrong organ, clinically significant error)

---

### Category C: Detection Metrics (Bonus — not in paper's main table)

We also compute **sentence-level hallucination detection F1** — how accurately our Turn 1
output (the `<hallucinated>` / `<non-hallucinated>` tags) matches the ground truth annotations.

This is computed locally with no API key by:
1. Parsing ground truth: reading the `annotations` field in `test.json` (per-span labels)
2. Parsing model output: extracting `<hallucinated>` / `<non-hallucinated>` tags from Turn 1
3. Aligning by position and computing precision, recall, F1 as binary classification

**Why it matters:** A model could produce a perfect corrected caption (high QAAS/R-Sim)
even if its detection tags are noisy — it might "get lucky" in the correction step. Detection
F1 tells us if the model actually understands *which* sentences were wrong, not just how to
fix them.

---

## 3. How We Generate Our Numbers

### Step 1 — Run Inference on Clariden (GPU Required)

```bash
sbatch run_step4.sh
# or interactively:
cd ~/Mobile-O
python step4_generate_predictions.py \
    --model_path checkpoints/vlm_gutvlm_hal/epoch_4 \
    --test_json  ../Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json \
    --images_dir kvasir-v2-flat \
    --output_dir results/step4 \
    --resume
```

For each of the 366 test images, the script:
1. Loads the image from `kvasir-v2-flat/uuid.jpg`
2. Runs `detect_hallucinations(model, tokenizer, ip, image, original_text)` from `inference.py`
   - This makes **two forward passes**: detection turn + correction turn
3. Saves the corrected caption (Turn 2 output) to `predictions.json`
4. Saves the detection tags (Turn 1 output) to `detections.json`
5. Writes the gold corrections to `groundtruth.json`

**Expected runtime:** ~1.5–2 hours on GH200 (366 images × ~15 sec each).
The script saves after every image, so Ctrl-C + `--resume` recovers without losing work.

**Output files in `Mobile-O/results/step4/`:**

```
predictions.json   ← our corrected captions
[
  {"image_path": "389c76a0-...jpg", "response": "An image of colon. The abnormality..."},
  ...  (366 entries)
]

groundtruth.json   ← gold corrections (same content as test.json corrections field)
[
  {"images": ["389c76a0-...jpg"], "response": "An image of colon. The abnormality..."},
  ...
]

detections.json    ← our sentence-level detection tags
[
  {"image_path": "389c76a0-...jpg",
   "detection": "An image of pylorus. <hallucinated>\nThe abnormality... <non-hallucinated>\n...",
   "original_text": "An image of pylorus. The abnormality..."},
  ...
]
```

---

### Step 2 — Local Metrics (No API Needed)

After downloading the 3 JSON files from Clariden:

```bash
pip install rouge-score sacrebleu nltk
python Mobile-O/step4_eval_local.py
```

This script:
1. Aligns `predictions.json` with `groundtruth.json` by image ID
2. Runs ROUGE-L scorer on all 366 (prediction, reference) pairs
3. Runs sacrebleu corpus BLEU-1/2/4
4. Runs NLTK METEOR
5. Parses detection tags and computes F1

All computations are local Python — no internet, no API key, no GPU.

---

### Step 3 — QAAS (OpenAI API Required, ~$11)

```bash
export OPENAI_API_KEY=sk-...
python Mobile-O/step4_eval_qaas.py
```

This runs the 4-step pipeline described above. Three intermediate files are cached, so
if the script crashes halfway through you just re-run it and it continues from where it
stopped — GPT-4o calls already made are not repeated.

```
results/step4/qaas/
├── pred_vqa_raw.json      ← Step 1: raw 12-answer extractions for our predictions
├── pred_vqa_parsed.json   ← Step 2: normalized answers with "chosen answer" lists
├── gt_vqa_parsed.json     ← Step 2b: normalized GT answers (done locally, free)
└── qaas_results.json      ← Final: {"QAAS_%": 87.3, "correct": 3800, "total_questions": 4354}
```

---

### Step 4 — R-Sim (OpenAI API Required, ~$5)

```bash
export OPENAI_API_KEY=sk-...
python Mobile-O/step4_eval_rsim.py
```

Makes one GPT-4o call per image (366 calls total). Also cached per-image so resumable.

```
results/step4/rsim/
└── rsim_results.json   ← [{"image": "uuid.jpg", "score": 4}, ..., {"average_score": 3.8}]
```

---

## 4. Comparison Table We Are Building

This is the table we will fill in after running the evaluation:

```
============================================================
Metric                    Paper (7B)   Ours (0.5B)
------------------------------------------------------------
ROUGE-L                      ?            ?
BLEU-1                       ?            ?
BLEU-2                       ?            ?
BLEU-4                       ?            ?
METEOR                       ?            ?
------------------------------------------------------------
QAAS (%)               90.89%           ?        ← primary
R-Sim (1–5)                  ?            ?
------------------------------------------------------------
Detection Precision          -            ?        ← bonus
Detection Recall             -            ?
Detection F1                 -            ?
============================================================
```

The paper only reports QAAS and R-Sim in Table 2 (the text metrics are sometimes in
supplementary material). The key number to match/beat is **90.89% QAAS** for
hallucination-aware FT.

---

## 5. What Numbers Should We Expect?

### Why Our Model Is Smaller

The paper uses **Qwen2-VL-7B** (7 billion parameters) as the backbone. We use
**Mobile-O-0.5B** (500 million parameters) — 14× fewer parameters. Smaller models
generally score lower on all metrics because they have less capacity to memorize
medical terminology and reason about subtle differences.

However, we are evaluating the same exact task with the same exact data, so the
comparison is fair in the sense that any gap reflects model capacity, not evaluation design.

### Rough Expectations

| Scenario | QAAS Expected |
|---|---|
| Model just repeats the input unchanged | ~75% (the non-hallucinated sentences are already correct, only ~27% are wrong) |
| Standard FT without hallucination awareness | ~78–83% |
| Hallucination-aware FT (what we did) | ~82–88% |
| Paper's 7B model with hal-aware FT | 90.89% |

The **baseline you should always compare against** is the "repeat input" baseline (~75%):
if our model scores below that, something is very wrong. Any score above ~83% means
our hallucination-aware finetuning is working as intended.

### Why Detection F1 Might Be Imperfect

The detection F1 aligns predicted sentence tags with ground truth by position (1st predicted
tag with 1st annotated sentence, 2nd with 2nd, etc.). If the model outputs a different
number of sentences than the ground truth (e.g., it splits or merges sentences), the
alignment will be off and F1 will be artificially low. This is a known limitation of
position-based alignment. A perfect detection score is NOT required for a good QAAS score —
what matters is that the final corrected caption (Turn 2) is right, regardless of whether
the Turn 1 tags were perfectly formatted.

---

## 6. Quick Reference: Files and Commands

| What | Where | Command |
|---|---|---|
| Inference script | `Mobile-O/step4_generate_predictions.py` | `sbatch run_step4.sh` |
| SLURM script | `run_step4.sh` | upload to Clariden |
| Local metrics | `Mobile-O/step4_eval_local.py` | `python step4_eval_local.py` |
| QAAS pipeline | `Mobile-O/step4_eval_qaas.py` | needs `OPENAI_API_KEY` |
| R-Sim pipeline | `Mobile-O/step4_eval_rsim.py` | needs `OPENAI_API_KEY` |
| Test annotations | `Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/test.json` | 366 images |
| GT VQA answers | `Hallucination-Aware-VLM/dataset/Gut-VLM/train_test_split/VQA_format_testset_only.json` | used by QAAS |

### Minimal Workflow (Free — No API Key)

1. Run inference on Clariden → download `predictions.json`, `groundtruth.json`, `detections.json`
2. `pip install rouge-score sacrebleu nltk`
3. `python Mobile-O/step4_eval_local.py` → gets ROUGE-L, BLEU, METEOR, detection F1

### Full Workflow (With OpenAI)

All steps above, plus:
4. `python Mobile-O/step4_eval_qaas.py` → gets QAAS %
5. `python Mobile-O/step4_eval_rsim.py` → gets R-Sim 1–5
